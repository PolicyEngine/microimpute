"""Quantile Regression Forest imputation model with sequential imputation."""

import gc
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import validate_call
from quantile_forest import RandomForestQuantileRegressor

from microimpute.config import VALIDATE_CONFIG
from microimpute.models.imputer import Imputer, ImputerResults

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def _get_sequential_predictors(
    predictors: List[str],
    imputed_variables: List[str],
    current_variable_index: int,
) -> List[str]:
    """Get the predictor set for sequential imputation.

    Args:
        predictors: Original predictor variables
        imputed_variables: Variables being imputed
        current_variable_index: Index of the current variable being imputed

    Returns:
        List of predictor columns including previously imputed variables
    """
    return predictors + imputed_variables[:current_variable_index]


class _QRFModel:
    """Internal class to handle QRF model with quantile prediction logic."""

    def __init__(self, seed: int, logger):
        self.seed = seed
        self.logger = logger
        self.qrf = None
        self.output_column = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **qrf_kwargs: Any) -> None:
        """Fit the QRF model.

        Note: Assumes X is already preprocessed with categorical encoding
        handled by the base Imputer class.
        """
        self.output_column = y.name

        # Create and fit model
        self.qrf = RandomForestQuantileRegressor(
            random_state=self.seed, **qrf_kwargs
        )
        self.qrf.fit(X, y.values.ravel())

    def predict(
        self,
        X: pd.DataFrame,
        mean_quantile: float = 0.5,
        count_samples: int = 10,
    ) -> pd.Series:
        """Predict using the fitted model with beta distribution sampling.

        Note: Assumes X is already preprocessed with categorical encoding
        handled by the base ImputerResults class.
        """
        # Generate quantile grid
        eps = 1.0 / (count_samples + 1)
        quantile_grid = np.linspace(eps, 1.0 - eps, count_samples)
        pred = self.qrf.predict(X, quantiles=list(quantile_grid))

        # Sample from beta distribution
        random_generator = np.random.default_rng(self.seed)
        a = mean_quantile / (1 - mean_quantile)
        input_quantiles = (
            random_generator.beta(a, 1, size=len(X)) * count_samples
        )
        input_quantiles = np.clip(
            input_quantiles.astype(int), 0, count_samples - 1
        )

        # Extract predictions
        if len(pred.shape) == 2:
            predictions = pred[np.arange(len(pred)), input_quantiles]
        else:
            predictions = pred[np.arange(len(pred)), :, input_quantiles]

        return pd.Series(predictions, index=X.index, name=self.output_column)


class QRFResults(ImputerResults):
    """
    Fitted QRF instance ready for imputation.
    """

    def __init__(
        self,
        models: Dict[str, _QRFModel],
        predictors: List[str],
        imputed_variables: List[str],
        seed: int,
        imputed_vars_dummy_info: Optional[Dict[str, str]] = None,
        original_predictors: Optional[List[str]] = None,
        log_level: Optional[str] = "WARNING",
    ) -> None:
        """Initialize the QRF results.

        Args:
            models: Dictionary of fitted QRF models for each variable.
            predictors: List of column names used as predictors.
            imputed_variables: List of column names to be imputed.
            seed: Random seed for reproducibility.
            imputed_vars_dummy_info: Optional dictionary containing information
                about dummy variables for imputed variables.
            original_predictors: Optional list of original predictor variable
                names before dummy encoding.
        """
        super().__init__(
            predictors,
            imputed_variables,
            seed,
            imputed_vars_dummy_info,
            original_predictors,
            log_level,
        )
        self.models = models

    @validate_call(config=VALIDATE_CONFIG)
    def _predict(
        self,
        X_test: pd.DataFrame,
        quantiles: Optional[List[float]] = None,
        mean_quantile: Optional[float] = 0.5,
    ) -> Dict[float, pd.DataFrame]:
        """Predict values at specified quantiles using the QRF model.

        Args:
            X_test: DataFrame containing the test data.
            quantiles: List of quantiles to predict (the quantile affects the
                center of the beta distribution from which to sample when imputing each data point).
            mean_quantile: The mean quantile to used for prediction if
                quantiles are not provided.

        Returns:
            Dictionary mapping quantiles to predicted values.

        Raises:
            RuntimeError: If prediction fails.
        """
        try:
            # Create output dictionary with results
            imputations: Dict[float, pd.DataFrame] = {}

            # Convert single mean_quantile to a list if quantiles not provided
            quantiles_to_process = quantiles if quantiles else [mean_quantile]

            if quantiles:
                self.logger.info(
                    f"Predicting at {len(quantiles)} quantiles: {quantiles}"
                )
            else:
                self.logger.info(
                    f"Predicting from a beta distribution centered at quantile: {mean_quantile:.4f}"
                )

            for q in quantiles_to_process:
                imputed_df = pd.DataFrame()
                # Create a copy of X_test that we'll augment with imputed values
                X_test_augmented = X_test.copy()

                for i, variable in enumerate(self.imputed_variables):
                    var_start_time = time.time()

                    if not quantiles:
                        self.logger.info(
                            f"[{i+1}/{len(self.imputed_variables)}] Predicting for '{variable}'"
                        )

                    model = self.models[variable]

                    # Build predictor set: original predictors + previously imputed variables
                    var_predictors = _get_sequential_predictors(
                        self.predictors, self.imputed_variables, i
                    )

                    # Ensure we have all needed columns in X_test_augmented
                    missing_cols = set(var_predictors) - set(
                        X_test_augmented.columns
                    )
                    if missing_cols:
                        self.logger.warning(
                            f"Missing columns for {variable}: {missing_cols}. "
                            "Using available columns only."
                        )
                        var_predictors = [
                            col
                            for col in var_predictors
                            if col in X_test_augmented.columns
                        ]

                    # Predict using the appropriate predictor set
                    imputed_values = model.predict(
                        X_test_augmented[var_predictors], mean_quantile=q
                    )
                    imputed_df[variable] = imputed_values

                    # Add the imputed values to X_test_augmented for subsequent variables
                    X_test_augmented[variable] = imputed_values

                    # Log timing for individual variables when not processing multiple quantiles
                    if not quantiles:
                        var_time = time.time() - var_start_time
                        self.logger.info(
                            f"  ✓ {variable} predicted in {var_time:.2f}s ({len(imputed_values)} samples)"
                        )

                    self.logger.info(
                        f"QRF predictions completed for {variable} imputed variable"
                    )

                imputations[q] = imputed_df

            qs = imputations.keys()
            if len(qs) < 2:
                q = list(qs)[0]

            return imputations if quantiles else imputations[q]

        except Exception as e:
            self.logger.error(f"Error during QRF prediction: {str(e)}")
            raise RuntimeError(
                f"Failed to predict with QRF model: {str(e)}"
            ) from e


class QRF(Imputer):
    """
    Quantile Regression Forest model for imputation.

    This model uses a Quantile Random Forest to predict quantiles.
    The underlying QRF implementation is from the quantile_forest package.
    """

    def __init__(
        self,
        log_level: Optional[str] = "WARNING",
        memory_efficient: bool = False,
        batch_size: Optional[int] = None,
        cleanup_interval: int = 10,
    ) -> None:
        """Initialize the QRF model.

        Args:
            log_level: Logging level for the imputer.
            memory_efficient: Enable memory optimization features.
            batch_size: Process variables in batches to reduce memory usage.
            cleanup_interval: Frequency of garbage collection (every N variables).
        """
        super().__init__(log_level=log_level)
        self.models = {}
        self.log_level = log_level
        self.memory_efficient = memory_efficient
        self.batch_size = batch_size
        self.cleanup_interval = cleanup_interval

        self.logger.debug("Initializing QRF imputer")

        if memory_efficient:
            self.logger.info(
                f"Memory-efficient mode enabled with cleanup_interval={cleanup_interval}"
            )
            if batch_size:
                self.logger.info(
                    f"Batch processing enabled with batch_size={batch_size}"
                )

    def _get_memory_usage_info(self) -> str:
        """Get formatted memory usage information."""
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return f"{memory_mb:.1f}MB"
        return "N/A"

    @validate_call(config=VALIDATE_CONFIG)
    def _fit(
        self,
        X_train: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        original_predictors: Optional[List[str]] = None,
        tune_hyperparameters: bool = False,
        **qrf_kwargs: Any,
    ) -> QRFResults:
        """Fit the QRF model to the training data.

        Args:
            X_train: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.
            **qrf_kwargs: Additional keyword arguments to pass to QRF.

        Returns:
            The fitted model instance.

        Raises:
            RuntimeError: If model fitting fails.
        """
        try:
            if tune_hyperparameters:
                try:
                    qrf_kwargs = self._tune_hyperparameters(
                        data=X_train,
                        predictors=predictors,
                        imputed_variables=imputed_variables,
                    )

                    # Initialize and fit a QRF model for each variable
                    self.logger.info(
                        f"Training data shape: {X_train.shape}, Memory usage: {self._get_memory_usage_info()}"
                    )

                    # Handle batch processing if enabled
                    if (
                        self.batch_size
                        and len(imputed_variables) > self.batch_size
                    ):
                        self.logger.info(
                            f"Processing {len(imputed_variables)} variables in batches of {self.batch_size}"
                        )
                        variable_batches = [
                            imputed_variables[i : i + self.batch_size]
                            for i in range(
                                0, len(imputed_variables), self.batch_size
                            )
                        ]
                        for batch_idx, batch_variables in enumerate(
                            variable_batches
                        ):
                            self.logger.info(
                                f"Processing batch {batch_idx + 1}/{len(variable_batches)} "
                                f"({len(batch_variables)} variables)"
                            )
                            self._fit_variable_batch(
                                X_train,
                                predictors,
                                imputed_variables,
                                batch_variables,
                                qrf_kwargs,
                            )

                            # Memory cleanup after each batch
                            if self.memory_efficient:
                                gc.collect()
                                self.logger.info(
                                    f"Batch {batch_idx + 1} completed. Memory usage: {self._get_memory_usage_info()}"
                                )
                    else:
                        # Process all variables sequentially
                        for i, variable in enumerate(imputed_variables):
                            var_start_time = time.time()

                            # Build predictor set: original predictors + previously imputed variables
                            current_predictors = _get_sequential_predictors(
                                predictors, imputed_variables, i
                            )

                            # Log detailed pre-imputation information
                            self.logger.info(
                                f"[{i+1}/{len(imputed_variables)}] Starting imputation for '{variable}'"
                            )
                            self.logger.info(
                                f"  Features: {len(current_predictors)} predictors"
                            )
                            self.logger.info(
                                f"  Memory usage: {self._get_memory_usage_info()}"
                            )

                            # Create and fit model
                            model = _QRFModel(
                                seed=self.seed, logger=self.logger
                            )

                            try:
                                model.fit(
                                    X_train[current_predictors],
                                    X_train[variable],
                                    **qrf_kwargs,
                                )

                                # Log post-imputation information
                                var_time = time.time() - var_start_time
                                self.logger.info(
                                    f"  ✓ Success: {variable} fitted in {var_time:.2f}s"
                                )

                                # Get model complexity metrics if available
                                if hasattr(model.qrf, "n_estimators"):
                                    self.logger.info(
                                        f"  Model complexity: {model.qrf.n_estimators} trees"
                                    )

                                self.models[variable] = model

                            except Exception as e:
                                self.logger.error(
                                    f"  ✗ Failed: {variable} - {str(e)}"
                                )
                                raise

                            # Memory cleanup if enabled
                            if (
                                self.memory_efficient
                                and (i + 1) % self.cleanup_interval == 0
                            ):
                                gc.collect()
                                self.logger.debug(
                                    f"  Memory cleanup performed. Usage: {self._get_memory_usage_info()}"
                                )

                    return (
                        QRFResults(
                            models=self.models,
                            predictors=predictors,
                            imputed_variables=imputed_variables,
                            imputed_vars_dummy_info=self.imputed_vars_dummy_info,
                            original_predictors=self.original_predictors,
                            seed=self.seed,
                        ),
                        qrf_kwargs,
                    )

                except Exception as e:
                    self.logger.error(
                        f"Error tuning hyperparameters: {str(e)}"
                    )
                    raise RuntimeError(
                        f"Failed to tune hyperparameters: {str(e)}"
                    ) from e

            else:
                self.logger.info(
                    f"Fitting QRF model with {len(predictors)} predictors and "
                    f"optional parameters: {qrf_kwargs}"
                )
                self.logger.info(
                    f"Training data shape: {X_train.shape}, Memory usage: {self._get_memory_usage_info()}"
                )

                # Handle batch processing if enabled
                if (
                    self.batch_size
                    and len(imputed_variables) > self.batch_size
                ):
                    self.logger.info(
                        f"Processing {len(imputed_variables)} variables in batches of {self.batch_size}"
                    )
                    variable_batches = [
                        imputed_variables[i : i + self.batch_size]
                        for i in range(
                            0, len(imputed_variables), self.batch_size
                        )
                    ]
                    for batch_idx, batch_variables in enumerate(
                        variable_batches
                    ):
                        self.logger.info(
                            f"Processing batch {batch_idx + 1}/{len(variable_batches)} "
                            f"({len(batch_variables)} variables)"
                        )
                        self._fit_variable_batch(
                            X_train,
                            predictors,
                            imputed_variables,
                            batch_variables,
                            qrf_kwargs,
                        )

                        # Memory cleanup after each batch
                        if self.memory_efficient:
                            gc.collect()
                            self.logger.info(
                                f"Batch {batch_idx + 1} completed. Memory usage: {self._get_memory_usage_info()}"
                            )
                else:
                    # Process all variables sequentially
                    # Initialize and fit a QRF model for each variable
                    for i, variable in enumerate(imputed_variables):
                        var_start_time = time.time()

                        # Build predictor set: original predictors + previously imputed variables
                        current_predictors = _get_sequential_predictors(
                            predictors, imputed_variables, i
                        )

                        # Log detailed pre-imputation information
                        self.logger.info(
                            f"[{i+1}/{len(imputed_variables)}] Starting imputation for '{variable}'"
                        )
                        self.logger.info(
                            f"  Features: {len(current_predictors)} predictors"
                        )
                        self.logger.info(
                            f"  Memory usage: {self._get_memory_usage_info()}"
                        )

                        # Create and fit model
                        model = _QRFModel(seed=self.seed, logger=self.logger)

                        try:
                            model.fit(
                                X_train[current_predictors],
                                X_train[variable],
                                **qrf_kwargs,
                            )

                            # Log post-imputation information
                            var_time = time.time() - var_start_time
                            self.logger.info(
                                f"  ✓ Success: {variable} fitted in {var_time:.2f}s"
                            )

                            # Get model complexity metrics if available
                            if hasattr(model.qrf, "n_estimators"):
                                self.logger.info(
                                    f"  Model complexity: {model.qrf.n_estimators} trees"
                                )

                            self.models[variable] = model

                        except Exception as e:
                            self.logger.error(
                                f"  ✗ Failed: {variable} - {str(e)}"
                            )
                            raise

                        # Memory cleanup if enabled
                        if (
                            self.memory_efficient
                            and (i + 1) % self.cleanup_interval == 0
                        ):
                            gc.collect()
                            self.logger.debug(
                                f"  Memory cleanup performed. Usage: {self._get_memory_usage_info()}"
                            )

                # Final memory cleanup if enabled
                if self.memory_efficient:
                    gc.collect()

                self.logger.info(
                    f"QRF model fitting completed. Final memory usage: {self._get_memory_usage_info()}"
                )

                return QRFResults(
                    models=self.models,
                    predictors=predictors,
                    imputed_variables=imputed_variables,
                    imputed_vars_dummy_info=self.imputed_vars_dummy_info,
                    original_predictors=self.original_predictors,
                    seed=self.seed,
                    log_level=self.log_level,
                )
        except Exception as e:
            self.logger.error(f"Error fitting QRF model: {str(e)}")
            raise RuntimeError(f"Failed to fit QRF model: {str(e)}") from e

    def _fit_variable_batch(
        self,
        X_train: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        batch_variables: List[str],
        qrf_kwargs: Dict[str, Any],
    ) -> None:
        """Fit models for a batch of variables.

        Args:
            X_train: Training data
            predictors: Original predictor variables
            imputed_variables: All variables being imputed
            batch_variables: Variables in current batch
            qrf_kwargs: QRF model parameters
        """
        for variable in batch_variables:
            var_start_time = time.time()
            i = imputed_variables.index(variable)

            # Build predictor set: original predictors + previously imputed variables
            current_predictors = _get_sequential_predictors(
                predictors, imputed_variables, i
            )

            # Log detailed pre-imputation information
            self.logger.info(
                f"[{i+1}/{len(imputed_variables)}] Starting imputation for '{variable}'"
            )
            self.logger.info(
                f"  Features: {len(current_predictors)} predictors"
            )
            self.logger.info(
                f"  Memory usage: {self._get_memory_usage_info()}"
            )

            # Create and fit model
            # Note: X_train is already preprocessed by base class
            model = _QRFModel(seed=self.seed, logger=self.logger)

            try:
                model.fit(
                    X_train[current_predictors],
                    X_train[variable],
                    **qrf_kwargs,
                )

                # Log post-imputation information
                var_time = time.time() - var_start_time
                self.logger.info(
                    f"  ✓ Success: {variable} fitted in {var_time:.2f}s"
                )

                # Get model complexity metrics if available
                if hasattr(model.qrf, "n_estimators"):
                    self.logger.info(
                        f"  Model complexity: {model.qrf.n_estimators} trees"
                    )

                self.models[variable] = model

            except Exception as e:
                self.logger.error(f"  ✗ Failed: {variable} - {str(e)}")
                raise

            # Memory cleanup if enabled
            if self.memory_efficient and (i + 1) % self.cleanup_interval == 0:
                gc.collect()
                self.logger.debug(
                    f"  Memory cleanup performed. Usage: {self._get_memory_usage_info()}"
                )

    @validate_call(config=VALIDATE_CONFIG)
    def _tune_hyperparameters(
        self,
        data: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
    ) -> Dict[str, Any]:
        """Tune hyperparameters for the QRF model using Optuna.

        Args:
            X_train: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.

        Returns:
            Dictionary of tuned hyperparameters.
        """
        import optuna
        from sklearn.model_selection import train_test_split

        # Suppress Optuna's logs during optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Create a validation split (80% train, 20% validation)
        X_train, X_test = train_test_split(
            data, test_size=0.2, random_state=self.seed
        )

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split", 2, 20
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf", 1, 10
                ),
                "max_features": trial.suggest_float("max_features", 0.1, 1.0),
                "bootstrap": trial.suggest_categorical(
                    "bootstrap", [True, False]
                ),
            }

            # Track errors for all variables
            var_errors = []

            # Create copies for augmented data
            X_train_augmented = X_train.copy()
            X_test_augmented = X_test.copy()

            # For each imputed variable
            for i, var in enumerate(imputed_variables):
                # Build predictor set: original predictors + previously imputed variables
                current_predictors = _get_sequential_predictors(
                    predictors, imputed_variables, i
                )

                # Extract target variable values
                y_test = X_test[var]

                # Create and fit QRF model with trial parameters
                # Note: X_train_augmented is already preprocessed by base class
                model = _QRFModel(seed=self.seed, logger=self.logger)
                model.fit(
                    X_train_augmented[current_predictors],
                    X_train[var],
                    **params,
                )

                # Predict and calculate error
                y_pred = model.predict(X_test_augmented[current_predictors])

                # Add predictions to augmented datasets for next variable
                X_train_augmented[var] = model.predict(
                    X_train_augmented[current_predictors]
                )
                X_test_augmented[var] = y_pred

                # Normalize error by variable's standard deviation
                std = np.std(y_test.values.flatten())
                mse = np.mean(
                    (y_pred.values.flatten() - y_test.values.flatten()) ** 2
                )
                normalized_mse = mse / (std**2) if std > 0 else mse

                var_errors.append(normalized_mse)

            # Return mean error across all variables
            return np.mean(var_errors)

        # Create and run the study
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )

        # Suppress warnings during optimization
        import os

        os.environ["PYTHONWARNINGS"] = "ignore"

        study.optimize(objective, n_trials=30)

        best_value = study.best_value
        self.logger.info(f"Lowest average normalized MSE: {best_value}")

        best_params = study.best_params
        self.logger.info(f"Best hyperparameters found: {best_params}")

        return best_params
