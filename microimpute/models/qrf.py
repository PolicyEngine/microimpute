"""Quantile Regression Forest imputation model with sequential imputation."""

import gc
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import validate_call
from quantile_forest import RandomForestQuantileRegressor
from sklearn.ensemble import RandomForestClassifier

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


class _RandomForestClassifierModel:
    """Internal class to handle classification for categorical/boolean targets."""

    def __init__(self, seed: int, logger):
        self.seed = seed
        self.logger = logger
        self.classifier = None
        self.output_column = None
        self.var_type = None
        self.categories = None
        self.label_map = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        var_type: str,
        categories: List = None,
        **rf_kwargs: Any,
    ) -> None:
        """Fit classifier for categorical/boolean target.

        Note: y should be the ORIGINAL categorical/boolean column,
        not dummy encoded.
        """
        self.output_column = y.name
        self.var_type = var_type

        if var_type == "boolean":
            # For boolean, convert to 0/1 but keep as single target
            y_encoded = y.astype(int)
            self.categories = [False, True]
        else:
            # For categorical, create label encoding
            self.categories = categories if categories else y.unique().tolist()
            self.label_map = {cat: i for i, cat in enumerate(self.categories)}
            y_encoded = y.map(self.label_map)

            # Check for unmapped values
            if y_encoded.isna().any():
                self.logger.warning(
                    f"Found {y_encoded.isna().sum()} unmapped values in {self.output_column}"
                )
                y_encoded = y_encoded.fillna(0)  # Default to first category

        # Extract relevant RF parameters from kwargs
        classifier_params = {
            "n_estimators": rf_kwargs.get("n_estimators", 100),
            "max_depth": rf_kwargs.get("max_depth", None),
            "min_samples_split": rf_kwargs.get("min_samples_split", 2),
            "min_samples_leaf": rf_kwargs.get("min_samples_leaf", 1),
            "max_features": rf_kwargs.get("max_features", "sqrt"),
            "random_state": self.seed,
        }

        self.classifier = RandomForestClassifier(**classifier_params)
        self.classifier.fit(X, y_encoded)

    def predict(
        self, X: pd.DataFrame, return_probs: bool = False
    ) -> pd.Series:
        """Predict classes or probabilities."""
        if return_probs:
            probs = self.classifier.predict_proba(X)
            # Return both probabilities and the original category labels
            # The probabilities are ordered according to self.classifier.classes_
            # which are the encoded values, but we need to return the original labels
            # in the same order

            if self.var_type == "boolean":
                # For boolean, classes are simply False and True
                # sklearn's classifier.classes_ will be [0, 1] in order
                original_classes = [False, True]
            else:
                # For categorical, map encoded values back to original labels
                original_classes = []
                for encoded_val in self.classifier.classes_:
                    # Find the original category for this encoded value
                    for cat, enc in self.label_map.items():
                        if enc == encoded_val:
                            original_classes.append(cat)
                            break

            return {
                "probabilities": probs,
                "classes": np.array(original_classes),
            }
        else:
            y_pred = self.classifier.predict(X)

            if self.var_type == "boolean":
                predictions = pd.Series(y_pred.astype(bool), index=X.index)
            else:
                # Map back to original categories
                reverse_map = {i: cat for cat, i in self.label_map.items()}
                predictions = pd.Series(y_pred).map(reverse_map)
                predictions.index = X.index

            predictions.name = self.output_column
            return predictions


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
        models: Dict[
            str, Any
        ],  # Can be _QRFModel, _RandomForestClassifierModel, or _ConstantValueModel
        predictors: List[str],
        imputed_variables: List[str],
        seed: int,
        imputed_vars_dummy_info: Optional[Dict[str, str]] = None,
        original_predictors: Optional[List[str]] = None,
        categorical_targets: Optional[Dict[str, Dict]] = None,
        boolean_targets: Optional[Dict[str, Dict]] = None,
        constant_targets: Optional[Dict[str, Dict]] = None,
        dummy_processor: Optional[Any] = None,
        log_level: Optional[str] = "WARNING",
    ) -> None:
        """Initialize the QRF results.

        Args:
            models: Dictionary of fitted models (QRF or RF classifier) for each variable.
            predictors: List of column names used as predictors.
            imputed_variables: List of column names to be imputed.
            seed: Random seed for reproducibility.
            imputed_vars_dummy_info: Optional dictionary containing information
                about dummy variables for imputed variables.
            original_predictors: Optional list of original predictor variable
                names before dummy encoding.
            categorical_targets: Dictionary of categorical target info.
            boolean_targets: Dictionary of boolean target info.
            dummy_processor: Processor for handling dummy encoding in test data.
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
        self.categorical_targets = categorical_targets or {}
        self.boolean_targets = boolean_targets or {}
        self.constant_targets = constant_targets or {}
        self.dummy_processor = dummy_processor

    @validate_call(config=VALIDATE_CONFIG)
    def _predict(
        self,
        X_test: pd.DataFrame,
        quantiles: Optional[List[float]] = None,
        mean_quantile: Optional[float] = 0.5,
        return_probs: bool = False,
    ) -> Dict[float, pd.DataFrame]:
        """Predict values at specified quantiles using the QRF model.

        Args:
            X_test: DataFrame containing the test data.
            quantiles: List of quantiles to predict (the quantile affects the
                center of the beta distribution from which to sample when imputing each data point).
            mean_quantile: The mean quantile to used for prediction if
                quantiles are not provided.
            return_probs: If True, return probability distributions for categorical variables.

        Returns:
            Dictionary mapping quantiles to predicted values.
            If return_probs=True, includes 'probabilities' key.

        Raises:
            RuntimeError: If prediction fails.
        """
        try:
            # Create output dictionary with results
            imputations: Dict[float, pd.DataFrame] = {}
            prob_results = {} if return_probs else None

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
                self.logger.debug(
                    f"X_test columns at start of _predict: {X_test_augmented.columns.tolist()}"
                )

                # Track dummy columns created from imputed categorical variables
                imputed_dummy_cols = set()

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
                    self.logger.debug(
                        f"var_predictors for {variable}: {var_predictors}"
                    )
                    self.logger.debug(
                        f"Available columns in X_test_augmented: {X_test_augmented.columns.tolist()}"
                    )

                    # Ensure we have all needed columns in X_test_augmented
                    missing_cols = set(var_predictors) - set(
                        X_test_augmented.columns
                    )
                    if missing_cols:
                        # Check if these are dummy columns from previously imputed categorical variables
                        imputed_missing = missing_cols & imputed_dummy_cols

                        if imputed_missing:
                            self.logger.debug(
                                f"Adding zero-filled columns for missing categories "
                                f"from imputed variables: {imputed_missing}"
                            )
                            # Add zeros for dummy columns from imputed categoricals
                            for col in imputed_missing:
                                X_test_augmented[col] = 0.0

                        # Any other missing columns will cause an error when we try to select them,
                        # which is the desired behavior to alert the user of missing predictors

                    # Import constant model
                    from microimpute.models.imputer import _ConstantValueModel

                    # Predict using the appropriate predictor set
                    if isinstance(model, _ConstantValueModel):
                        # Constant model - just return the constant value
                        imputed_values = model.predict(X_test_augmented)
                    elif isinstance(model, _RandomForestClassifierModel):
                        # Classification for categorical/boolean targets
                        if return_probs and prob_results is not None:
                            # Get probabilities and classes
                            prob_info = model.predict(
                                X_test_augmented[var_predictors],
                                return_probs=True,
                            )
                            prob_results[variable] = prob_info

                        # Get class predictions
                        imputed_values = model.predict(
                            X_test_augmented[var_predictors],
                            return_probs=False,
                        )
                    else:
                        # Regression for numeric targets
                        imputed_values = model.predict(
                            X_test_augmented[var_predictors], mean_quantile=q
                        )

                    imputed_df[variable] = imputed_values

                    # Add the imputed values to X_test_augmented for subsequent variables
                    X_test_augmented[variable] = imputed_values

                    # If this is a categorical variable, track its dummy columns
                    # for future sequential imputation steps
                    if variable in self.categorical_targets:
                        # Track which dummy columns would be created for this variable
                        # using drop_first=True convention
                        unique_values = imputed_values.unique()
                        if len(unique_values) > 1:
                            # With drop_first=True, we create dummies for all but the first category
                            for val in sorted(unique_values)[1:]:
                                dummy_col = f"{variable}_{val}"
                                imputed_dummy_cols.add(dummy_col)
                                # Also create the actual dummy column if it will be used
                                X_test_augmented[dummy_col] = (
                                    imputed_values == val
                                ).astype(float)

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

            # Add probabilities to results if requested
            if return_probs and prob_results:
                imputations["probabilities"] = prob_results

            qs = [k for k in imputations.keys() if k != "probabilities"]
            if len(qs) < 2:
                q = list(qs)[0]

            # If quantiles not provided, decide what to return based on return_probs
            if not quantiles:
                if return_probs and prob_results:
                    # Return dict with both quantile predictions and probabilities
                    return imputations
                else:
                    # Return just the DataFrame for the single quantile
                    return imputations[q]
            else:
                # Multiple quantiles requested, return the full dict
                return imputations

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

    def _create_model_for_variable(self, variable: str, **kwargs) -> Any:
        """Create the appropriate model (classifier or regressor) based on variable type."""
        categorical_targets = getattr(self, "categorical_targets", {})
        boolean_targets = getattr(self, "boolean_targets", {})

        if variable in categorical_targets:
            # Use classifier for categorical targets
            return _RandomForestClassifierModel(
                seed=self.seed, logger=self.logger
            )
        elif variable in boolean_targets:
            # Use classifier for boolean targets
            return _RandomForestClassifierModel(
                seed=self.seed, logger=self.logger
            )
        else:
            # Use QRF for numeric targets
            return _QRFModel(seed=self.seed, logger=self.logger)

    def _fit_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        variable: str,
        **kwargs,
    ) -> None:
        """Fit the model with appropriate parameters based on variable type."""
        categorical_targets = getattr(self, "categorical_targets", {})
        boolean_targets = getattr(self, "boolean_targets", {})

        if isinstance(model, _RandomForestClassifierModel):
            if variable in categorical_targets:
                model.fit(
                    X,
                    y,
                    var_type=categorical_targets[variable]["type"],
                    categories=categorical_targets[variable].get("categories"),
                    **kwargs,
                )
            elif variable in boolean_targets:
                model.fit(X, y, var_type="boolean", **kwargs)
        else:
            # Regular QRF fit
            model.fit(X, y, **kwargs)

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
        categorical_targets: Optional[Dict[str, Dict]] = None,
        boolean_targets: Optional[Dict[str, Dict]] = None,
        numeric_targets: Optional[List[str]] = None,
        constant_targets: Optional[Dict[str, Dict]] = None,
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
                                constant_targets,
                            )

                            # Memory cleanup after each batch
                            if self.memory_efficient:
                                gc.collect()
                                self.logger.info(
                                    f"Batch {batch_idx + 1} completed. Memory usage: {self._get_memory_usage_info()}"
                                )
                    else:
                        # Process all variables sequentially
                        # Import constant model
                        from microimpute.models.imputer import (
                            _ConstantValueModel,
                        )

                        for i, variable in enumerate(imputed_variables):
                            var_start_time = time.time()

                            # Handle constant targets
                            if variable in (constant_targets or {}):
                                constant_val = constant_targets[variable][
                                    "value"
                                ]
                                self.models[variable] = _ConstantValueModel(
                                    constant_val, variable
                                )
                                self.logger.info(
                                    f"Using constant value {constant_val} for variable {variable}"
                                )
                                continue

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

                            # Create appropriate model based on variable type
                            model = self._create_model_for_variable(variable)
                            self._fit_model(
                                model,
                                X_train[current_predictors],
                                X_train[variable],
                                variable,
                                **qrf_kwargs,
                            )

                            try:

                                # Log post-imputation information
                                var_time = time.time() - var_start_time
                                self.logger.info(
                                    f"  ✓ Success: {variable} fitted in {var_time:.2f}s"
                                )

                                # Get model complexity metrics if available
                                if hasattr(model, "qrf") and hasattr(
                                    model.qrf, "n_estimators"
                                ):
                                    self.logger.info(
                                        f"  Model complexity: {model.qrf.n_estimators} trees"
                                    )
                                elif hasattr(model, "classifier") and hasattr(
                                    model.classifier, "n_estimators"
                                ):
                                    self.logger.info(
                                        f"  Model complexity: {model.classifier.n_estimators} trees (classifier)"
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
                            categorical_targets=categorical_targets,
                            boolean_targets=boolean_targets,
                            constant_targets=constant_targets,
                            dummy_processor=getattr(
                                self, "dummy_processor", None
                            ),
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
                            constant_targets,
                        )

                        # Memory cleanup after each batch
                        if self.memory_efficient:
                            gc.collect()
                            self.logger.info(
                                f"Batch {batch_idx + 1} completed. Memory usage: {self._get_memory_usage_info()}"
                            )
                else:
                    # Process all variables sequentially
                    # Import constant model
                    from microimpute.models.imputer import _ConstantValueModel

                    # Initialize and fit a QRF model for each variable
                    for i, variable in enumerate(imputed_variables):
                        var_start_time = time.time()

                        # Handle constant targets
                        if variable in (constant_targets or {}):
                            constant_val = constant_targets[variable]["value"]
                            self.models[variable] = _ConstantValueModel(
                                constant_val, variable
                            )
                            self.logger.info(
                                f"Using constant value {constant_val} for variable {variable}"
                            )
                            continue

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
                        model = self._create_model_for_variable(variable)

                        try:
                            self._fit_model(
                                model,
                                X_train[current_predictors],
                                X_train[variable],
                                variable,
                                **qrf_kwargs,
                            )

                            # Log post-imputation information
                            var_time = time.time() - var_start_time
                            self.logger.info(
                                f"  ✓ Success: {variable} fitted in {var_time:.2f}s"
                            )

                            # Get model complexity metrics if available
                            if hasattr(model, "qrf") and hasattr(
                                model.qrf, "n_estimators"
                            ):
                                self.logger.info(
                                    f"  Model complexity: {model.qrf.n_estimators} trees"
                                )
                            elif hasattr(model, "classifier") and hasattr(
                                model.classifier, "n_estimators"
                            ):
                                self.logger.info(
                                    f"  Model complexity: {model.classifier.n_estimators} trees (classifier)"
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
                    categorical_targets=categorical_targets,
                    boolean_targets=boolean_targets,
                    constant_targets=constant_targets,
                    dummy_processor=getattr(self, "dummy_processor", None),
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
        constant_targets: Optional[Dict[str, Dict]] = None,
    ) -> None:
        """Fit models for a batch of variables.

        Args:
            X_train: Training data
            predictors: Original predictor variables
            imputed_variables: All variables being imputed
            batch_variables: Variables in current batch
            qrf_kwargs: QRF model parameters
        """
        # Import constant model
        from microimpute.models.imputer import _ConstantValueModel

        for variable in batch_variables:
            var_start_time = time.time()
            i = imputed_variables.index(variable)

            # Handle constant targets
            if variable in (constant_targets or {}):
                constant_val = constant_targets[variable]["value"]
                self.models[variable] = _ConstantValueModel(
                    constant_val, variable
                )
                self.logger.info(
                    f"Using constant value {constant_val} for variable {variable}"
                )
                continue

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
            model = self._create_model_for_variable(variable)

            try:
                self._fit_model(
                    model,
                    X_train[current_predictors],
                    X_train[variable],
                    variable,
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
                model = self._create_model_for_variable(var)
                self._fit_model(
                    model,
                    X_train_augmented[current_predictors],
                    X_train[var],
                    var,
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
