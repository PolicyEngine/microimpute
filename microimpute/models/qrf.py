"""Quantile Regression Forest imputation model."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import validate_call

from microimpute.config import VALIDATE_CONFIG
from microimpute.models.imputer import Imputer, ImputerResults
from microimpute.utils import qrf


class QRFResults(ImputerResults):
    """
    Fitted QRF instance ready for imputation.
    """

    def __init__(
        self,
        models: Dict[str, "QRF"],
        predictors: List[str],
        imputed_variables: List[str],
        seed: int,
        imputed_vars_dummy_info: Optional[Dict[str, str]] = None,
        original_predictors: Optional[List[str]] = None,
        log_level: Optional[str] = "WARNING",
    ) -> None:
        """Initialize the QRF results.

        Args:
            model: Fitted QRF model.
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

            if quantiles:
                self.logger.info(
                    f"Predicting at {len(quantiles)} quantiles: {quantiles}"
                )
                for q in quantiles:
                    imputed_df = pd.DataFrame()
                    for variable in self.imputed_variables:
                        model = self.models[variable]
                        imputed_df[variable] = model.predict(
                            X_test[self.predictors], mean_quantile=q
                        )
                    imputations[q] = imputed_df
            else:
                self.logger.info(
                    f"Predicting from a beta distribution centered at quantile: {mean_quantile:.4f}"
                )
                imputed_df = pd.DataFrame()
                for variable in self.imputed_variables:
                    self.logger.info(f"Imputing variable {variable}")
                    model = self.models[variable]
                    imputed_df[variable] = model.predict(
                        X_test[self.predictors], mean_quantile=mean_quantile
                    )
                imputations[mean_quantile] = imputed_df

            self.logger.info(
                f"QRF predictions completed for {len(X_test)} samples"
            )
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
    The underlying QRF implementation is from utils.qrf.
    """

    def __init__(self, log_level: Optional[str] = "WARNING") -> None:
        """Initialize the QRF model."""
        super().__init__(log_level=log_level)
        self.models = {}
        self.log_level = log_level
        self.logger.debug("Initializing QRF imputer")

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

                    # Extract training data
                    X = X_train[predictors]
                    # Initialize and fit a QRF model for each variable
                    for variable in imputed_variables:
                        model = qrf.QRF(seed=self.seed)
                        y = pd.DataFrame(X_train[variable])
                        # Fit the QRF model
                        model.fit(X, y, **qrf_kwargs)

                        self.logger.info(
                            f"QRF model fitted successfully with {len(X)} training samples"
                        )

                        self.models[variable] = model
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

                # Extract training data
                X = X_train[predictors]
                # Initialize and fit a QRF model for each variable
                for variable in imputed_variables:
                    model = qrf.QRF(seed=self.seed)
                    y = pd.DataFrame(X_train[variable])
                    # Fit the QRF model
                    model.fit(X, y, **qrf_kwargs)

                    self.logger.info(
                        f"QRF model fitted successfully with {len(X)} training samples"
                    )

                    self.models[variable] = model
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

            # For each imputed variable
            for var in imputed_variables:
                # Extract target variable values
                y_test = X_test[var]

                # Create and fit QRF model with trial parameters
                model = qrf.QRF(seed=self.seed)
                model.fit(
                    X_train[predictors], pd.DataFrame(X_train[var]), **params
                )

                # Predict and calculate error
                y_pred = model.predict(X_test[predictors])

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
