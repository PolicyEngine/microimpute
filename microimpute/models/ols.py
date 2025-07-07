"""Ordinary Least Squares regression model for imputation."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydantic import validate_call
from scipy.stats import norm

from microimpute.config import VALIDATE_CONFIG
from microimpute.models.imputer import Imputer, ImputerResults


class OLSResults(ImputerResults):
    """
    Fitted OLS instance ready for imputation.
    """

    def __init__(
        self,
        models: Dict[str, "OLS"],
        predictors: List[str],
        imputed_variables: List[str],
        seed: int,
        imputed_vars_dummy_info: Optional[Dict[str, str]] = None,
        original_predictors: Optional[List[str]] = None,
        log_level: Optional[str] = "WARNING",
    ) -> None:
        """Initialize the OLS results.

        Args:
            model: Fitted OLS model.
            predictors: List of predictor variable names.
            imputed_variables: List of imputed variable names.
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
        random_quantile_sample: Optional[bool] = False,
    ) -> Dict[float, pd.DataFrame]:
        """Predict values at specified quantiles using the OLS model.

        Args:
            X_test: DataFrame containing the test data.
            quantiles: List of quantiles to predict.
            random_quantile_sample: If True, use random quantile sampling for prediction.

        Returns:
            Dictionary mapping quantiles to predicted values.

        Raises:
            RuntimeError: If prediction fails.
        """
        try:
            # Create output dictionary with results
            imputations: Dict[float, pd.DataFrame] = {}

            X_test_with_const = sm.add_constant(X_test[self.predictors])

            if quantiles:
                if random_quantile_sample:
                    self.logger.warning(
                        f"Predicting at random quantiles sampled from a beta distribution is not possible when specified quantiles are provided."
                    )
                self.logger.info(
                    f"Predicting at {len(quantiles)} quantiles: {quantiles}"
                )
                for q in quantiles:
                    imputed_df = pd.DataFrame()
                    for variable in self.imputed_variables:
                        model = self.models[variable]
                        mean_preds = model.predict(X_test_with_const)
                        se = np.sqrt(model.scale)
                        imputed_df[variable] = self._predict_quantile(
                            mean_preds=mean_preds,
                            se=se,
                            mean_quantile=q,
                            random_sample=random_quantile_sample,
                        )
                    imputations[q] = pd.DataFrame(imputed_df)
            else:
                q = 0.5
                imputed_df = pd.DataFrame()
                for variable in self.imputed_variables:
                    self.logger.info(f"Imputing variable {variable}")
                    model = self.models[variable]
                    mean_preds = model.predict(X_test_with_const)
                    se = np.sqrt(model.scale)
                    imputed_df[variable] = self._predict_quantile(
                        mean_preds=mean_preds,
                        se=se,
                        mean_quantile=q,
                        random_sample=random_quantile_sample,
                    )
                imputations[q] = pd.DataFrame(imputed_df)
            return imputations

        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise RuntimeError(
                f"Failed to predict with OLS model: {str(e)}"
            ) from e

    @validate_call(config=VALIDATE_CONFIG)
    def _predict_quantile(
        self,
        mean_preds: pd.Series,
        se: float,
        mean_quantile: float,
        random_sample: bool,
        count_samples: int = 10,
    ) -> np.ndarray:
        """Predict values at a specified quantile.

        Args:
            mean_preds: Mean predictions from the model.
            se: Standard error of the predictions.
            mean_quantile: Quantile to predict (the quantile affects the center
                of the beta distribution from which to sample when imputing each data point).
            random_sample: If True, use random quantile sampling for prediction.
            count_samples: Number of quantile samples to generate when
                random_sample is True.

        Returns:
            Array of predicted values at the specified quantile.

        Raises:
            RuntimeError: If prediction fails.
        """
        try:
            if random_sample == True:
                self.logger.info(
                    f"Predicting at random quantiles sampled from a beta distribution with mean quantile {mean_quantile}"
                )
                random_generator = np.random.default_rng(self.seed)

                # Calculate alpha parameter for beta distribution
                a = mean_quantile / (1 - mean_quantile)

                # Generate count_samples beta distributed values with parameter a
                beta_samples = random_generator.beta(a, 1, size=count_samples)

                # Convert to normal quantiles using norm.ppf
                normal_quantiles = norm.ppf(beta_samples)

                # For each mean prediction, randomly select one of the quantiles
                sampled_indices = random_generator.integers(
                    0, count_samples, size=len(mean_preds)
                )
                selected_quantiles = normal_quantiles[sampled_indices]

                # Adjust each mean prediction by corresponding sampled quantile times standard error
                return mean_preds + selected_quantiles * se
            else:
                self.logger.info(
                    f"Predicting at specified quantile {mean_quantile}"
                )
                specified_quantile = norm.ppf(mean_quantile)
                return mean_preds + specified_quantile * se

        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            self.logger.error(
                f"Error predicting at random quantiles with mean quantile {mean_quantile}: {str(e)}"
            )
            raise RuntimeError(
                f"Failed to predict at random quantiles with mean quantile {mean_quantile}: {str(e)}"
            ) from e


class OLS(Imputer):
    """
    Ordinary Least Squares regression model for imputation.

    This model predicts different quantiles by assuming normally
    distributed residuals.
    """

    def __init__(self, log_level: Optional[str] = "WARNING") -> None:
        """Initialize the OLS model."""
        super().__init__(log_level=log_level)
        self.model = None
        self.log_level = log_level
        self.logger.debug("Initializing OLS imputer")

    @validate_call(config=VALIDATE_CONFIG)
    def _fit(
        self,
        X_train: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        original_predictors: Optional[List[str]] = None,
    ) -> OLSResults:
        """Fit the OLS model to the training data.

        Args:
            X_train: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.

        Returns:
            The fitted model instance.

        Raises:
            RuntimeError: If model fitting fails.
        """
        try:
            self.logger.info(
                f"Fitting OLS model with {len(predictors)} predictors"
            )

            self.models = {}
            X_with_const = sm.add_constant(X_train[predictors])
            for variable in imputed_variables:
                Y = X_train[variable]
                model = sm.OLS(Y, X_with_const).fit()
                self.logger.info(
                    f"OLS model fitted successfully for the imputed variable {variable}, R-squared: {model.rsquared:.4f}"
                )
                self.models[variable] = model
            return OLSResults(
                models=self.models,
                predictors=predictors,
                imputed_variables=imputed_variables,
                imputed_vars_dummy_info=self.imputed_vars_dummy_info,
                original_predictors=self.original_predictors,
                seed=self.seed,
                log_level=self.log_level,
            )
        except Exception as e:
            self.logger.error(f"Error fitting OLS model: {str(e)}")
            raise RuntimeError(f"Failed to fit OLS model: {str(e)}") from e
