"""Quantile Regression imputation model."""

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydantic import validate_call
from statsmodels.tools.sm_exceptions import IterationLimitWarning

from microimpute.config import VALIDATE_CONFIG
from microimpute.models.imputer import Imputer, ImputerResults

warnings.filterwarnings("ignore", category=IterationLimitWarning)


class QuantRegResults(ImputerResults):
    """
    Fitted QuantReg instance ready for imputation.
    """

    def __init__(
        self,
        models: Dict[float, "QuantReg"],
        predictors: List[str],
        imputed_variables: List[str],
        seed: int,
        imputed_vars_dummy_info: Optional[Dict[str, str]] = None,
        original_predictors: Optional[List[str]] = None,
        log_level: Optional[str] = "WARNING",
    ) -> None:
        """Initialize the QRF results.

        Args:
            models: Dict of quantiles and fitted QuantReg models.
            predictors: List of column names used as predictors.
            imputed_variables: List of column names to be imputed.
            seed: Random seed for reproducibility.
            imptuted_vars_dummy_info: Optional dictionary containing information
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
        """Predict values at specified quantiles using the Quantile Regression model.

        Args:
            X_test: DataFrame containing the test data.
            quantiles: List of quantiles to predict. If None, uses the
                quantiles from training.
            random_quantile_sample: If True, use random quantile sampling for prediction.

        Returns:
            Dictionary mapping quantiles to predicted values.

        Raises:
            ValueError: If a requested quantile was not fitted during training.
            RuntimeError: If prediction fails.
        """
        try:
            # Create output dictionary with results
            imputations: Dict[float, pd.DataFrame] = {}

            X_test_with_const = sm.add_constant(X_test[self.predictors])
            self.logger.info(f"Prepared test data with {len(X_test)} samples")

            if quantiles is not None:
                # Predict for each requested quantile
                for q in quantiles:
                    imputed_df = pd.DataFrame()
                    self.logger.info(f"Predicting with model for q={q}")
                    for variable in self.imputed_variables:
                        try:
                            if q not in self.models[variable]:
                                error_msg = f"Model for quantile {q} not fitted. Available quantiles: {list(self.models.keys())}"
                                self.logger.error(error_msg)
                                raise ValueError(error_msg)
                        except Exception as quantile_error:
                            self.logger.error(
                                f"Error accessing quantiles: {str(quantile_error)}"
                            )
                            raise RuntimeError(
                                f"Failed to access {q} quantile for prediction"
                            ) from quantile_error

                        model = self.models[variable][q]
                        imputed_df[variable] = model.predict(X_test_with_const)
                    imputations[q] = imputed_df
            else:
                quantiles = list(self.models[self.imputed_variables[0]].keys())
                if random_quantile_sample:
                    self.logger.info(
                        "Sampling random quantiles for each prediction"
                    )
                    mean_quantile = np.mean(quantiles)

                    # Get predictions for all quantiles first
                    random_q_imputations = {}
                    for q in quantiles:
                        imputed_df = pd.DataFrame()
                        for variable in self.imputed_variables:
                            model = self.models[variable][q]
                            imputed_df[variable] = model.predict(
                                X_test_with_const
                            )
                        random_q_imputations[q] = imputed_df

                    # Create a final dataframe to hold the random quantile imputed values
                    result_df = pd.DataFrame(
                        index=random_q_imputations[quantiles[0]].index,
                        columns=self.imputed_variables,
                    )

                    # Sample one quantile per row
                    rng = np.random.default_rng(self.seed)
                    for idx in result_df.index:
                        sampled_q = rng.choice(quantiles)

                        # For all variables, use the sampled quantile for this row
                        for variable in self.imputed_variables:
                            result_df.loc[idx, variable] = (
                                random_q_imputations[sampled_q].loc[
                                    idx, variable
                                ]
                            )

                    # Add to imputations dictionary using the mean quantile as key
                    imputations[mean_quantile] = result_df
                else:
                    # Predict for all quantiles that were already fitted
                    self.logger.info(
                        f"Predicting on already fitted {quantiles} quantiles"
                    )
                    for q in quantiles:
                        self.logger.info(f"Predicting with model for q={q}")
                        imputed_df = pd.DataFrame()
                        for variable in self.imputed_variables:
                            model = self.models[variable][q]
                            imputed_df[variable] = model.predict(
                                X_test_with_const
                            )
                        imputations[q] = imputed_df

            self.logger.info(
                f"Completed predictions for {len(quantiles)} quantiles"
            )
            return imputations

        except ValueError as e:
            # Re-raise value errors directly
            raise e
        except Exception as e:
            self.logger.error(f"Error in QuantReg prediction: {str(e)}")
            raise RuntimeError(
                f"Failed to predict with QuantReg model: {str(e)}"
            ) from e


class QuantReg(Imputer):
    """
    Quantile Regression model for imputation.

    This model uses statsmodels' QuantReg implementation to
    directly predict specific quantiles.
    """

    def __init__(self, log_level: Optional[str] = "WARNING") -> None:
        """Initialize the Quantile Regression model."""
        super().__init__(log_level=log_level)
        self.models: Dict[str, Any] = {}
        self.log_level = log_level
        self.logger.debug("Initializing QuantReg imputer")

    @validate_call(config=VALIDATE_CONFIG)
    def _fit(
        self,
        X_train: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        original_predictors: Optional[List[str]] = None,
        quantiles: Optional[List[float]] = None,
    ) -> QuantRegResults:
        """Fit the Quantile Regression model to the training data.

        Args:
            X_train: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.
            quantiles: List of quantiles to fit models for.

        Returns:
            The fitted model instance.

        Raises:
            ValueError: If any quantile is outside the [0, 1] range.
            RuntimeError: If model fitting fails.
        """
        try:
            for variable in imputed_variables:
                self.models[variable] = {}

            # Validate quantiles if provided
            if quantiles:
                invalid_quantiles = [q for q in quantiles if not 0 <= q <= 1]
                if invalid_quantiles:
                    error_msg = f"Quantiles must be between 0 and 1, got: {invalid_quantiles}"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
                self.logger.info(
                    f"Fitting QuantReg models for {len(quantiles)} quantiles: {quantiles}"
                )

            X_with_const = sm.add_constant(X_train[predictors])
            self.logger.info(
                f"Prepared training data with {len(X_train)} samples, {len(predictors)} predictors"
            )

            if quantiles:
                for q in quantiles:
                    self.logger.info(f"Fitting quantile regression for q={q}")
                    for variable in imputed_variables:
                        Y = X_train[variable]
                        self.models[variable][q] = sm.QuantReg(
                            Y, X_with_const
                        ).fit(q=q)
                    self.logger.info(f"Model for q={q} fitted successfully")
            else:
                random_generator = np.random.default_rng(self.seed)
                q = 0.5
                self.logger.info(
                    f"Fitting quantile regression for random quantile {q:.4f}"
                )
                for variable in imputed_variables:
                    self.logger.info(f"Imputing variable {variable}")
                    Y = X_train[variable]
                    self.models[variable][q] = sm.QuantReg(
                        Y, X_with_const
                    ).fit(q=q)
                self.logger.info(f"Model for q={q:.4f} fitted successfully")

            self.logger.info(f"QuantReg has {len(self.models)} fitted models")
            return QuantRegResults(
                models=self.models,
                predictors=predictors,
                imputed_variables=imputed_variables,
                imputed_vars_dummy_info=self.imputed_vars_dummy_info,
                original_predictors=self.original_predictors,
                seed=self.seed,
                log_level=self.log_level,
            )
        except Exception as e:
            self.logger.error(f"Error fitting QuantReg model: {str(e)}")
            raise RuntimeError(
                f"Failed to fit QuantReg model: {str(e)}"
            ) from e
