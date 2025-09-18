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
        quantiles_specified: bool = False,
        boolean_targets: Optional[Dict[str, Dict]] = None,
        constant_targets: Optional[Dict[str, Dict]] = None,
    ) -> None:
        """Initialize the QuantReg results.

        Args:
            models: Dict of quantiles and fitted QuantReg models.
            predictors: List of column names used as predictors.
            imputed_variables: List of column names to be imputed.
            seed: Random seed for reproducibility.
            imptuted_vars_dummy_info: Optional dictionary containing information
                about dummy variables for imputed variables.
            original_predictors: Optional list of original predictor variable
                names before dummy encoding.
            quantiles_specified: Whether quantiles were explicitly specified during fit.
            boolean_targets: Dictionary of boolean target info for conversion back to bool.
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
        self.quantiles_specified = quantiles_specified
        self.boolean_targets = boolean_targets or {}
        self.constant_targets = constant_targets or {}

    @validate_call(config=VALIDATE_CONFIG)
    def _predict(
        self,
        X_test: pd.DataFrame,
        quantiles: Optional[List[float]] = None,
        random_quantile_sample: Optional[bool] = False,
        return_probs: bool = False,
    ) -> Dict[float, pd.DataFrame]:
        """Predict values at specified quantiles using the Quantile Regression model.

        Args:
            X_test: DataFrame containing the test data.
            quantiles: List of quantiles to predict. If None, uses the
                quantiles from training.
            random_quantile_sample: If True, use random quantile sampling for prediction.
            return_probs: Ignored for QuantReg (included for API consistency).

        Returns:
            Dictionary mapping quantiles to predicted values.

        Raises:
            ValueError: If a requested quantile was not fitted during training.
            RuntimeError: If prediction fails.
        """
        # Log warning if return_probs is used with QuantReg
        if return_probs:
            self.logger.warning(
                "return_probs parameter will be ignored by QuantReg, as QuantReg only supports numeric targets."
            )
        try:
            # Create output dictionary with results
            imputations: Dict[float, pd.DataFrame] = {}

            # Store original quantiles parameter to determine return type
            quantiles_param = quantiles

            X_test_with_const = sm.add_constant(X_test[self.predictors])
            self.logger.info(f"Prepared test data with {len(X_test)} samples")

            if quantiles is not None:
                # Predict for each requested quantile
                for q in quantiles:
                    imputed_df = pd.DataFrame()
                    self.logger.info(f"Predicting with model for q={q}")
                    for variable in self.imputed_variables:
                        # Import constant model
                        from microimpute.models.imputer import (
                            _ConstantValueModel,
                        )

                        # Check if this is a constant target
                        # For constant targets, use any available quantile since value is the same
                        if variable in self.constant_targets:
                            # Get the constant model from any quantile (they're all the same)
                            available_q = list(self.models[variable].keys())[0]
                            model = self.models[variable][available_q]
                            predictions = model.predict(X_test)
                        else:
                            # Regular variable - check quantile exists
                            try:
                                if q not in self.models[variable]:
                                    error_msg = f"Model for quantile {q} not fitted. Available quantiles: {list(self.models[variable].keys())}"
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
                            if isinstance(model, _ConstantValueModel):
                                # This shouldn't happen as we handle constant targets above
                                predictions = model.predict(X_test)
                            else:
                                predictions = model.predict(X_test_with_const)
                                # Convert to boolean if this was a boolean target
                                if variable in self.boolean_targets:
                                    predictions = predictions > 0.5
                        imputed_df[variable] = predictions
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
                            # Import constant model
                            from microimpute.models.imputer import (
                                _ConstantValueModel,
                            )

                            # Check if this is a constant target
                            if variable in self.constant_targets:
                                # Get the constant model from any quantile
                                available_q = list(
                                    self.models[variable].keys()
                                )[0]
                                model = self.models[variable][available_q]
                                predictions = model.predict(X_test)
                            else:
                                model = self.models[variable][q]
                                if isinstance(model, _ConstantValueModel):
                                    # Constant model - just return the constant value
                                    predictions = model.predict(X_test)
                                else:
                                    predictions = model.predict(
                                        X_test_with_const
                                    )
                                    # Convert to boolean if this was a boolean target
                                    if variable in self.boolean_targets:
                                        predictions = predictions > 0.5
                            imputed_df[variable] = predictions
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
                            # Import constant model
                            from microimpute.models.imputer import (
                                _ConstantValueModel,
                            )

                            # Check if this is a constant target
                            if variable in self.constant_targets:
                                # Get the constant model from any quantile
                                available_q = list(
                                    self.models[variable].keys()
                                )[0]
                                model = self.models[variable][available_q]
                                predictions = model.predict(X_test)
                            else:
                                model = self.models[variable][q]
                                if isinstance(model, _ConstantValueModel):
                                    # Constant model - just return the constant value
                                    predictions = model.predict(X_test)
                                else:
                                    predictions = model.predict(
                                        X_test_with_const
                                    )
                                    # Convert to boolean if this was a boolean target
                                    if variable in self.boolean_targets:
                                        predictions = predictions > 0.5
                            imputed_df[variable] = predictions
                        imputations[q] = imputed_df

            self.logger.info(
                f"Completed predictions for {len(imputations)} quantiles"
            )

            # Return behavior based on how the model was fitted:
            # - If quantiles were explicitly specified during fit OR predict, return dict
            # - Otherwise, return DataFrame directly for single quantile
            if quantiles_param is not None or self.quantiles_specified:
                return imputations
            else:
                # Default behavior: return DataFrame directly
                q = list(imputations.keys())[0]
                return imputations[q]

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
        categorical_targets: Optional[Dict[str, Dict]] = None,
        boolean_targets: Optional[Dict[str, Dict]] = None,
        numeric_targets: Optional[List[str]] = None,
        constant_targets: Optional[Dict[str, Dict]] = None,
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
        # Check for unsupported categorical targets
        if categorical_targets:
            unsupported = list(categorical_targets.keys())
            error_msg = (
                f"QuantReg does not support categorical imputation targets: {unsupported}. "
                f"Use QRF, OLS, or Matching models instead for categorical variables. "
                f"QuantReg can only handle numeric and boolean targets."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Warn about boolean targets being treated as numeric
        if boolean_targets:
            boolean_vars = list(boolean_targets.keys())
            self.logger.warning(
                f"Boolean targets will be treated as numeric [0,1]: {boolean_vars}. "
                f"Values will be thresholded at 0.5 during prediction."
            )

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

            # Import constant model
            from microimpute.models.imputer import _ConstantValueModel

            if quantiles:
                for q in quantiles:
                    self.logger.info(f"Fitting quantile regression for q={q}")
                    for variable in imputed_variables:
                        # Handle constant targets
                        if variable in (constant_targets or {}):
                            constant_val = constant_targets[variable]["value"]
                            self.models[variable][q] = _ConstantValueModel(
                                constant_val, variable
                            )
                            self.logger.info(
                                f"Using constant value {constant_val} for variable {variable}"
                            )
                            continue

                        Y = X_train[variable]
                        # Convert boolean to numeric for regression
                        if variable in (boolean_targets or {}):
                            Y = Y.astype(float)
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
                    # Handle constant targets
                    if variable in (constant_targets or {}):
                        constant_val = constant_targets[variable]["value"]
                        self.models[variable][q] = _ConstantValueModel(
                            constant_val, variable
                        )
                        self.logger.info(
                            f"Using constant value {constant_val} for variable {variable}"
                        )
                        continue

                    Y = X_train[variable]
                    # Convert boolean to numeric for regression
                    if variable in (boolean_targets or {}):
                        Y = Y.astype(float)
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
                quantiles_specified=(quantiles is not None),
                boolean_targets=boolean_targets,
                constant_targets=constant_targets,
            )
        except Exception as e:
            self.logger.error(f"Error fitting QuantReg model: {str(e)}")
            raise RuntimeError(
                f"Failed to fit QuantReg model: {str(e)}"
            ) from e
