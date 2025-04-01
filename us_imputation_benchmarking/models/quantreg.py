from typing import Any, Collection, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydantic import validate_call
from us_imputation_benchmarking.config import validate_config

from us_imputation_benchmarking.models.imputer import Imputer, ImputerResults


class QuantReg(Imputer):
    """
    Quantile Regression model for imputation.

    This model uses statsmodels' QuantReg implementation to
    directly predict specific quantiles.
    """

    def __init__(self) -> None:
        """Initialize the Quantile Regression model."""
        super().__init__()
        self.models: Dict[float, Any] = {}
        self.logger.debug("Initializing QuantReg imputer")

    @validate_call(config=validate_config, validate_return=False)
    def _fit(
        self,
        X_train: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        quantiles: Optional[List[float]] = None,
    ) -> Any:  # Will return QuantRegResults
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
            # Validate quantiles if provided
            if quantiles:
                invalid_quantiles = [q for q in quantiles if not 0 <= q <= 1]
                if invalid_quantiles:
                    error_msg = (
                        f"Quantiles must be between 0 and 1, got: {invalid_quantiles}"
                    )
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
                self.logger.info(
                    f"Fitting QuantReg models for {len(quantiles)} quantiles: {quantiles}"
                )
            else:
                self.logger.info(
                    "No quantiles provided, will fit a single random quantile"
                )

            Y = X_train[imputed_variables]
            X_with_const = sm.add_constant(X_train[predictors])
            self.logger.info(
                f"Prepared training data with {len(X_train)} samples, {len(predictors)} predictors"
            )

            if quantiles:
                for q in quantiles:
                    self.logger.info(f"Fitting quantile regression for q={q}")
                    self.models[q] = sm.QuantReg(Y, X_with_const).fit(q=q)
                    self.logger.info(f"Model for q={q} fitted successfully")
            else:
                q = np.random.uniform(0, 1)
                self.logger.info(f"Fitting quantile regression for random q={q:.4f}")
                self.models[q] = sm.QuantReg(Y, X_with_const).fit(q=q)
                self.logger.info(f"Model for q={q:.4f} fitted successfully")

            self.logger.info(f"QuantReg has {len(self.models)} fitted models")
            return _QuantRegResults(
                models=self.models,
                predictors=predictors,
                imputed_variables=imputed_variables,
            )
        except Exception as e:
            self.logger.error(f"Error fitting QuantReg model: {str(e)}")
            raise RuntimeError(f"Failed to fit QuantReg model: {str(e)}") from e


class _QuantRegResults(ImputerResults):
    """
    Fitted QuantReg instance ready for imputation.
    """
    def __init__(
        self,
        models: Dict[float, Any],
        predictors: List[str],
        imputed_variables: List[str],
    ) -> None:
        """Initialize the QRF results.

        Args:
            models: Dict of quantiles and fitted QuantReg models.
            predictors: List of column names used as predictors.
            imputed_variables: List of column names to be imputed.
        """
        super().__init__(predictors, imputed_variables)
        self.models = models

    @validate_call(config=validate_config)
    def _predict(
        self, X_test: pd.DataFrame, 
        quantiles: Optional[List[float]] = None
    ) -> Dict[float, pd.DataFrame]:
        """Predict values at specified quantiles using the Quantile Regression model.

        Args:
            X_test: DataFrame containing the test data.
            quantiles: List of quantiles to predict. If None, uses the quantiles
                from training.

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
                    try:
                        if q not in self.models:
                            error_msg = f"Model for quantile {q} not fitted. Available quantiles: {list(self.models.keys())}"
                            self.logger.error(error_msg)
                            raise ValueError(error_msg)
                    except Exception as quantile_error:
                        self.logger.error(f"Error accessing quantiles: {str(quantile_error)}")
                        raise RuntimeError(
                            f"Failed to access {q} quantile for prediction"
                        ) from quantile_error

                    self.logger.info(f"Predicting with model for q={q}")
                    imputation = self.models[q].predict(X_test_with_const)
                    imputations[q] = pd.DataFrame(imputation)
            else:
                # Predict for all quantiles that were already fitted
                quantiles = list(self.models.keys())
                self.logger.info(f"Predicting on already fitted {quantiles} quantiles")
                for q in quantiles:
                    self.logger.info(f"Predicting with model for q={q}")
                    imputation = self.models[q].predict(X_test_with_const)
                    imputations[q] = pd.DataFrame(imputation)

            self.logger.info(f"Completed predictions for {len(quantiles)} quantiles")
            return imputations

        except ValueError as e:
            # Re-raise value errors directly
            raise e
        except Exception as e:
            self.logger.error(f"Error in QuantReg prediction: {str(e)}")
            raise RuntimeError(
                f"Failed to predict with QuantReg model: {str(e)}"
            ) from e
