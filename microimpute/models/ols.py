"""Ordinary Least Squares regression model for imputation."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydantic import validate_call
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from microimpute.config import VALIDATE_CONFIG
from microimpute.models.imputer import Imputer, ImputerResults


class _LogisticRegressionModel:
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
        **lr_kwargs: Any,
    ) -> None:
        """Fit logistic regression for categorical/boolean target.

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

        # Extract relevant LR parameters from kwargs
        classifier_params = {
            "penalty": lr_kwargs.get("penalty", "l2"),
            "C": lr_kwargs.get("C", 1.0),
            "max_iter": lr_kwargs.get("max_iter", 1000),
            "solver": lr_kwargs.get(
                "solver", "lbfgs" if len(self.categories) <= 2 else "saga"
            ),
            "random_state": self.seed,
        }

        self.classifier = LogisticRegression(**classifier_params)
        self.classifier.fit(X, y_encoded)

    def predict(
        self,
        X: pd.DataFrame,
        return_probs: bool = False,
        quantile: float = 0.5,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Predict classes or probabilities.

        Args:
            X: Input features
            return_probs: If True, return probability distributions
            quantile: For stochastic prediction, can influence decision threshold
        """
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
            # For quantile-based prediction, we could adjust the threshold
            # but for simplicity, using standard prediction
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


class _OLSModel:
    """Internal class to handle OLS regression."""

    def __init__(self, seed: int, logger):
        self.seed = seed
        self.logger = logger
        self.model = None
        self.output_column = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """Fit OLS model."""
        self.output_column = y.name
        X_with_const = sm.add_constant(X)
        self.model = sm.OLS(y, X_with_const).fit()
        self.scale = self.model.scale

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using OLS model."""
        X_with_const = sm.add_constant(X)
        return self.model.predict(X_with_const)


class OLSResults(ImputerResults):
    """
    Fitted OLS instance ready for imputation.
    """

    def _predict_variable(
        self,
        model: Any,
        variable: str,
        X_test: pd.DataFrame,
        quantile: float,
        random_sample: bool,
        return_probs: bool,
        prob_results: Optional[Dict],
    ) -> pd.Series:
        """Predict a single variable using the appropriate model type.

        Args:
            model: The model (_LogisticRegressionModel, _OLSModel, or _ConstantValueModel)
            variable: Name of the variable to predict
            X_test: Test data DataFrame
            quantile: Quantile to predict
            random_sample: Whether to use random sampling
            return_probs: Whether to return probabilities
            prob_results: Dictionary to store probability results

        Returns:
            Series of predicted values
        """
        # Import here to avoid circular import
        from microimpute.models.imputer import _ConstantValueModel

        if isinstance(model, _ConstantValueModel):
            # Constant model - just return the constant value
            return model.predict(X_test)
        elif isinstance(model, _LogisticRegressionModel):
            # Classification for categorical/boolean targets
            if return_probs and prob_results is not None:
                # Get probabilities and classes
                prob_info = model.predict(
                    X_test[self.predictors], return_probs=True
                )
                prob_results[variable] = prob_info

            # Get class predictions
            imputed_values = model.predict(
                X_test[self.predictors], return_probs=False, quantile=quantile
            )
        else:
            # Regression for numeric targets
            X_test_with_const = sm.add_constant(X_test[self.predictors])
            mean_preds = model.predict(X_test_with_const)
            se = np.sqrt(model.scale)
            imputed_values = self._predict_quantile(
                mean_preds=mean_preds,
                se=se,
                mean_quantile=quantile,
                random_sample=random_sample,
            )

        return imputed_values

    def __init__(
        self,
        models: Dict[
            str, Any
        ],  # Can be _OLSModel, _LogisticRegressionModel, or _ConstantValueModel
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
        """Initialize the OLS results.

        Args:
            models: Dictionary of fitted models (OLS or LogisticRegression) for each variable.
            predictors: List of predictor variable names.
            imputed_variables: List of imputed variable names.
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
        random_quantile_sample: Optional[bool] = False,
        return_probs: bool = False,
    ) -> Dict[float, pd.DataFrame]:
        """Predict values at specified quantiles using the OLS model.

        Args:
            X_test: DataFrame containing the test data.
            quantiles: List of quantiles to predict.
            random_quantile_sample: If True, use random quantile sampling for prediction.
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
                        imputed_df[variable] = self._predict_variable(
                            model,
                            variable,
                            X_test,
                            q,
                            random_quantile_sample,
                            return_probs,
                            prob_results,
                        )
                    imputations[q] = pd.DataFrame(imputed_df)

                # Add probabilities to results if requested
                if return_probs and prob_results:
                    imputations["probabilities"] = prob_results

                return imputations
            else:
                q_default = 0.5
                imputed_df = pd.DataFrame()
                for variable in self.imputed_variables:
                    self.logger.info(f"Imputing variable {variable}")
                    model = self.models[variable]
                    imputed_df[variable] = self._predict_variable(
                        model,
                        variable,
                        X_test,
                        q_default,
                        random_quantile_sample,
                        return_probs,
                        prob_results,
                    )
                imputations[q_default] = pd.DataFrame(imputed_df)

                # Add probabilities to results if requested
                if return_probs and prob_results:
                    imputations["probabilities"] = prob_results
                    return imputations
                else:
                    return imputations[q_default]

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
        categorical_targets: Optional[Dict[str, Dict]] = None,
        boolean_targets: Optional[Dict[str, Dict]] = None,
        numeric_targets: Optional[List[str]] = None,
        constant_targets: Optional[Dict[str, Dict]] = None,
        **kwargs: Any,
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

            # Import the constant model from base imputer
            from microimpute.models.imputer import _ConstantValueModel

            for variable in imputed_variables:
                # Handle constant targets separately
                if variable in (constant_targets or {}):
                    constant_val = constant_targets[variable]["value"]
                    model = _ConstantValueModel(constant_val, variable)
                    self.models[variable] = model
                    self.logger.info(
                        f"Using constant value {constant_val} for variable {variable}"
                    )
                    continue

                Y = X_train[variable]

                # Choose appropriate model based on variable type
                if variable in (categorical_targets or {}):
                    # Use logistic regression for categorical targets
                    model = _LogisticRegressionModel(
                        seed=self.seed, logger=self.logger
                    )
                    model.fit(
                        X_train[predictors],
                        Y,
                        var_type=categorical_targets[variable]["type"],
                        categories=categorical_targets[variable].get(
                            "categories"
                        ),
                        **kwargs,
                    )
                    self.logger.info(
                        f"Logistic regression fitted for categorical variable {variable}"
                    )
                elif variable in (boolean_targets or {}):
                    # Use logistic regression for boolean targets
                    model = _LogisticRegressionModel(
                        seed=self.seed, logger=self.logger
                    )
                    model.fit(
                        X_train[predictors], Y, var_type="boolean", **kwargs
                    )
                    self.logger.info(
                        f"Logistic regression fitted for boolean variable {variable}"
                    )
                else:
                    # Use OLS for numeric targets
                    model = _OLSModel(seed=self.seed, logger=self.logger)
                    model.fit(X_train[predictors], Y, **kwargs)
                    self.logger.info(
                        f"OLS regression fitted for numeric variable {variable}"
                    )

                self.models[variable] = model

            return OLSResults(
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
            self.logger.error(f"Error fitting OLS model: {str(e)}")
            raise RuntimeError(f"Failed to fit OLS model: {str(e)}") from e
