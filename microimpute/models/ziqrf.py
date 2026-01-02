"""Zero-Inflated Quantile Regression Forest imputation model.

This model handles zero-inflated continuous variables using a two-stage approach:
1. Stage 1: Classify zero vs non-zero using RandomForestClassifier
2. Stage 2: Predict non-zero values using QRF (only trained on non-zero observations)

For variables that are not zero-inflated (based on threshold), standard QRF is used.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import validate_call
from quantile_forest import RandomForestQuantileRegressor
from sklearn.ensemble import RandomForestClassifier

from microimpute.config import VALIDATE_CONFIG
from microimpute.models.imputer import Imputer, ImputerResults
from microimpute.models.qrf import (
    QRF,
    QRFResults,
    _get_sequential_predictors,
    _QRFModel,
    _RandomForestClassifierModel,
)


class _ZeroInflatedModel:
    """Two-stage model for zero-inflated continuous variables.

    Stage 1: Classify zero vs non-zero
    Stage 2: QRF on non-zero values only
    """

    def __init__(self, seed: int, logger, mean_quantile: float = 0.5):
        self.seed = seed
        self.logger = logger
        self.mean_quantile = mean_quantile
        self.classifier = None
        self.qrf = None
        self.output_column = None
        self.zero_fraction = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **rf_kwargs: Any,
    ) -> None:
        """Fit two-stage model for zero-inflated target.

        Args:
            X: Predictor DataFrame
            y: Target Series (zero-inflated continuous)
            **rf_kwargs: Additional parameters for RandomForest models
        """
        self.output_column = y.name

        # Stage 1: Classify zero vs non-zero
        is_nonzero = (y != 0).astype(int)
        self.zero_fraction = (y == 0).mean()

        self.logger.debug(
            f"  Zero-inflated model: {self.zero_fraction:.1%} zeros in {self.output_column}"
        )

        # Extract classifier params
        classifier_params = {
            "n_estimators": rf_kwargs.get("n_estimators", 100),
            "max_depth": rf_kwargs.get("max_depth", None),
            "min_samples_split": rf_kwargs.get("min_samples_split", 2),
            "min_samples_leaf": rf_kwargs.get("min_samples_leaf", 1),
            "max_features": rf_kwargs.get("max_features", "sqrt"),
            "random_state": self.seed,
            "n_jobs": rf_kwargs.get("n_jobs", -1),
        }

        self.classifier = RandomForestClassifier(**classifier_params)
        self.classifier.fit(X, is_nonzero)

        # Stage 2: Train QRF on non-zero values only
        nonzero_mask = y != 0
        n_nonzero = nonzero_mask.sum()

        if n_nonzero >= 10:  # Need enough non-zero samples
            # Extract QRF params
            qrf_params = {
                "n_estimators": rf_kwargs.get("n_estimators", 100),
                "max_depth": rf_kwargs.get("max_depth", None),
                "min_samples_split": rf_kwargs.get("min_samples_split", 2),
                "min_samples_leaf": rf_kwargs.get("min_samples_leaf", 1),
                "max_features": rf_kwargs.get("max_features", "sqrt"),
                "random_state": self.seed,
                "n_jobs": rf_kwargs.get("n_jobs", -1),
            }

            self.qrf = RandomForestQuantileRegressor(**qrf_params)
            self.qrf.fit(X[nonzero_mask], y[nonzero_mask])
            self.logger.debug(f"  Trained QRF on {n_nonzero} non-zero samples")
        else:
            self.logger.warning(
                f"  Only {n_nonzero} non-zero samples for {self.output_column}, "
                "using mean of non-zeros for predictions"
            )
            self.qrf = None
            self.nonzero_mean = y[nonzero_mask].mean() if n_nonzero > 0 else 0

    def predict(
        self,
        X: pd.DataFrame,
        count_samples: int = 100,
        mean_quantile: float = 0.5,
    ) -> pd.Series:
        """Predict using two-stage model.

        Args:
            X: Predictor DataFrame
            count_samples: Number of quantile samples for QRF
            mean_quantile: Mean of beta distribution for quantile sampling

        Returns:
            Series of predictions (zeros for predicted-zero, QRF values for predicted-nonzero)
        """
        # Stage 1: Classify zero vs non-zero
        pred_nonzero = self.classifier.predict(X)

        # Initialize with zeros
        predictions = np.zeros(len(X))

        # Stage 2: Predict non-zero values using QRF
        nonzero_mask = pred_nonzero == 1
        if nonzero_mask.sum() > 0:
            if self.qrf is not None:
                # Sample from quantile distribution
                eps = 1.0 / (count_samples + 1)
                quantile_grid = np.linspace(eps, 1.0 - eps, count_samples)

                X_nonzero = X[nonzero_mask]
                pred = self.qrf.predict(X_nonzero, quantiles=list(quantile_grid))

                # Sample from beta distribution for quantile selection
                random_generator = np.random.default_rng(self.seed)
                a = mean_quantile / (1 - mean_quantile)
                input_quantiles = (
                    random_generator.beta(a, 1, size=len(X_nonzero)) * count_samples
                )
                input_quantiles = np.clip(
                    input_quantiles.astype(int), 0, count_samples - 1
                )

                # Extract predictions
                if len(pred.shape) == 2:
                    qrf_predictions = pred[np.arange(len(pred)), input_quantiles]
                else:
                    qrf_predictions = pred[
                        np.arange(len(pred)), :, input_quantiles
                    ]

                predictions[nonzero_mask] = qrf_predictions
            else:
                # Fallback to mean
                predictions[nonzero_mask] = self.nonzero_mean

        return pd.Series(predictions, index=X.index, name=self.output_column)


class ZIQRFResults(ImputerResults):
    """Fitted ZI-QRF instance ready for imputation."""

    def __init__(
        self,
        models: Dict[str, Any],
        predictors: List[str],
        imputed_variables: List[str],
        seed: int,
        zero_inflated_vars: List[str],
        imputed_vars_dummy_info: Optional[Dict[str, str]] = None,
        original_predictors: Optional[List[str]] = None,
        categorical_targets: Optional[Dict[str, Dict]] = None,
        boolean_targets: Optional[Dict[str, Dict]] = None,
        constant_targets: Optional[Dict[str, Dict]] = None,
        dummy_processor: Optional[Any] = None,
        log_level: Optional[str] = "WARNING",
    ) -> None:
        """Initialize the ZI-QRF results.

        Args:
            models: Dictionary of fitted models for each variable.
            predictors: List of column names used as predictors.
            imputed_variables: List of column names to be imputed.
            seed: Random seed for reproducibility.
            zero_inflated_vars: List of variables treated as zero-inflated.
            imputed_vars_dummy_info: Optional dictionary with dummy variable info.
            original_predictors: Optional list of original predictor names.
            categorical_targets: Dictionary of categorical target info.
            boolean_targets: Dictionary of boolean target info.
            constant_targets: Dictionary of constant target info.
            dummy_processor: Processor for handling dummy encoding.
            log_level: Logging level.
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
        self.zero_inflated_vars = zero_inflated_vars
        self.categorical_targets = categorical_targets or {}
        self.boolean_targets = boolean_targets or {}
        self.constant_targets = constant_targets or {}
        self.dummy_processor = dummy_processor

    def _get_encoded_predictors(
        self, current_predictors: List[str]
    ) -> List[str]:
        """Get properly encoded predictor columns for sequential imputation."""
        if self.dummy_processor:
            return self.dummy_processor.get_sequential_predictor_columns(
                current_predictors
            )
        else:
            return current_predictors

    def _encode_imputed_variable(
        self, data: pd.DataFrame, variable: str
    ) -> pd.DataFrame:
        """Encode a categorical imputed variable for use as predictor."""
        if (
            self.dummy_processor
            and variable in self.dummy_processor.imputed_var_dummy_mapping
        ):
            data = self.dummy_processor.sequential_imputed_predictor_encoding(
                data, variable
            )
        return data

    @validate_call(config=VALIDATE_CONFIG)
    def predict(
        self,
        X_test: pd.DataFrame,
        quantiles: Optional[List[float]] = None,
        return_probs: bool = False,
        mean_quantile: float = 0.5,
        **kwargs,
    ) -> Dict[float, pd.DataFrame]:
        """Predict imputed values for test data.

        Override base class to skip problematic preprocessing for simple numeric data.

        Args:
            X_test: DataFrame with predictor columns.
            quantiles: Optional list of quantiles to predict.
            return_probs: Whether to return classification probabilities.
            mean_quantile: Mean of beta distribution for quantile sampling.

        Returns:
            Dictionary mapping quantile to DataFrame of predictions.
        """
        # Skip base class preprocessing which can incorrectly treat numeric as categorical
        # Just call _predict directly
        return self._predict(X_test, quantiles, mean_quantile, return_probs)

    @validate_call(config=VALIDATE_CONFIG)
    def _predict(
        self,
        X_test: pd.DataFrame,
        quantiles: Optional[List[float]] = None,
        mean_quantile: Optional[float] = 0.5,
        return_probs: bool = False,
    ) -> Dict[float, pd.DataFrame]:
        """Internal prediction implementation.

        Args:
            X_test: DataFrame with predictor columns.
            quantiles: Optional list of quantiles to predict.
            mean_quantile: Mean of beta distribution for quantile sampling.
            return_probs: Whether to return classification probabilities.

        Returns:
            Dictionary mapping quantile to DataFrame of predictions.
        """
        if quantiles is None:
            quantiles = [mean_quantile]

        imputations = {}

        for q in quantiles:
            imputed_df = X_test.copy()

            for i, variable in enumerate(self.imputed_variables):
                model = self.models.get(variable)
                if model is None:
                    self.logger.warning(f"No model found for {variable}")
                    continue

                # Build predictor set
                current_predictors = _get_sequential_predictors(
                    self.predictors, self.imputed_variables, i
                )
                encoded_predictors = self._get_encoded_predictors(current_predictors)

                # Filter to available columns
                available_cols = [c for c in encoded_predictors if c in imputed_df.columns]

                if isinstance(model, _ZeroInflatedModel):
                    # Use zero-inflated prediction
                    imputed_values = model.predict(
                        imputed_df[available_cols], mean_quantile=q
                    )
                elif isinstance(model, _QRFModel):
                    # Use standard QRF prediction
                    imputed_values = model.predict(
                        imputed_df[available_cols], mean_quantile=q
                    )
                elif isinstance(model, _RandomForestClassifierModel):
                    # Use classifier prediction
                    imputed_values = model.predict(imputed_df[available_cols])
                else:
                    # Fallback: constant model
                    from microimpute.models.imputer import _ConstantValueModel

                    if hasattr(model, "value"):
                        imputed_values = pd.Series(
                            [model.value] * len(imputed_df),
                            index=imputed_df.index,
                            name=variable,
                        )
                    else:
                        imputed_values = pd.Series(
                            [0] * len(imputed_df),
                            index=imputed_df.index,
                            name=variable,
                        )

                imputed_df[variable] = imputed_values

                # Encode for next iteration if needed
                imputed_df = self._encode_imputed_variable(imputed_df, variable)

            imputations[q] = imputed_df[self.imputed_variables]

        return imputations


class ZIQRF(QRF):
    """Zero-Inflated Quantile Regression Forest model for imputation.

    This model extends QRF to handle zero-inflated continuous variables using
    a two-stage approach:
    1. Stage 1: Classify zero vs non-zero using RandomForestClassifier
    2. Stage 2: Predict non-zero values using QRF

    Variables with zero fraction above the threshold are treated as zero-inflated.
    """

    def __init__(
        self,
        zero_inflation_threshold: float = 0.1,
        log_level: Optional[str] = "WARNING",
        memory_efficient: bool = False,
        batch_size: Optional[int] = None,
        cleanup_interval: int = 10,
    ) -> None:
        """Initialize the ZI-QRF model.

        Args:
            zero_inflation_threshold: Fraction of zeros above which a variable
                is treated as zero-inflated. Default 0.1 (10% zeros).
            log_level: Logging level for the imputer.
            memory_efficient: Enable memory optimization features.
            batch_size: Process variables in batches to reduce memory usage.
            cleanup_interval: Frequency of garbage collection.
        """
        super().__init__(
            log_level=log_level,
            memory_efficient=memory_efficient,
            batch_size=batch_size,
            cleanup_interval=cleanup_interval,
        )
        self.zero_inflation_threshold = zero_inflation_threshold
        self.zero_inflated_vars = []
        self.logger.debug(
            f"Initializing ZI-QRF with threshold={zero_inflation_threshold}"
        )

    def _is_zero_inflated(self, y: pd.Series) -> bool:
        """Check if a variable is zero-inflated."""
        zero_fraction = (y == 0).mean()
        return zero_fraction >= self.zero_inflation_threshold

    def _create_model_for_variable(
        self, variable: str, y: Optional[pd.Series] = None, **kwargs
    ) -> Any:
        """Create appropriate model based on variable type and zero-inflation.

        Args:
            variable: Variable name
            y: Target series (used to check zero-inflation)
            **kwargs: Additional arguments

        Returns:
            Model instance (_ZeroInflatedModel, _QRFModel, or _RandomForestClassifierModel)
        """
        # Check if boolean or categorical first
        if variable in self.boolean_targets:
            return _RandomForestClassifierModel(self.seed, self.logger)
        elif variable in self.categorical_targets:
            return _RandomForestClassifierModel(self.seed, self.logger)
        elif y is not None and self._is_zero_inflated(y):
            # Use zero-inflated model for numeric variables with many zeros
            self.zero_inflated_vars.append(variable)
            self.logger.info(f"  Using zero-inflated model for {variable}")
            return _ZeroInflatedModel(self.seed, self.logger)
        else:
            # Standard QRF
            return _QRFModel(self.seed, self.logger)

    @validate_call(config=VALIDATE_CONFIG)
    def fit(
        self,
        X_train: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        weight_col: Optional[str] = None,
        skip_missing: bool = False,
        not_numeric_categorical: Optional[List[str]] = None,
        original_predictors: Optional[List[str]] = None,
        categorical_targets: Optional[Dict[str, Dict]] = None,
        boolean_targets: Optional[Dict[str, Dict]] = None,
        numeric_targets: Optional[List[str]] = None,
        constant_targets: Optional[Dict[str, Dict]] = None,
        tune_hyperparameters: bool = False,
        **qrf_kwargs: Any,
    ) -> ZIQRFResults:
        """Fit the ZI-QRF model to the training data.

        Args:
            X_train: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.
            weight_col: Optional weight column.
            skip_missing: Skip missing variables with warning.
            not_numeric_categorical: Variables to treat as numeric despite appearance.
            original_predictors: Original predictor names before encoding.
            categorical_targets: Categorical target info.
            boolean_targets: Boolean target info.
            numeric_targets: Numeric target list.
            constant_targets: Constant target info.
            tune_hyperparameters: Whether to tune hyperparameters.
            **qrf_kwargs: Additional QRF parameters.

        Returns:
            ZIQRFResults instance ready for prediction.
        """
        # Store target type information
        self.categorical_targets = categorical_targets or {}
        self.boolean_targets = boolean_targets or {}
        self.numeric_targets = numeric_targets or []
        self.constant_targets = constant_targets or {}
        self.imputed_variables = imputed_variables
        self.zero_inflated_vars = []

        self.logger.info(f"Fitting ZI-QRF on {len(imputed_variables)} variables")

        # Import constant model
        from microimpute.models.imputer import _ConstantValueModel

        for i, variable in enumerate(imputed_variables):
            var_start_time = time.time()

            # Handle constant targets
            if variable in (constant_targets or {}):
                constant_val = constant_targets[variable]["value"]
                self.models[variable] = _ConstantValueModel(constant_val, variable)
                self.logger.info(f"Using constant value {constant_val} for {variable}")
                continue

            # Build predictor set
            current_predictors = _get_sequential_predictors(
                predictors, imputed_variables, i
            )

            # Get encoded predictors
            dummy_processor = getattr(self, "dummy_processor", None)
            encoded_predictors = self._get_encoded_predictors(
                current_predictors, dummy_processor
            )

            self.logger.info(
                f"[{i+1}/{len(imputed_variables)}] Fitting model for '{variable}'"
            )

            # Get target data
            y = X_train[variable]

            # Create appropriate model (checks zero-inflation)
            model = self._create_model_for_variable(variable, y=y)

            # Fit the model
            if isinstance(model, _ZeroInflatedModel):
                model.fit(
                    X_train[encoded_predictors],
                    y,
                    **qrf_kwargs,
                )
            elif isinstance(model, _RandomForestClassifierModel):
                var_type = (
                    "boolean"
                    if variable in self.boolean_targets
                    else "categorical"
                )
                categories = (
                    self.categorical_targets.get(variable, {}).get("categories")
                    if variable in self.categorical_targets
                    else None
                )
                model.fit(
                    X_train[encoded_predictors],
                    y,
                    var_type=var_type,
                    categories=categories,
                    **qrf_kwargs,
                )
            else:
                # Standard QRF
                model.fit(
                    X_train[encoded_predictors],
                    y,
                    **qrf_kwargs,
                )

            self.models[variable] = model

            var_time = time.time() - var_start_time
            self.logger.info(f"  Fitted in {var_time:.2f}s")

            # Encode for next iteration
            X_train = self._encode_imputed_variable(X_train, variable, dummy_processor)

        return ZIQRFResults(
            models=self.models,
            predictors=predictors,
            imputed_variables=imputed_variables,
            seed=self.seed,
            zero_inflated_vars=self.zero_inflated_vars,
            imputed_vars_dummy_info=self.imputed_vars_dummy_info,
            original_predictors=original_predictors or predictors,
            categorical_targets=categorical_targets,
            boolean_targets=boolean_targets,
            constant_targets=constant_targets,
            dummy_processor=getattr(self, "dummy_processor", None),
            log_level=self.log_level,
        )
