"""Abstract base classes for imputation models.

This module defines the core architecture for imputation models in MicroImpute.
It provides two abstract base classes:
1. Imputer - For model initialization and fitting
2. ImputerResults - For storing fitted models and making predictions

All model implementations should extend these classes to ensure a consistent interface.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import SkipValidation, validate_call

from microimpute.config import RANDOM_STATE, VALIDATE_CONFIG
from microimpute.utils.type_handling import (
    DummyVariableProcessor,
    VariableTypeDetector,
)


class _ConstantValueModel:
    """Simple model that always returns a constant value."""

    def __init__(self, constant_value, variable_name: str):
        self.constant_value = constant_value
        self.variable_name = variable_name

    def predict(self, X: pd.DataFrame, **kwargs) -> pd.Series:
        """Return the constant value for all rows."""
        return pd.Series(
            self.constant_value, index=X.index, name=self.variable_name
        )


class Imputer(ABC):
    """
    Abstract base class for fitting imputation models.

    All imputation models should inherit from this class and implement
    the required methods.
    """

    def __init__(
        self,
        seed: Optional[int] = RANDOM_STATE,
        log_level: Optional[str] = "WARNING",
    ) -> None:
        """Initialize the imputer model."""
        self.predictors: Optional[List[str]] = None
        self.imputed_variables: Optional[List[str]] = None
        self.imputed_vars_dummy_info: Optional[Dict[str, Any]] = None
        self.original_predictors: Optional[List[str]] = None
        self.categorical_targets: Dict[str, Dict] = (
            {}
        )  # {var_name: {"type": "categorical", "categories": [...]}}
        self.boolean_targets: Dict[str, Dict] = (
            {}
        )  # {var_name: {"type": "boolean", "dtype": ...}}
        self.numeric_targets: List[str] = []  # [var_name, ...]
        self.constant_targets: Dict[str, Dict] = (
            {}
        )  # {var_name: {"value": constant, "dtype": ...}}
        self.seed = seed
        self.logger = logging.getLogger(__name__)

        log_level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        self.logger.setLevel(log_level_map.get(log_level, logging.WARNING))

    @validate_call(config=VALIDATE_CONFIG)
    def _validate_data(self, data: pd.DataFrame, columns: List[str]) -> None:
        """Validate that all required columns are in the data.

        Args:
            data: DataFrame to validate
            columns: Column names that should be present

        Raises:
            ValueError: If any columns are missing from the data or if data is empty
        """
        if data is None or data.empty:
            raise ValueError("Data must not be None or empty")

        missing_columns: Set[str] = set(columns) - set(data.columns)
        if missing_columns:
            error_msg = f"Missing columns in data: {missing_columns}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        missing_count = data.isna().sum().sum()
        if missing_count > 0:
            self.logger.warning(
                f"Data contains {missing_count} missing values"
            )

    def identify_target_types(
        self,
        data: pd.DataFrame,
        imputed_variables: List[str],
        not_numeric_categorical: Optional[List[str]] = None,
    ) -> None:
        """Identify and track variable types for imputation targets.

        Args:
            data: DataFrame containing the data.
            imputed_variables: List of variables to be imputed.
            not_numeric_categorical: Optional list of variable names that should
                be treated as numeric even if they would normally be detected as
                numeric_categorical.
        """
        detector = VariableTypeDetector()
        not_numeric_categorical = not_numeric_categorical or []

        for var in imputed_variables:
            if var not in data.columns:
                continue

            # First check if the variable has a constant value
            unique_values = data[var].dropna().unique()
            if len(unique_values) == 1:
                constant_val = unique_values[0]
                self.constant_targets[var] = {
                    "value": constant_val,
                    "dtype": data[var].dtype,
                }
                self.logger.warning(
                    f"Target variable '{var}' has constant value {constant_val}. "
                    f"All imputations will use this constant value."
                )
                continue

            var_type, categories = detector.categorize_variable(
                data[var],
                var,
                self.logger,
                force_numeric=(var in not_numeric_categorical),
            )

            if var_type == "bool":
                self.boolean_targets[var] = {
                    "type": "boolean",
                    "dtype": data[var].dtype,
                }
                self.logger.info(f"Identified boolean target: {var}")

            elif var_type in ["categorical", "numeric_categorical"]:
                self.categorical_targets[var] = {
                    "type": var_type,
                    "categories": categories,
                    "dtype": data[var].dtype,
                }
                self.logger.info(
                    f"Identified categorical target: {var} with {len(categories) if categories else 0} categories"
                )

            else:
                self.numeric_targets.append(var)
                self.logger.debug(f"Identified numeric target: {var}")

    @validate_call(config=VALIDATE_CONFIG)
    def preprocess_data_types(
        self,
        data: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        not_numeric_categorical: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, List[str], List[str], Dict[str, Any]]:
        """Preprocess predictors only - convert categorical predictors to dummies.
        Imputation targets remain in original form for classification.

        Args:
            data: DataFrame containing the data.
            predictors: List of predictor column names.
            imputed_variables: List of variables to impute (kept in original form).
            not_numeric_categorical: Optional list of variable names that should
                be treated as numeric even if they would normally be detected as
                numeric_categorical.

        Returns:
            Tuple of (processed_data, updated_predictors, imputed_variables, empty_dict)

        Raises:
            ValueError: If any column cannot be processed.
        """
        try:
            processor = DummyVariableProcessor(self.logger)
            processed_data, updated_predictors = (
                processor.preprocess_predictors(
                    data,
                    predictors,
                    imputed_variables,
                    not_numeric_categorical,
                )
            )

            # Store the processor for later use in test data
            self.dummy_processor = processor

            # Return empty dict as we no longer need dummy info for targets
            return processed_data, updated_predictors, imputed_variables, {}

        except Exception as e:
            self.logger.error(f"Error during data preprocessing: {str(e)}")
            raise RuntimeError("Failed to preprocess data types") from e

    @validate_call(config=VALIDATE_CONFIG)
    def fit(
        self,
        X_train: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        weight_col: Optional[Union[str, np.ndarray, pd.Series]] = None,
        skip_missing: bool = False,
        not_numeric_categorical: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:  # Returns ImputerResults
        """Fit the model to the training data.

        Args:
            X_train: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.
            weight_col: Optional name of the column or column array/series containing sampling weights. When provided, `X_train` will be sampled with replacement using this column as selection probabilities before fitting the model.
            skip_missing: If True, skip variables missing from training data with warning. If False, raise error for missing variables.
            not_numeric_categorical: Optional list of variable names that should
                be treated as numeric even if they would normally be detected as
                numeric_categorical.
            **kwargs: Additional model-specific parameters.

        Returns:
            The fitted model instance.

        Raises:
            ValueError: If input data is invalid or missing required columns.
            RuntimeError: If model fitting fails.
            NotImplementedError: If method is not implemented by subclass.
        """
        original_predictors = predictors.copy()

        try:
            # Handle missing variables if skip_missing is enabled
            if skip_missing:
                imputed_variables = self._handle_missing_variables(
                    X_train, imputed_variables
                )

            # Validate data
            self._validate_data(X_train, predictors + imputed_variables)

            for variable in imputed_variables:
                if variable in predictors:
                    error_msg = (
                        f"Variable '{variable}' is both in the predictors and imputed "
                        "variables list. Please ensure they are distinct."
                    )
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
        except Exception as e:
            raise ValueError(f"Invalid input data for model: {str(e)}") from e

        weights = None
        if weight_col is not None and isinstance(weight_col, str):
            if weight_col not in X_train.columns:
                raise ValueError(
                    f"Weight column '{weight_col}' not found in training data"
                )
            weights = X_train[weight_col]
        elif weight_col is not None and isinstance(weight_col, np.ndarray):
            weights = pd.Series(weight_col, index=X_train.index)

        if weights is not None and (weights <= 0).any():
            raise ValueError("Weights must be positive")

        # Identify target types BEFORE preprocessing
        self.identify_target_types(
            X_train, imputed_variables, not_numeric_categorical
        )

        X_train, predictors, imputed_variables, imputed_vars_dummy_info = (
            self.preprocess_data_types(
                X_train, predictors, imputed_variables, not_numeric_categorical
            )
        )

        if weights is not None:
            weights_normalized = weights / weights.sum()
            X_train = X_train.sample(
                n=len(X_train),
                replace=True,
                weights=weights_normalized,
                random_state=self.seed,
            ).reset_index(drop=True)

        # Save predictors and imputed variables
        self.predictors = predictors
        self.imputed_variables = imputed_variables
        self.imputed_vars_dummy_info = imputed_vars_dummy_info
        self.original_predictors = original_predictors

        # Defer actual training to subclass with all parameters
        fitted_model = self._fit(
            X_train,
            self.predictors,
            self.imputed_variables,
            self.original_predictors,
            categorical_targets=self.categorical_targets,
            boolean_targets=self.boolean_targets,
            numeric_targets=self.numeric_targets,
            constant_targets=self.constant_targets,
            **kwargs,
        )
        return fitted_model

    @abstractmethod
    @validate_call(config=VALIDATE_CONFIG)
    def _fit(
        self,
        X_train: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        original_predictors: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Actual model-fitting logic (overridden in method subclass).

        Args:
            X_train: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.
            original_predictors: Optional list of original predictor names
                before dummy encoding.
            **kwargs: Additional model-specific parameters.

        Raises:
            ValueError: If specific model parameters are invalid.
            RuntimeError: If model fitting fails.
        """
        raise NotImplementedError("Subclasses must implement `_fit`")

    def _handle_missing_variables(
        self, X_train: pd.DataFrame, imputed_variables: List[str]
    ) -> List[str]:
        """Handle missing variables in the training data.

        Args:
            X_train: Training data DataFrame
            imputed_variables: List of variables to impute

        Returns:
            List of available variables to impute
        """
        # Identify available and missing variables
        available_vars = [v for v in imputed_variables if v in X_train.columns]
        missing_vars = [
            v for v in imputed_variables if v not in X_train.columns
        ]

        if missing_vars:
            self.logger.warning(
                f"Variables not found in X_train: {missing_vars}. "
                f"Available variables: {available_vars}"
            )

            self.logger.warning(
                f"Skipping missing variables and proceeding with {len(available_vars)} available variables"
            )

        return available_vars


class ImputerResults(ABC):
    """
    Abstract base class representing a fitted model for imputation.

    All imputation models should inherit from this class and implement
    the required methods.

    predict() can only be called once the model is fitted in an
    ImputerResults instance.
    """

    def __init__(
        self,
        predictors: List[str],
        imputed_variables: List[str],
        seed: int,
        imputed_vars_dummy_info: Optional[Dict[str, Any]] = None,
        original_predictors: Optional[List[str]] = None,
        log_level: Optional[str] = "WARNING",
    ):
        self.predictors = predictors
        self.imputed_variables = imputed_variables
        self.imputed_vars_dummy_info = imputed_vars_dummy_info
        self.original_predictors = original_predictors
        self.seed = seed
        self.logger = logging.getLogger(__name__)

        log_level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        self.logger.setLevel(log_level_map.get(log_level, logging.WARNING))

    @validate_call(config=VALIDATE_CONFIG)
    def _validate_quantiles(
        self,
        quantiles: Optional[List[float]],
    ) -> None:
        """Validate that all provided quantiles are valid.

        Args:
            quantiles: List of quantiles to validate

        Raises:
            ValueError: If passed quantiles are not in the correct format
        """
        if quantiles is not None:
            if not isinstance(quantiles, list):
                self.logger.error(
                    f"quantiles must be a list, got {type(quantiles)}"
                )
                raise ValueError(
                    f"quantiles must be a list, got {type(quantiles)}"
                )

            invalid_quantiles = [q for q in quantiles if not 0 <= q <= 1]
            if invalid_quantiles:
                self.logger.error(
                    f"Invalid quantiles (must be between 0 and 1): {invalid_quantiles}"
                )
                raise ValueError(
                    f"All quantiles must be between 0 and 1, got {invalid_quantiles}"
                )

    @validate_call(config=VALIDATE_CONFIG)
    def preprocess_data_types(
        self,
        data: pd.DataFrame,
        predictors: List[str],
        dummy_processor: Optional[DummyVariableProcessor] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Apply dummy encoding to test data predictors based on training mapping.

        Args:
            data: DataFrame containing the test data.
            predictors: List of original predictor column names.
            dummy_processor: Processor with training mappings (if available).

        Returns:
            Tuple of (processed_data, updated_predictors)

        Raises:
            ValueError: If any column cannot be converted to numeric.
        """
        try:
            if dummy_processor and hasattr(dummy_processor, "dummy_mapping"):
                # Use existing processor with training mappings
                return dummy_processor.apply_dummy_encoding_to_test(
                    data, predictors
                )
            else:
                # Fallback: create new processor (shouldn't happen normally)
                processor = DummyVariableProcessor(self.logger)
                # This will only encode predictors in test data
                return processor.preprocess_predictors(data, predictors, [])
        except Exception as e:
            self.logger.error(
                f"Error during test data preprocessing: {str(e)}"
            )
            raise RuntimeError("Failed to preprocess data types") from e

    # Note: postprocess_imputations removed - categorical targets now handled directly by classification

    @validate_call(config=VALIDATE_CONFIG)
    def predict(
        self,
        X_test: pd.DataFrame,
        quantiles: Optional[List[float]] = None,
        return_probs: bool = False,
        **kwargs: Any,
    ) -> Dict[float, pd.DataFrame]:
        """Predict imputed values at specified quantiles.

        Will validate that quantiles passed are in the correct format.

        Args:
            X_test: DataFrame containing the test data.
            quantiles: List of quantiles to predict. If None, uses random quantile.
            return_probs: If True, also return probability distributions for categorical/boolean variables.
            **kwargs: Additional model-specific parameters.

        Returns:
            Dictionary mapping quantiles to imputed values.
            If return_probs=True, includes 'probabilities' key with probability distributions.

        Raises:
            ValueError: If input data is invalid.
            RuntimeError: If imputation fails.
        """
        try:
            self._validate_quantiles(quantiles)
        except Exception as quantile_error:
            raise ValueError(
                f"Invalid quantiles: {str(quantile_error)}"
            ) from quantile_error

        # Get dummy processor from parent imputer if available
        dummy_processor = getattr(self, "dummy_processor", None)
        X_test, updated_predictors = self.preprocess_data_types(
            X_test, self.original_predictors, dummy_processor
        )

        # Note: Missing dummy categories are already handled in apply_dummy_encoding_to_test
        # Missing actual predictors will raise an error during preprocessing

        # Defer actual imputations to subclass with all parameters
        imputations = self._predict(
            X_test, quantiles, return_probs=return_probs, **kwargs
        )
        # No more postprocessing - categorical targets handled directly
        return imputations

    @abstractmethod
    @validate_call(config=VALIDATE_CONFIG)
    def _predict(
        self,
        X_test: pd.DataFrame,
        quantiles: Optional[List[float]] = None,
        return_probs: bool = False,
    ) -> Dict[float, pd.DataFrame]:
        """Predict imputed values at specified quantiles.

        Args:
            X_test: DataFrame containing the test data.
            quantiles: List of quantiles to predict. If None, uses random quantile.

        Returns:
            Dictionary mapping quantiles to imputed values.

        Raises:
            RuntimeError: If imputation fails.
            NotImplementedError: If method is not implemented by subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement the predict method"
        )
