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


class VariableTypeDetector:
    """Utility class for detecting and categorizing variable types."""

    @staticmethod
    def is_boolean_variable(series: pd.Series) -> bool:
        """Check if a series represents boolean data."""
        if pd.api.types.is_bool_dtype(series):
            return True

        unique_vals = set(series.dropna().unique())
        if pd.api.types.is_integer_dtype(series) and unique_vals <= {0, 1}:
            return True
        if pd.api.types.is_float_dtype(series) and unique_vals <= {0.0, 1.0}:
            return True

        return False

    @staticmethod
    def is_categorical_variable(series: pd.Series) -> bool:
        """Check if a series represents categorical string/object data."""
        return pd.api.types.is_string_dtype(
            series
        ) or pd.api.types.is_object_dtype(series)

    @staticmethod
    def is_numeric_categorical_variable(
        series: pd.Series, max_unique: int = 10
    ) -> bool:
        """Check if a numeric series should be treated as categorical."""
        if not pd.api.types.is_numeric_dtype(series):
            return False

        if series.nunique() >= max_unique:
            return False

        # Check for equal spacing between values
        unique_values = np.sort(series.dropna().unique())
        if len(unique_values) < 2:
            return True

        differences = np.diff(unique_values)
        return np.allclose(differences, differences[0], rtol=1e-9)

    @staticmethod
    def categorize_variable(
        series: pd.Series, col_name: str, logger: logging.Logger
    ) -> Tuple[str, Optional[List]]:
        """
        Categorize a variable and return its type and categories if applicable.

        Returns:
            Tuple of (variable_type, categories)
            variable_type: 'bool', 'categorical', 'numeric_categorical', or 'numeric'
            categories: List of unique values for categorical types, None for numeric
        """
        if VariableTypeDetector.is_boolean_variable(series):
            return "bool", None

        if VariableTypeDetector.is_categorical_variable(series):
            return "categorical", series.unique().tolist()

        if VariableTypeDetector.is_numeric_categorical_variable(series):
            categories = [float(i) for i in series.unique().tolist()]
            logger.info(
                f"Treating numeric variable '{col_name}' as categorical due to low unique count and equal spacing"
            )
            return "numeric_categorical", categories

        return "numeric", None


class DummyVariableProcessor:
    """Handles conversion between original variables and dummy variables."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.dummy_info = {
            "original_dtypes": {},
            "column_mapping": {},
            "original_categories": {},
        }

    def preprocess_variables(
        self,
        data: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
    ) -> Tuple[pd.DataFrame, List[str], List[str], Dict]:
        """
        Process all variables, converting categoricals to dummies as needed.

        Returns:
            Tuple of (processed_data, updated_predictors, updated_imputed_variables, imputed_vars_dummy_info)
        """
        data = data[predictors + imputed_variables].copy()
        detector = VariableTypeDetector()

        # Categorize all columns
        column_categories = {}
        for col in data.columns:
            var_type, categories = detector.categorize_variable(
                data[col], col, self.logger
            )
            column_categories[col] = (var_type, categories, data[col].dtype)

        # Process variables according to their types
        bool_columns = [
            col
            for col, (vtype, _, _) in column_categories.items()
            if vtype == "bool"
        ]
        if bool_columns:
            self._process_boolean_columns(
                data, bool_columns, column_categories
            )

        categorical_columns = [
            col
            for col, (vtype, _, _) in column_categories.items()
            if vtype in ["categorical", "numeric_categorical"]
        ]

        if categorical_columns:
            data, predictors, imputed_variables = (
                self._process_categorical_columns(
                    data,
                    categorical_columns,
                    column_categories,
                    predictors,
                    imputed_variables,
                )
            )

        imputed_vars_dummy_info = self._filter_imputed_vars_info(
            imputed_variables
        )

        return data, predictors, imputed_variables, imputed_vars_dummy_info

    def _process_boolean_columns(
        self,
        data: pd.DataFrame,
        bool_columns: List[str],
        column_categories: Dict,
    ) -> None:
        """Process boolean columns by converting to float."""
        self.logger.info(
            f"Converting {len(bool_columns)} boolean columns: {bool_columns}"
        )

        for col in bool_columns:
            _, _, original_dtype = column_categories[col]
            self.dummy_info["original_dtypes"][col] = ("bool", original_dtype)
            self.dummy_info["column_mapping"][col] = [col]
            data[col] = data[col].astype("float64")

    def _process_categorical_columns(
        self,
        data: pd.DataFrame,
        categorical_columns: List[str],
        column_categories: Dict,
        predictors: List[str],
        imputed_variables: List[str],
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """Process categorical columns by creating dummy variables."""
        for col in categorical_columns:
            var_type, categories, original_dtype = column_categories[col]
            self.dummy_info["original_dtypes"][col] = (
                (
                    "numeric categorical"
                    if var_type == "numeric_categorical"
                    else "categorical"
                ),
                original_dtype,
            )
            if categories:
                self.dummy_info["original_categories"][col] = categories

            if var_type == "numeric_categorical":
                data[col] = data[col].astype("float64").astype("category")

        # Create dummy variables
        dummy_data = pd.get_dummies(
            data[categorical_columns],
            columns=categorical_columns,
            dtype="float64",
            drop_first=True,
        )

        self.logger.debug(
            f"Created {dummy_data.shape[1]} dummy variables from {len(categorical_columns)} categorical columns"
        )

        # Create column mappings
        for orig_col in categorical_columns:
            related_dummies = [
                col
                for col in dummy_data.columns
                if col.startswith(f"{orig_col}_")
            ]

            if not related_dummies:
                self._handle_single_category_variable(
                    data, orig_col, column_categories[orig_col]
                )
                self.dummy_info["column_mapping"][orig_col] = [orig_col]
            else:
                self.dummy_info["column_mapping"][orig_col] = related_dummies

        # Combine data
        numeric_data = data.drop(
            columns=[col for col in categorical_columns if col in data.columns]
        )
        data = pd.concat([numeric_data, dummy_data], axis=1)

        # Update predictor and imputed_variables lists with dummy columns' names
        predictors, imputed_variables = self._update_variable_lists(
            predictors, imputed_variables, data.columns
        )

        for col in data.columns:
            data[col] = data[col].astype("float64")

        return data, predictors, imputed_variables

    def _handle_single_category_variable(
        self, data: pd.DataFrame, col: str, col_info: Tuple[str, List, Any]
    ) -> None:
        """Handle variables with only a single category."""
        var_type, categories, _ = col_info

        if var_type == "numeric_categorical":
            self.logger.info(
                f"Keeping numeric categorical '{col}' as numeric column"
            )
            if categories:
                data[col] = categories[0]
        else:
            self.logger.info(
                f"Converting single-value categorical '{col}' to numeric encoding (1.0)"
            )
            data[col] = 1.0

    def _update_variable_lists(
        self,
        predictors: List[str],
        imputed_variables: List[str],
        data_columns: pd.Index,
    ) -> Tuple[List[str], List[str]]:
        """Update predictor and imputed variable lists with dummy columns."""
        new_predictors = predictors.copy()
        new_imputed_variables = imputed_variables.copy()

        for col, dummy_cols in self.dummy_info["column_mapping"].items():
            if len(dummy_cols) > 0 and all(
                dc in data_columns for dc in dummy_cols
            ):
                if col in new_predictors:
                    new_predictors.remove(col)
                    new_predictors.extend(dummy_cols)
                elif col in new_imputed_variables:
                    new_imputed_variables.remove(col)
                    new_imputed_variables.extend(dummy_cols)

        return new_predictors, new_imputed_variables

    def _filter_imputed_vars_info(self, imputed_variables: List[str]) -> Dict:
        """Create dummy info specific to imputed variables."""
        imputed_vars_dummy_info = {
            "original_dtypes": {},
            "column_mapping": {},
            "original_categories": {},
        }

        for col in self.dummy_info["column_mapping"]:
            dummy_cols = self.dummy_info["column_mapping"][col]
            if any(dc in imputed_variables for dc in dummy_cols):
                imputed_vars_dummy_info["column_mapping"][col] = dummy_cols
                imputed_vars_dummy_info["original_dtypes"][col] = (
                    self.dummy_info["original_dtypes"][col]
                )
                if col in self.dummy_info["original_categories"]:
                    imputed_vars_dummy_info["original_categories"][col] = (
                        self.dummy_info["original_categories"][col]
                    )

        return imputed_vars_dummy_info

    def reverse_dummy_encoding(
        self,
        imputations: Union[Dict[float, pd.DataFrame], pd.DataFrame],
        dummy_info: Dict[str, Any],
    ) -> Union[Dict[float, pd.DataFrame], pd.DataFrame]:
        """Convert dummy variables back to original categorical format."""
        if isinstance(imputations, dict):
            processed_imputations = {}
            for quantile, df in imputations.items():
                processed_imputations[quantile] = (
                    self._process_single_dataframe(df.copy(), dummy_info)
                )
        else:
            processed_imputations = self._process_single_dataframe(
                imputations.copy(), dummy_info
            )

        return processed_imputations

    def _process_single_dataframe(
        self, df: pd.DataFrame, dummy_info: Dict[str, Any]
    ) -> pd.DataFrame:
        """Process a single quantile DataFrame."""
        for orig_col, dummy_cols in dummy_info.get(
            "column_mapping", {}
        ).items():
            if orig_col not in dummy_info.get("original_dtypes", {}):
                continue

            dtype_info = dummy_info["original_dtypes"][orig_col]
            if not isinstance(dtype_info, tuple) or len(dtype_info) != 2:
                self.logger.warning(
                    f"Unexpected dtype format for {orig_col}: {dtype_info}"
                )
                continue

            dtype_category, original_pandas_dtype = dtype_info

            if dtype_category == "bool" and orig_col in df.columns:
                df[orig_col] = self._reverse_boolean(
                    df[orig_col], original_pandas_dtype
                )
            elif dtype_category in ["categorical", "numeric_categorical"]:
                df = self._reverse_categorical(
                    df,
                    orig_col,
                    dummy_cols,
                    dummy_info,
                    dtype_category,
                    original_pandas_dtype,
                )

        return df

    def _reverse_boolean(
        self, series: pd.Series, original_dtype: Any
    ) -> pd.Series:
        """Convert float back to boolean."""
        threshold = 0.5
        bool_series = series > threshold
        return bool_series.astype(original_dtype)

    def _reverse_categorical(
        self,
        df: pd.DataFrame,
        orig_col: str,
        dummy_cols: List[str],
        dummy_info: Dict,
        dtype_category: str,
        original_dtype: Any,
    ) -> pd.DataFrame:
        """Convert dummy variables back to categorical."""
        available_dummies = [col for col in dummy_cols if col in df.columns]

        if not available_dummies:
            return self._handle_single_category_reverse(
                df, orig_col, dummy_cols, dummy_info, original_dtype
            )

        categories = dummy_info["original_categories"][orig_col]
        reference_category = self._find_reference_category(
            orig_col, available_dummies, categories
        )

        # Convert dummies back to categorical
        df[orig_col] = self._dummies_to_categorical(
            df[available_dummies], orig_col, categories, reference_category
        )

        # Convert to original dtype if needed
        if original_dtype != "object":
            try:
                df[orig_col] = df[orig_col].astype(original_dtype)
            except (ValueError, TypeError) as e:
                self.logger.warning(
                    f"Could not convert {orig_col} to {original_dtype}: {e}"
                )

        # Drop dummy columns
        df = df.drop(columns=available_dummies)

        return df

    def _handle_single_category_reverse(
        self,
        df: pd.DataFrame,
        orig_col: str,
        dummy_cols: List[str],
        dummy_info: Dict,
        original_dtype: Any,
    ) -> pd.DataFrame:
        """Handle reversal for single-category variables."""
        if (
            orig_col in df.columns
            and len(dummy_cols) == 1
            and dummy_cols[0] == orig_col
        ):
            categories = dummy_info["original_categories"][orig_col]
            df[orig_col] = categories[0]

            if original_dtype != "object":
                try:
                    df[orig_col] = df[orig_col].astype(original_dtype)
                except (ValueError, TypeError) as e:
                    self.logger.warning(
                        f"Could not convert {orig_col} to original dtype: {e}"
                    )

        return df

    def _find_reference_category(
        self,
        orig_col: str,
        available_dummies: List[str],
        original_categories: List,
    ) -> Any:
        """Find the reference category that was dropped during dummy encoding."""
        dummy_categories = []
        for dummy_col in available_dummies:
            category_part = dummy_col.replace(f"{orig_col}_", "", 1)
            try:
                if category_part.replace(".", "").replace("-", "").isdigit():
                    dummy_categories.append(float(category_part))
                else:
                    dummy_categories.append(category_part)
            except:
                dummy_categories.append(category_part)

        for cat in original_categories:
            if cat not in dummy_categories:
                return cat

        return original_categories[0] if original_categories else None

    def _dummies_to_categorical(
        self,
        dummy_df: pd.DataFrame,
        orig_col: str,
        categories: List,
        reference_category: Any,
    ) -> pd.Series:
        """Convert dummy columns to categorical values."""
        category_mapping = {
            f"{orig_col}_{cat}": cat
            for cat in categories
            if f"{orig_col}_{cat}" in dummy_df.columns
        }

        # Find max dummy value per row
        max_idx = dummy_df.idxmax(axis=1)
        max_values = dummy_df.max(axis=1)

        # Initialize with reference category
        result = pd.Series(reference_category, index=dummy_df.index)

        # Assign to dummy categories where confidence > threshold
        threshold = 0.5
        high_confidence_mask = max_values >= threshold
        if high_confidence_mask.any():
            result.loc[high_confidence_mask] = max_idx[
                high_confidence_mask
            ].map(category_mapping)

        nan_mask = result.isna()
        if nan_mask.any():
            result.loc[nan_mask] = reference_category
            self.logger.warning(
                f"Some values could not be mapped for {orig_col}, using reference category"
            )

        self.logger.info(
            f"Assigned {high_confidence_mask.sum()} observations to dummy categories, "
            f"{(~high_confidence_mask).sum()} to reference category '{reference_category}'"
        )

        return result


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

    @validate_call(config=VALIDATE_CONFIG)
    def preprocess_data_types(
        self,
        data: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
    ) -> Tuple[pd.DataFrame, List[str], List[str], Dict[str, Any]]:
        """Ensure all predictor columns are numeric. Transform boolean and categorical variables if necessary.

        Args:
            data: DataFrame containing the data.
            predictors: List of column names to ensure are numeric.
            imputed_variables: List of column names to ensure are numeric.

        Returns:
            Tuple of (data, predictors, imputed_variables, dummy_info)

        Raises:
            ValueError: If any column cannot be converted to numeric.
        """
        try:
            processor = DummyVariableProcessor(self.logger)
            return processor.preprocess_variables(
                data, predictors, imputed_variables
            )

        except Exception as e:
            self.logger.error(
                f"Error during donor data preprocessing: {str(e)}"
            )
            raise RuntimeError("Failed to preprocess data types") from e

    @validate_call(config=VALIDATE_CONFIG)
    def fit(
        self,
        X_train: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        weight_col: Optional[Union[str, np.ndarray, pd.Series]] = None,
        skip_missing: bool = False,
        **kwargs: Any,
    ) -> Any:  # Returns ImputerResults
        """Fit the model to the training data.

        Args:
            X_train: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.
            weight_col: Optional name of the column or column array/series containing sampling weights. When provided, `X_train` will be sampled with replacement using this column as selection probabilities before fitting the model.
            skip_missing: If True, skip variables missing from training data with warning. If False, raise error for missing variables.
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

        X_train, predictors, imputed_variables, imputed_vars_dummy_info = (
            self.preprocess_data_types(X_train, predictors, imputed_variables)
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
    ) -> pd.DataFrame:
        """Ensure all predictor columns are numeric. Transform booleand and categorical variables if necessary.

        Args:
            data: DataFrame containing the data.
            predictors: List of column names to ensure are numeric.

        Returns:
            data: DataFrame with specified variables converted to numeric types.

        Raises:
            ValueError: If any column cannot be converted to numeric.
        """
        try:
            processor = DummyVariableProcessor(self.logger)
            processed_data, _, _, _ = processor.preprocess_variables(
                data, predictors, []
            )
            return processed_data
        except Exception as e:
            self.logger.error(
                f"Error during receiver data preprocessing: {str(e)}"
            )
            raise RuntimeError("Failed to preprocess data types") from e

    @validate_call(config=VALIDATE_CONFIG)
    def postprocess_imputations(
        self,
        imputations: Union[Dict[float, pd.DataFrame], pd.DataFrame],
        dummy_info: Dict[str, Any],
    ) -> Union[Dict[float, pd.DataFrame], pd.DataFrame]:
        """Convert imputed bool and categorical dummy variables back to original data types.

        This function reverses the encoding applied by preprocess_data,
        converting dummy variables back to their original boolean or categorical forms.
        For numeric categorical variables, values are rounded to the nearest valid category.

        Args:
            imputations: Dictionary mapping quantiles to DataFrames of imputed values
            dummy_info: Dictionary containing information about dummy variable mappings
                and original data types

        Returns:
            Dictionary mapping quantiles to DataFrames with original data types restored or a single DataFrame if only one quantile is provided.

        Raises:
            RuntimeError: If conversion back to original types fails
        """
        try:
            processor = DummyVariableProcessor(self.logger)
            return processor.reverse_dummy_encoding(imputations, dummy_info)
        except Exception as e:
            self.logger.error(
                f"Error when postprocessing imputations: {str(e)}"
            )
            raise RuntimeError(
                f"Failed to post-process imputations: {str(e)}"
            ) from e

    @validate_call(config=VALIDATE_CONFIG)
    def predict(
        self,
        X_test: pd.DataFrame,
        quantiles: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> Dict[float, pd.DataFrame]:
        """Predict imputed values at specified quantiles.

        Will validate that quantiles passed are in the correct format.

        Args:
            X_test: DataFrame containing the test data.
            quantiles: List of quantiles to predict. If None, uses random quantile.
            **kwargs: Additional model-specific parameters.

        Returns:
            Dictionary mapping quantiles to imputed values.

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

        X_test = self.preprocess_data_types(X_test, self.original_predictors)

        for col in self.predictors:
            if col not in X_test.columns:
                self.logger.info(
                    f"Predictor '{col}' not found in test data columns. \n"
                    "Will create a dummy variable with 0.0 values for this column."
                )
                X_test[col] = np.zeros(len(X_test), dtype="float64")

        # Defer actual imputations to subclass with all parameters
        imputations = self._predict(X_test, quantiles, **kwargs)
        if self.imputed_vars_dummy_info is not None:
            imputations = self.postprocess_imputations(
                imputations, self.imputed_vars_dummy_info
            )
        return imputations

    @abstractmethod
    @validate_call(config=VALIDATE_CONFIG)
    def _predict(
        self, X_test: pd.DataFrame, quantiles: Optional[List[float]] = None
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
