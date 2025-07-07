"""Abstract base classes for imputation models.

This module defines the core architecture for imputation models in MicroImpute.
It provides two abstract base classes:
1. Imputer - For model initialization and fitting
2. ImputerResults - For storing fitted models and making predictions

All model implementations should extend these classes to ensure a consistent interface.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
from pydantic import SkipValidation, validate_call

from microimpute.config import RANDOM_STATE, VALIDATE_CONFIG


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
        if log_level == "DEBUG":
            log_level = logging.DEBUG
        elif log_level == "INFO":
            log_level = logging.INFO
        elif log_level == "WARNING":
            log_level = logging.WARNING
        elif log_level == "ERROR":
            log_level = logging.ERROR
        elif log_level == "CRITICAL":
            log_level = logging.CRITICAL
        self.logger.setLevel(log_level)

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
    def _preprocess_data_types(
        self,
        data: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
    ) -> pd.DataFrame:
        """Ensure all predictor columns are numeric. Transform booleand and categorical variables if necessary.

        Args:
            data: DataFrame containing the data.
            predictors: List of column names to ensure are numeric.
            imputed_variables: List of column names to ensure are numeric.

        Returns:
            data: DataFrame with specified variables converted to numeric types.
            predictors: List of predictor column names after conversion.
            imputed_variables: List of imputed variable column names after conversion.
            dummy_info: Dictionary containing information about dummy variables created for post-processing of imputed variables.

        Raises:
            ValueError: If any column cannot be converted to numeric.
        """
        # Initialize dummy information dictionary
        dummy_info = {
            "original_dtypes": {},
            "column_mapping": {},
            "original_categories": {},
        }

        data = data[predictors + imputed_variables].copy()

        try:
            self.logger.debug(
                "Converting boolean and categorical columns to numerical format"
            )
            # Identify boolean columns and convert them to strings
            bool_columns = [
                col
                for col in data.columns
                if (
                    pd.api.types.is_bool_dtype(data[col])
                    or (
                        pd.api.types.is_integer_dtype(data[col])
                        and set(data[col].unique()) == {0, 1}
                    )
                    or (
                        pd.api.types.is_float_dtype(data[col])
                        and set(data[col].unique()) == {0.0, 1.0}
                    )
                )
            ]

            if bool_columns:
                self.logger.info(
                    f"Found {len(bool_columns)} boolean columns to convert: {bool_columns}"
                )
                for col in bool_columns:
                    dummy_info["original_dtypes"][col] = (
                        "bool",
                        data[col].dtype,
                    )
                    # For boolean columns, map the column to itself since we don't create dummies
                    dummy_info["column_mapping"][col] = [col]
                    data[col] = data[col].astype("float64")

            # Identify string and object columns (excluding already processed booleans)
            string_columns = [
                col
                for col in data.columns
                if (
                    pd.api.types.is_string_dtype(data[col])
                    or pd.api.types.is_object_dtype(data[col])
                )
                and col not in bool_columns
            ]

            # Identify numeric columns that represent categorical data
            numeric_categorical_columns = [
                col
                for col in data.columns
                if pd.api.types.is_numeric_dtype(data[col])
                and data[col].nunique()
                < 10  # Parse as category if unique count < 10
                and col
                not in bool_columns  # Exclude already processed boolean columns
            ]

            if numeric_categorical_columns:
                self.logger.info(
                    f"Found {len(numeric_categorical_columns)} numeric columns with unique values < 10, treating as categorical: {numeric_categorical_columns}. Converting to dummy variables."
                )
                for col in numeric_categorical_columns:
                    dummy_info["original_categories"][col] = [
                        float(i) for i in data[col].unique().tolist()
                    ]
                    dummy_info["original_dtypes"][col] = (
                        "numeric categorical",
                        data[col].dtype,
                    )
                    data[col] = data[col].astype("float64")
                    data[col] = data[col].astype("category")

            if string_columns:
                self.logger.info(
                    f"Found {len(string_columns)} categorical columns to convert: {string_columns}"
                )

                # Store original categories and dtypes for categorical columns
                for col in string_columns:
                    dummy_info["original_dtypes"][col] = (
                        "categorical",
                        data[col].dtype,
                    )
                    dummy_info["original_categories"][col] = (
                        data[col].unique().tolist()
                    )

            if string_columns or numeric_categorical_columns:
                # Use pandas get_dummies to create one-hot encoded features
                categorical_columns = (
                    string_columns + numeric_categorical_columns
                )
                dummy_data = pd.get_dummies(
                    data[categorical_columns],
                    columns=categorical_columns,
                    dtype="float64",
                    drop_first=True,
                )
                for col in dummy_data.columns:
                    dummy_data[col] = dummy_data[col].astype("float64")
                self.logger.debug(
                    f"Created {dummy_data.shape[1]} dummy variables from {len(categorical_columns)} categorical columns"
                )

                # Create mapping from original columns to their resulting dummy columns
                for orig_col in categorical_columns:
                    # Find all dummy columns that came from this original column
                    related_dummies = [
                        col
                        for col in dummy_data.columns
                        if col.startswith(f"{orig_col}_")
                    ]
                    dummy_info["column_mapping"][orig_col] = (
                        related_dummies
                        if len(related_dummies) > 0
                        else [orig_col]
                    )

                # Drop original string and numeric categorical columns and join the dummy variables
                numeric_data = data.drop(columns=categorical_columns)
                self.logger.debug(
                    f"Removed original string and numeric categorical columns, data shape: {numeric_data.shape}"
                )

                # Combine numeric columns with dummy variables
                data = pd.concat([numeric_data, dummy_data], axis=1)
                for col in data.columns:
                    data[col] = data[col].astype("float64")
                self.logger.info(
                    f"Data shape after dummy variable conversion: {data.shape}"
                )

            imputed_vars_dummy_info = {
                "original_dtypes": {},
                "column_mapping": {},
                "original_categories": {},
            }
            for col, dummy_cols in dummy_info["column_mapping"].items():
                if col in predictors:
                    predictors.remove(col)
                    predictors.extend(dummy_cols)
                elif col in imputed_variables:
                    imputed_variables.remove(col)
                    imputed_variables.extend(dummy_cols)
                    imputed_vars_dummy_info["column_mapping"][col] = dummy_cols
                    imputed_vars_dummy_info["original_dtypes"][col] = (
                        dummy_info["original_dtypes"][col][0],
                        dummy_info["original_dtypes"][col][1],
                    )
                    if col in dummy_info["original_categories"]:
                        imputed_vars_dummy_info["original_categories"][col] = (
                            dummy_info["original_categories"][col]
                        )

            return data, predictors, imputed_variables, imputed_vars_dummy_info

        except Exception as e:
            self.logger.error(
                f"Error during string column conversion: {str(e)}"
            )
            raise RuntimeError(
                "Failed to convert string columns to dummy variables"
            ) from e

    @validate_call(config=VALIDATE_CONFIG)
    def fit(
        self,
        X_train: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        weight_col: Optional[Union[str, np.ndarray, pd.Series]] = None,
        **kwargs: Any,
    ) -> Any:  # Returns ImputerResults
        """Fit the model to the training data.

        Args:
            X_train: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.
            weight_col: Optional name of the column or column array/series containing sampling weights. When provided, `X_train` will be sampled with replacement using this column as selection probabilities before fitting the model.
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
            self._preprocess_data_types(X_train, predictors, imputed_variables)
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
        if log_level == "DEBUG":
            log_level = logging.DEBUG
        elif log_level == "INFO":
            log_level = logging.INFO
        elif log_level == "WARNING":
            log_level = logging.WARNING
        elif log_level == "ERROR":
            log_level = logging.ERROR
        elif log_level == "CRITICAL":
            log_level = logging.CRITICAL
        self.logger.setLevel(log_level)

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
    def _preprocess_data_types(
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
        # Initialize dummy information dictionary
        dummy_info = {
            "original_dtypes": {},
            "column_mapping": {},
            "original_categories": {},
        }

        data = data[predictors].copy()

        try:
            self.logger.debug(
                "Converting boolean and categorical columns to numerical format"
            )
            # Identify boolean columns and convert them to strings
            bool_columns = [
                col
                for col in data.columns
                if (
                    pd.api.types.is_bool_dtype(data[col])
                    or (
                        pd.api.types.is_integer_dtype(data[col])
                        and set(data[col].unique()) == {0, 1}
                    )
                    or (
                        pd.api.types.is_float_dtype(data[col])
                        and set(data[col].unique()) == {0.0, 1.0}
                    )
                )
            ]

            if bool_columns:
                self.logger.info(
                    f"Found {len(bool_columns)} boolean columns to convert: {bool_columns}"
                )
                for col in bool_columns:
                    dummy_info["original_dtypes"][col] = (
                        "bool",
                        data[col].dtype,
                    )
                    # For boolean columns, map the column to itself since we don't create dummies
                    dummy_info["column_mapping"][col] = [col]
                    data[col] = data[col].astype("float64")

            # Identify string and object columns (excluding already processed booleans)
            string_columns = [
                col
                for col in data.columns
                if (
                    pd.api.types.is_string_dtype(data[col])
                    or pd.api.types.is_object_dtype(data[col])
                )
                and col not in bool_columns
            ]

            # Identify numeric columns that represent categorical data
            numeric_categorical_columns = [
                col
                for col in data.columns
                if pd.api.types.is_numeric_dtype(data[col])
                and data[col].nunique()
                < 10  # Parse as category if unique count < 10
                and col
                not in bool_columns  # Exclude already processed boolean columns
            ]

            if numeric_categorical_columns:
                self.logger.info(
                    f"Found {len(numeric_categorical_columns)} numeric columns with unique values < 10, treating as categorical: {numeric_categorical_columns}. Converting to dummy variables."
                )
                for col in numeric_categorical_columns:
                    dummy_info["original_categories"][col] = [
                        float(i) for i in data[col].unique().tolist()
                    ]
                    dummy_info["original_dtypes"][col] = (
                        "numeric categorical",
                        data[col].dtype,
                    )
                    data[col] = data[col].astype("float64")
                    data[col] = data[col].astype("category")

            if string_columns:
                self.logger.info(
                    f"Found {len(string_columns)} categorical columns to convert: {string_columns}"
                )

                # Store original categories and dtypes for categorical columns
                for col in string_columns:
                    dummy_info["original_dtypes"][col] = (
                        "categorical",
                        data[col].dtype,
                    )
                    dummy_info["original_categories"][col] = (
                        data[col].unique().tolist()
                    )

            if string_columns or numeric_categorical_columns:
                # Use pandas get_dummies to create one-hot encoded features
                categorical_columns = (
                    string_columns + numeric_categorical_columns
                )
                dummy_data = pd.get_dummies(
                    data[categorical_columns],
                    columns=categorical_columns,
                    dtype="float64",
                    drop_first=True,
                )
                for col in dummy_data.columns:
                    dummy_data[col] = dummy_data[col].astype("float64")
                self.logger.debug(
                    f"Created {dummy_data.shape[1]} dummy variables from {len(categorical_columns)} categorical columns"
                )

                # Create mapping from original columns to their resulting dummy columns
                for orig_col in categorical_columns:
                    # Find all dummy columns that came from this original column
                    related_dummies = [
                        col
                        for col in dummy_data.columns
                        if col.startswith(f"{orig_col}_")
                    ]
                    dummy_info["column_mapping"][orig_col] = (
                        related_dummies
                        if len(related_dummies) > 0
                        else [orig_col]
                    )

                # Drop original string and numeric categorical columns and join the dummy variables
                numeric_data = data.drop(columns=categorical_columns)
                self.logger.debug(
                    f"Removed original string and numeric categorical columns, data shape: {numeric_data.shape}"
                )

                # Combine numeric columns with dummy variables
                data = pd.concat([numeric_data, dummy_data], axis=1)
                for col in data.columns:
                    data[col] = data[col].astype("float64")
                self.logger.info(
                    f"Data shape after dummy variable conversion: {data.shape}"
                )

            return data

        except Exception as e:
            self.logger.error(
                f"Error during string column conversion: {str(e)}"
            )
            raise RuntimeError(
                "Failed to convert string columns to dummy variables"
            ) from e

    @validate_call(config=VALIDATE_CONFIG)
    def _postprocess_imputations(
        self,
        imputations: Dict[float, pd.DataFrame],
        dummy_info: Dict[str, Any],
    ) -> Dict[float, pd.DataFrame]:
        """Convert imputed bool and categorical dummy variables back to original data types.

        This function reverses the encoding applied by preprocess_data,
        converting dummy variables back to their original boolean or categorical forms.
        For numeric categorical variables, values are rounded to the nearest valid category.

        Args:
            imputations: Dictionary mapping quantiles to DataFrames of imputed values
            dummy_info: Dictionary containing information about dummy variable mappings
                and original data types

        Returns:
            Dictionary mapping quantiles to DataFrames with original data types restored

        Raises:
            ValueError: If dummy_info is missing required information
            RuntimeError: If conversion back to original types fails
        """

        def _get_reference_category(
            orig_col: str,
            available_dummies: List[str],
            original_categories: List,
        ) -> Any:
            """Identify the reference category that was dropped during dummy encoding."""
            dummy_categories = []
            for dummy_col in available_dummies:
                # Remove the original column name and underscore prefix
                category_part = dummy_col.replace(f"{orig_col}_", "", 1)
                try:
                    # Try to convert back to original type if it was numeric
                    if (
                        category_part.replace(".", "")
                        .replace("-", "")
                        .isdigit()
                    ):
                        dummy_categories.append(float(category_part))
                    else:
                        dummy_categories.append(category_part)
                except:
                    dummy_categories.append(category_part)

            # Find which original category is missing (the reference category)
            reference_category = None
            for cat in original_categories:
                if cat not in dummy_categories:
                    reference_category = cat
                    break

            return (
                reference_category
                if reference_category is not None
                else original_categories[0]
            )

        self.logger.debug(
            f"Post-processing {len(imputations)} quantile imputations with dummy_info keys: {dummy_info.keys()}"
        )

        try:
            processed_imputations = {}

            for quantile, df in imputations.items():
                self.logger.debug(
                    f"Processing quantile {quantile} with shape {df.shape}"
                )
                df_processed = df.copy()

                for orig_col, dummy_cols in dummy_info.get(
                    "column_mapping", {}
                ).items():
                    if orig_col in dummy_info.get("original_dtypes", {}):
                        orig_dtype_info = dummy_info["original_dtypes"][
                            orig_col
                        ]

                        # Extract dtype category and original pandas dtype
                        if (
                            isinstance(orig_dtype_info, tuple)
                            and len(orig_dtype_info) == 2
                        ):
                            dtype_category, original_pandas_dtype = (
                                orig_dtype_info
                            )
                        else:
                            # Fallback for old format
                            self.logger.warning(
                                f"Unexpected dtype format for {orig_col}: {orig_dtype_info}"
                            )
                            continue

                        # Check if this variable was imputed based on its type
                        is_imputed = False
                        if dtype_category == "bool":
                            # For bool, check if original column is present
                            is_imputed = orig_col in df_processed.columns
                        elif dtype_category in [
                            "categorical",
                            "numeric categorical",
                        ]:
                            # For regular and numeric categorical, check if dummy columns are present
                            available_dummies = [
                                col
                                for col in dummy_cols
                                if col in df_processed.columns
                            ]
                            is_imputed = len(available_dummies) > 0

                        if not is_imputed:
                            self.logger.debug(
                                f"Skipping {orig_col} - not in imputed variables"
                            )
                            continue

                        self.logger.debug(
                            f"Converting {orig_col} back to {dtype_category} with original dtype {original_pandas_dtype}"
                        )

                        if dtype_category == "bool":
                            # Convert back to boolean from float (>0.5 threshold for discretization)
                            df_processed[orig_col] = (
                                df_processed[orig_col] > 0.5
                            )
                            # Convert to original boolean dtype
                            df_processed[orig_col] = df_processed[
                                orig_col
                            ].astype(original_pandas_dtype)
                            self.logger.debug(
                                f"Converted {orig_col} back to boolean type {original_pandas_dtype}"
                            )

                        elif dtype_category in [
                            "categorical",
                            "numeric categorical",
                        ]:
                            # Find available dummy columns
                            available_dummies = [
                                col
                                for col in dummy_cols
                                if col in df_processed.columns
                            ]

                            if len(available_dummies) > 0:
                                self.logger.debug(
                                    f"Converting dummy columns back to categorical {orig_col}"
                                )

                                categories = dummy_info["original_categories"][
                                    orig_col
                                ]
                                dummy_subset = df_processed[available_dummies]

                                # Identify the reference category (the one that was dropped)
                                reference_category = _get_reference_category(
                                    orig_col, available_dummies, categories
                                )

                                # Create mapping from dummy columns to their categories
                                category_mapping = {}
                                for cat in categories:
                                    dummy_name = f"{orig_col}_{cat}"
                                    if dummy_name in available_dummies:
                                        category_mapping[dummy_name] = cat

                                # Find the dummy column with highest value for each row
                                max_idx = dummy_subset.idxmax(axis=1)
                                max_values = dummy_subset.max(axis=1)

                                # If max dummy value is < 0.5, assign to reference category
                                threshold = 0.5

                                # Initialize with reference category
                                df_processed[orig_col] = reference_category

                                # Only assign to dummy categories where max value exceeds threshold
                                high_confidence_mask = max_values >= threshold
                                if high_confidence_mask.any():
                                    df_processed.loc[
                                        high_confidence_mask, orig_col
                                    ] = max_idx[high_confidence_mask].map(
                                        category_mapping
                                    )

                                # Handle any NaN values that might occur from mapping
                                nan_mask = df_processed[orig_col].isna()
                                if nan_mask.any():
                                    df_processed.loc[nan_mask, orig_col] = (
                                        reference_category
                                    )
                                    self.logger.warning(
                                        f"Some values could not be mapped for {orig_col}, using reference category: {reference_category}"
                                    )

                                self.logger.info(
                                    f"Assigned {high_confidence_mask.sum()} observations to dummy categories, "
                                    f"{(~high_confidence_mask).sum()} to reference category '{reference_category}'"
                                )

                                # Convert to original categorical type if needed
                                try:
                                    if original_pandas_dtype != "object":
                                        df_processed[orig_col] = df_processed[
                                            orig_col
                                        ].astype(original_pandas_dtype)
                                        self.logger.debug(
                                            f"Converted {orig_col} back to categorical type: {original_pandas_dtype}"
                                        )
                                except (ValueError, TypeError) as e:
                                    self.logger.warning(
                                        f"Could not convert {orig_col} to {original_pandas_dtype}: {e}"
                                    )

                                # Drop the dummy columns
                                df_processed = df_processed.drop(
                                    columns=available_dummies
                                )
                                self.logger.debug(
                                    f"Converted dummy columns back to categorical {orig_col}"
                                )
                            else:
                                self.logger.warning(
                                    f"No dummy columns found for categorical variable {orig_col}"
                                )

                processed_imputations[quantile] = df_processed
                self.logger.debug(
                    f"Processed quantile {quantile}, final shape: {df_processed.shape}"
                )

            self.logger.info(
                f"Successfully post-processed {len(processed_imputations)} quantile imputations"
            )
            return processed_imputations

        except Exception as e:
            self.logger.error(f"Error in postprocess_imputations: {str(e)}")
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
            # Validate quantiles
            self._validate_quantiles(quantiles)
        except Exception as quantile_error:
            raise ValueError(
                f"Invalid quantiles: {str(quantile_error)}"
            ) from quantile_error

        X_test = self._preprocess_data_types(X_test, self.original_predictors)

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
            imputations = self._postprocess_imputations(
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
