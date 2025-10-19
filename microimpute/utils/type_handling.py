"""Variable type detection utilities.

This module provides utilities for detecting and categorizing variable types
in pandas DataFrames, helping determine whether variables are boolean, categorical,
numeric categorical, or purely numeric.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


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
        series: pd.Series,
        col_name: str,
        logger: logging.Logger,
        force_numeric: bool = False,
    ) -> Tuple[str, Optional[List]]:
        """
        Categorize a variable and return its type and categories if applicable.

        Args:
            series: The data series to categorize
            col_name: Name of the column for logging
            logger: Logger instance
            force_numeric: If True, treat the variable as numeric even if it
                would normally be detected as numeric_categorical

        Returns:
            Tuple of (variable_type, categories)
            variable_type: 'bool', 'categorical', 'numeric_categorical', or 'numeric'
            categories: List of unique values for categorical types, None for numeric
        """
        if VariableTypeDetector.is_boolean_variable(series):
            return "bool", None

        if VariableTypeDetector.is_categorical_variable(series):
            return "categorical", series.unique().tolist()

        # Check if it would normally be numeric_categorical
        if (
            not force_numeric
            and VariableTypeDetector.is_numeric_categorical_variable(series)
        ):
            categories = [float(i) for i in series.unique().tolist()]
            logger.info(
                f"Treating numeric variable '{col_name}' as categorical due to low unique count and equal spacing"
            )
            return "numeric_categorical", categories

        # If force_numeric is True or it's not numeric_categorical, treat as numeric
        if (
            force_numeric
            and VariableTypeDetector.is_numeric_categorical_variable(series)
        ):
            logger.info(
                f"Variable '{col_name}' forced to be treated as numeric (override numeric_categorical detection)"
            )

        return "numeric", None


class DummyVariableProcessor:
    """Handles conversion of categorical predictors to dummy variables."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.dummy_mapping = {}  # Maps original column to dummy columns
        self.imputed_var_dummy_mapping = (
            {}
        )  # Pre-computed dummy info for imputed vars

    def preprocess_predictors(
        self,
        data: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        not_numeric_categorical: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Process predictor variables and pre-compute dummy encodings.

        For predictors: converts categoricals to dummies and adds to dataframe.
        For imputed_variables: pre-computes dummy info but keeps original form.

        Args:
            data: DataFrame containing the data
            predictors: List of predictor column names
            imputed_variables: List of variables to impute
            not_numeric_categorical: Optional list of variable names that should
                be treated as numeric even if they would normally be detected as
                numeric_categorical.

        Returns:
            Tuple of (processed_data, updated_predictors)
        """
        # Start with a copy containing all needed columns
        all_columns = list(set(predictors + imputed_variables))
        data = data[all_columns].copy()
        detector = VariableTypeDetector()
        not_numeric_categorical = not_numeric_categorical or []

        # Identify categorical predictors (not imputed targets)
        categorical_predictors = []
        for col in predictors:
            if col not in data.columns:
                continue
            var_type, categories = detector.categorize_variable(
                data[col],
                col,
                self.logger,
                force_numeric=(col in not_numeric_categorical),
            )
            if var_type in ["categorical", "numeric_categorical"]:
                categorical_predictors.append(col)
                self.logger.info(
                    f"Will create dummy variables for predictor '{col}' ({var_type})"
                )

        # Pre-compute dummy info for categorical imputed variables
        for col in imputed_variables:
            if col not in data.columns:
                continue
            var_type, categories = detector.categorize_variable(
                data[col],
                col,
                self.logger,
                force_numeric=(col in not_numeric_categorical),
            )
            if var_type in ["categorical", "numeric_categorical"]:
                # Create dummy columns to determine structure
                dummy_df = pd.get_dummies(
                    data[[col]],
                    columns=[col],
                    dtype="float64",
                    drop_first=True,
                )
                dummy_cols = [
                    c for c in dummy_df.columns if c.startswith(f"{col}_")
                ]

                # Store pre-computed dummy info
                self.imputed_var_dummy_mapping[col] = {
                    "dummy_cols": dummy_cols,
                    "var_type": var_type,
                    "categories": categories,
                }
                self.logger.info(
                    f"Pre-computed {len(dummy_cols)} dummy columns for imputed variable '{col}' ({var_type})"
                )
            elif var_type == "bool":
                # Track boolean imputed variables
                self.imputed_var_dummy_mapping[col] = {
                    "dummy_cols": None,
                    "var_type": "bool",
                    "categories": None,
                }

        # Process categorical predictors (add to dataframe)
        updated_predictors = predictors.copy()

        if categorical_predictors:
            # Create dummy variables for categorical predictors only
            dummy_df = pd.get_dummies(
                data[categorical_predictors],
                columns=categorical_predictors,
                dtype="float64",
                drop_first=True,
            )

            # Track mapping for each original column
            for orig_col in categorical_predictors:
                dummy_cols = [
                    col
                    for col in dummy_df.columns
                    if col.startswith(f"{orig_col}_")
                ]
                self.dummy_mapping[orig_col] = dummy_cols

                # Update predictor list
                updated_predictors.remove(orig_col)
                updated_predictors.extend(dummy_cols)

                self.logger.debug(
                    f"Created {len(dummy_cols)} dummy variables for '{orig_col}'"
                )

            # Drop original categorical columns and add dummies
            data = data.drop(columns=categorical_predictors)
            data = pd.concat([data, dummy_df], axis=1)

        # Convert boolean predictors to float (but keep as single column)
        for col in predictors:
            if col in data.columns:
                var_type, _ = detector.categorize_variable(
                    data[col], col, self.logger
                )
                if var_type == "bool":
                    data[col] = data[col].astype("float64")
                    self.logger.debug(
                        f"Converted boolean predictor '{col}' to float64"
                    )

        return data, updated_predictors

    def sequential_imputed_predictor_encoding(
        self, data: pd.DataFrame, variable: str
    ) -> pd.DataFrame:
        """
        Encode a freshly imputed variable so it can become a predictor.

        For categorical imputed variables: adds pre-computed dummy columns.
        For boolean imputed variables: converts to float64.
        For numeric variables: no change needed.

        Args:
            data: DataFrame containing the imputed variable in original form
            variable: Name of the imputed variable to encode

        Returns:
            DataFrame with encoded variable (original column kept)
        """
        data = data.copy()

        if variable not in self.imputed_var_dummy_mapping:
            # Numeric variable - no encoding needed
            return data

        var_info = self.imputed_var_dummy_mapping[variable]

        if var_info["var_type"] in ["categorical", "numeric_categorical"]:
            # Add pre-computed dummy columns
            dummy_cols = var_info["dummy_cols"]

            # Create dummies from current data
            dummy_df = pd.get_dummies(
                data[[variable]],
                columns=[variable],
                dtype="float64",
                drop_first=True,
            )

            # Ensure we have all expected dummy columns
            for dummy_col in dummy_cols:
                if dummy_col not in dummy_df.columns:
                    dummy_df[dummy_col] = 0.0

            # Keep only pre-computed dummy columns
            dummy_df = dummy_df[dummy_cols]

            # Add dummy columns to dataframe (keep original too)
            data = pd.concat([data, dummy_df], axis=1)

            self.logger.debug(
                f"Added {len(dummy_cols)} dummy columns for sequential predictor '{variable}'"
            )

        elif var_info["var_type"] == "bool":
            # Convert boolean to float (in place)
            if variable in data.columns:
                data[variable] = data[variable].astype("float64")
                self.logger.debug(
                    f"Converted boolean sequential predictor '{variable}' to float64"
                )

        return data

    def get_sequential_predictor_columns(
        self, variables: List[str]
    ) -> List[str]:
        """
        Get correct column names for sequential predictors.

        For categorical imputed variables: returns dummy column names.
        For other variables: returns original column name.

        Args:
            variables: List of variable names

        Returns:
            List of column names to use as predictors
        """
        predictor_cols = []

        for var in variables:
            if var in self.imputed_var_dummy_mapping:
                var_info = self.imputed_var_dummy_mapping[var]
                if var_info["var_type"] in [
                    "categorical",
                    "numeric_categorical",
                ]:
                    # Use dummy columns
                    predictor_cols.extend(var_info["dummy_cols"])
                else:
                    # Boolean or numeric - use original column
                    predictor_cols.append(var)
            else:
                # Not in mapping - use original column
                predictor_cols.append(var)

        return predictor_cols

    def apply_dummy_encoding_to_test(
        self,
        data: pd.DataFrame,
        predictors: List[str],
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Apply same dummy encoding to test data based on training mapping."""
        detector = VariableTypeDetector()
        data = data.copy()
        updated_predictors = predictors.copy()

        # Apply dummy encoding based on stored mapping
        for orig_col, dummy_cols in self.dummy_mapping.items():
            if orig_col in predictors and orig_col in data.columns:
                # Create dummies for this column
                dummy_df = pd.get_dummies(
                    data[[orig_col]],
                    columns=[orig_col],
                    dtype="float64",
                    drop_first=False,  # Don't drop first, we'll handle missing manually
                )

                # Ensure we have the exact dummy columns from training
                for dummy_col in dummy_cols:
                    if dummy_col not in dummy_df.columns:
                        dummy_df[dummy_col] = 0.0  # Missing category gets 0

                # Keep only the dummy columns from training
                dummy_df = dummy_df[dummy_cols]

                # Update data
                data = data.drop(columns=[orig_col])
                data = pd.concat([data, dummy_df], axis=1)

                # Update predictor list
                updated_predictors.remove(orig_col)
                updated_predictors.extend(dummy_cols)

        # Convert boolean predictors to float
        for col in predictors:
            if col in data.columns:
                var_type, _ = detector.categorize_variable(
                    data[col], col, self.logger
                )
                if var_type == "bool":
                    data[col] = data[col].astype("float64")

        return data, updated_predictors
