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
