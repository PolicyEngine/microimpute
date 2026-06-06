"""Data validation utilities for imputation operations

This module provides centralized validation functions used throughout the microimpute
package to ensure data integrity, validate parameters, and maintain consistency
across different imputation methods.

The validation functions help prevent common errors and provide clear error messages
when data or parameters don't meet requirements.
"""

import logging
from typing import List, Optional

import pandas as pd

log = logging.getLogger(__name__)


def validate_quantiles(quantiles: List[float]) -> None:
    """Validate that quantiles are within [0, 1] range.

    Args:
        quantiles: List of quantile values to validate.

    Raises:
        ValueError: If any quantile is outside [0, 1] range.
    """
    invalid_quantiles = [q for q in quantiles if not 0 <= q <= 1]
    if invalid_quantiles:
        error_msg = (
            f"Invalid quantiles (must be between 0 and 1): {invalid_quantiles}"
        )
        log.error(error_msg)
        raise ValueError(error_msg)


def validate_columns_exist(
    data: pd.DataFrame, columns: List[str], data_name: str = "data"
) -> None:
    """Validate that specified columns exist in the DataFrame.

    Args:
        data: DataFrame to check.
        columns: List of column names that should exist.
        data_name: Name of the dataset for error messages.

    Raises:
        ValueError: If any columns are missing from the DataFrame.
    """
    missing_columns = [col for col in columns if col not in data.columns]
    if missing_columns:
        error_msg = f"Missing columns in {data_name}: {missing_columns}"
        log.error(error_msg)
        raise ValueError(error_msg)


def validate_dataframe_compatibility(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    check_columns: Optional[List[str]] = None,
    require_same_length: bool = True,
) -> None:
    """Validate that two DataFrames are compatible for operations.

    Args:
        df1: First DataFrame.
        df2: Second DataFrame.
        check_columns: If provided, verify these columns exist in both DataFrames.
        require_same_length: If True, verify DataFrames have the same number of rows.

    Raises:
        ValueError: If DataFrames are incompatible.
    """
    if require_same_length and len(df1) != len(df2):
        error_msg = (
            f"Length mismatch: first DataFrame has {len(df1)} rows, "
            f"second DataFrame has {len(df2)} rows"
        )
        log.error(error_msg)
        raise ValueError(error_msg)

    if check_columns:
        validate_columns_exist(df1, check_columns, "first DataFrame")
        validate_columns_exist(df2, check_columns, "second DataFrame")


def validate_imputation_inputs(
    donor_data: pd.DataFrame,
    receiver_data: pd.DataFrame,
    predictors: List[str],
    imputed_variables: List[str],
    weight_col: Optional[str] = None,
) -> None:
    """Validate inputs for imputation operations.

    Args:
        donor_data: Training data containing predictors and target variables.
        receiver_data: Data where imputations will be applied.
        predictors: List of predictor column names.
        imputed_variables: List of variables to impute.
        weight_col: Optional weight column name.

    Raises:
        ValueError: If validation fails.
    """
    # Validate donor data has all required columns
    validate_columns_exist(donor_data, predictors, "donor data")
    validate_columns_exist(donor_data, imputed_variables, "donor data")

    # Validate receiver data has predictor columns
    validate_columns_exist(receiver_data, predictors, "receiver data")

    # Validate weight column if provided
    if weight_col:
        validate_columns_exist(donor_data, [weight_col], "donor data")
