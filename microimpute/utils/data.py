"""Data preparation and transformation utilities

This module provides comprehensive data preparation functions for imputation workflows,
including data splitting, normalization, log transformation, asinh transformation,
and categorical variable handling.
These utilities ensure consistent data preprocessing across different imputation methods.

Key functions:
    - preprocess_data: split and optionally transform data for training/testing
    - unnormalize_predictions: convert normalized predictions back to original scale
    - unlog_transform_predictions: convert log-transformed predictions back to original scale
    - un_asinh_transform_predictions: convert asinh-transformed predictions back to original scale
    - Handle categorical variables through one-hot encoding
"""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import validate_call
from sklearn.model_selection import train_test_split

from microimpute.config import (
    RANDOM_STATE,
    TEST_SIZE,
    TRAIN_SIZE,
    VALIDATE_CONFIG,
)

logger = logging.getLogger(__name__)


@validate_call(config=VALIDATE_CONFIG)
def normalize_data(
    data: pd.DataFrame,
    columns_to_normalize: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Normalize numeric columns in a DataFrame.

    Categorical and boolean columns are excluded from normalization
    to prevent issues when they are later encoded as dummy variables.

    Args:
        data: DataFrame to normalize.
        columns_to_normalize: Optional list of specific columns to normalize.
            If None, all numeric columns will be normalized.

    Returns:
        Tuple of (normalized_data, normalization_params)
        where normalization_params is a dict mapping column names
        to {"mean": float, "std": float}.

    Raises:
        ValueError: If specified columns don't exist in data.
        RuntimeError: If normalization fails.
    """
    logger.debug("Normalizing data")
    try:
        from microimpute.utils.type_handling import VariableTypeDetector

        # Identify categorical columns to exclude from normalization
        detector = VariableTypeDetector()
        categorical_cols = []
        for col in data.columns:
            var_type, _ = detector.categorize_variable(data[col], col, logger)
            if var_type in ["categorical", "numeric_categorical", "bool"]:
                categorical_cols.append(col)

        if categorical_cols:
            logger.info(
                f"Excluding categorical columns from normalization: {categorical_cols}"
            )

        # Determine which columns to normalize
        if columns_to_normalize is not None:
            # Validate that specified columns exist
            missing_cols = set(columns_to_normalize) - set(data.columns)
            if missing_cols:
                error_msg = (
                    f"Columns specified for normalization not found in "
                    f"data: {missing_cols}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Only normalize specified columns that are not categorical
            numeric_cols = [
                col
                for col in columns_to_normalize
                if col not in categorical_cols
            ]

            # Warn if user specified categorical columns
            specified_categorical = [
                col for col in columns_to_normalize if col in categorical_cols
            ]
            if specified_categorical:
                logger.warning(
                    f"Skipping normalization for categorical columns: "
                    f"{specified_categorical}"
                )
        else:
            # Get all numeric columns for normalization
            numeric_cols = [
                col for col in data.columns if col not in categorical_cols
            ]

        if not numeric_cols:
            logger.warning("No numeric columns found for normalization")
            return data.copy(), {}

        # Normalize only numeric columns
        data = data.copy()
        mean = data[numeric_cols].mean(axis=0)
        std = data[numeric_cols].std(axis=0)

        # Check for constant columns (std=0)
        constant_cols = std[std == 0].index.tolist()
        if constant_cols:
            logger.warning(f"Found constant columns (std=0): {constant_cols}")
            # Handle constant columns by setting std to 1 to avoid division by zero
            for col in constant_cols:
                std[col] = 1

        # Apply normalization only to numeric columns
        data[numeric_cols] = (data[numeric_cols] - mean) / std
        logger.debug(
            f"Normalized {len(numeric_cols)} numeric columns successfully"
        )

        # Store normalization parameters only for numeric columns
        normalization_params = {
            col: {"mean": mean[col], "std": std[col]} for col in numeric_cols
        }

        logger.debug(f"Normalization parameters: {normalization_params}")

        return data, normalization_params

    except (TypeError, AttributeError) as e:
        logger.error(f"Error during data normalization: {str(e)}")
        raise RuntimeError("Failed to normalize data") from e


@validate_call(config=VALIDATE_CONFIG)
def log_transform_data(
    data: pd.DataFrame,
    columns_to_transform: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Apply log transformation to numeric columns in a DataFrame.

    Categorical and boolean columns are excluded from transformation
    to prevent issues when they are later encoded as dummy variables.

    Args:
        data: DataFrame to log transform.
        columns_to_transform: Optional list of specific columns to
            log transform. If None, all numeric columns will be transformed.

    Returns:
        Tuple of (log_transformed_data, log_transform_params)
        where log_transform_params is a dict mapping column names
        to {} for reversing the transformation.

    Raises:
        ValueError: If data contains non-positive values in numeric columns
            or if specified columns don't exist in data.
        RuntimeError: If log transformation fails.
    """
    logger.debug("Applying log transformation to data")
    try:
        from microimpute.utils.type_handling import VariableTypeDetector

        # Identify categorical columns to exclude from log transformation
        detector = VariableTypeDetector()
        categorical_cols = []
        for col in data.columns:
            var_type, _ = detector.categorize_variable(data[col], col, logger)
            if var_type in ["categorical", "numeric_categorical", "bool"]:
                categorical_cols.append(col)

        if categorical_cols:
            logger.info(
                f"Excluding categorical columns from log transformation: {categorical_cols}"
            )

        # Determine which columns to transform
        if columns_to_transform is not None:
            # Validate that specified columns exist
            missing_cols = set(columns_to_transform) - set(data.columns)
            if missing_cols:
                error_msg = (
                    f"Columns specified for log transformation not found "
                    f"in data: {missing_cols}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Only transform specified columns that are not categorical
            numeric_cols = [
                col
                for col in columns_to_transform
                if col not in categorical_cols
            ]

            # Warn if user specified categorical columns
            specified_categorical = [
                col for col in columns_to_transform if col in categorical_cols
            ]
            if specified_categorical:
                logger.warning(
                    f"Skipping log transformation for categorical "
                    f"columns: {specified_categorical}"
                )
        else:
            # Get all numeric columns for log transformation
            numeric_cols = [
                col for col in data.columns if col not in categorical_cols
            ]

        if not numeric_cols:
            logger.warning("No numeric columns found for log transformation")
            return data.copy(), {}

        # Check for non-positive values in numeric columns
        data_copy = data.copy()
        for col in numeric_cols:
            min_val = data_copy[col].min()
            if min_val <= 0:
                error_msg = (
                    f"Column '{col}' contains non-positive values "
                    f"(min={min_val}). Log transformation requires all "
                    f"positive values."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        # Apply log transformation only to numeric columns
        log_transform_params = {}
        for col in numeric_cols:
            data_copy[col] = np.log(data_copy[col])
            log_transform_params[col] = {}

        logger.debug(
            f"Log transformed {len(numeric_cols)} numeric columns successfully"
        )
        logger.debug(f"Log transformation parameters: {log_transform_params}")

        return data_copy, log_transform_params

    except ValueError:
        # Re-raise ValueError as-is (for non-positive values)
        raise
    except (TypeError, AttributeError) as e:
        logger.error(f"Error during log transformation: {str(e)}")
        raise RuntimeError("Failed to apply log transformation") from e


@validate_call(config=VALIDATE_CONFIG)
def asinh_transform_data(
    data: pd.DataFrame,
    columns_to_transform: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Apply inverse hyperbolic sine (asinh) transformation to numeric columns.

    The asinh transformation is similar to log transformation but handles
    zero and negative values. It behaves like log(2x) for large positive x,
    like -log(-2x) for large negative x, and like x near zero.

    Categorical and boolean columns are excluded from transformation
    to prevent issues when they are later encoded as dummy variables.

    Args:
        data: DataFrame to transform.
        columns_to_transform: Optional list of specific columns to transform.
            If None, all numeric columns will be transformed.

    Returns:
        Tuple of (asinh_transformed_data, asinh_transform_params)
        where asinh_transform_params is a dict mapping column names to {}.

    Raises:
        ValueError: If specified columns don't exist in data.
        RuntimeError: If asinh transformation fails.
    """
    logger.debug("Applying asinh transformation to data")
    try:
        from microimpute.utils.type_handling import VariableTypeDetector

        # Identify categorical columns to exclude from asinh transformation
        detector = VariableTypeDetector()
        categorical_cols = []
        for col in data.columns:
            var_type, _ = detector.categorize_variable(data[col], col, logger)
            if var_type in ["categorical", "numeric_categorical", "bool"]:
                categorical_cols.append(col)

        if categorical_cols:
            logger.info(
                f"Excluding categorical columns from asinh transformation: "
                f"{categorical_cols}"
            )

        # Determine which columns to transform
        if columns_to_transform is not None:
            # Validate that specified columns exist
            missing_cols = set(columns_to_transform) - set(data.columns)
            if missing_cols:
                error_msg = (
                    f"Columns specified for asinh transformation not found "
                    f"in data: {missing_cols}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Only transform specified columns that are not categorical
            numeric_cols = [
                col
                for col in columns_to_transform
                if col not in categorical_cols
            ]

            # Warn if user specified categorical columns
            specified_categorical = [
                col for col in columns_to_transform if col in categorical_cols
            ]
            if specified_categorical:
                logger.warning(
                    f"Skipping asinh transformation for categorical "
                    f"columns: {specified_categorical}"
                )
        else:
            # Get all numeric columns for asinh transformation
            numeric_cols = [
                col for col in data.columns if col not in categorical_cols
            ]

        if not numeric_cols:
            logger.warning("No numeric columns found for asinh transformation")
            return data.copy(), {}

        # Apply asinh transformation only to numeric columns
        data_copy = data.copy()
        asinh_transform_params = {}
        for col in numeric_cols:
            data_copy[col] = np.arcsinh(data_copy[col])
            asinh_transform_params[col] = {}

        logger.debug(
            f"Asinh transformed {len(numeric_cols)} numeric columns successfully"
        )
        logger.debug(
            f"Asinh transformation parameters: {asinh_transform_params}"
        )

        return data_copy, asinh_transform_params

    except ValueError:
        # Re-raise ValueError as-is
        raise
    except (TypeError, AttributeError) as e:
        logger.error(f"Error during asinh transformation: {str(e)}")
        raise RuntimeError("Failed to apply asinh transformation") from e


@validate_call(config=VALIDATE_CONFIG)
def preprocess_data(
    data: pd.DataFrame,
    full_data: Optional[bool] = False,
    train_size: Optional[float] = TRAIN_SIZE,
    test_size: Optional[float] = TEST_SIZE,
    random_state: Optional[int] = RANDOM_STATE,
    normalize: Optional[Union[bool, List[str]]] = False,
    log_transform: Optional[Union[bool, List[str]]] = False,
    asinh_transform: Optional[Union[bool, List[str]]] = False,
) -> Union[
    Tuple[pd.DataFrame, dict],  # when full_data=True
    Tuple[pd.DataFrame, pd.DataFrame, dict],  # when full_data=False
]:
    """Preprocess the data for model training and testing.

    Args:
        data: DataFrame containing the data to preprocess.
        full_data: Whether to return the complete dataset without splitting.
        train_size: Proportion of the dataset to include in the train split.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Random seed for reproducibility.
        normalize: Whether to normalize the data. Can be:
            - True: normalize all numeric columns
            - List of column names: normalize only those columns
            - False: no normalization (default)
        log_transform: Whether to apply log transformation to the data. Can be:
            - True: transform all numeric columns
            - List of column names: transform only those columns
            - False: no transformation (default)
        asinh_transform: Whether to apply asinh transformation to the data.
            The asinh transformation handles zero and negative values unlike
            log. Can be:
            - True: transform all numeric columns
            - List of column names: transform only those columns
            - False: no transformation (default)

    Returns:
        Different tuple formats depending on parameters:
          - If full_data=True and transformations applied:
            (data, transform_params)
          - If full_data=True and no transformations:
            data
          - If full_data=False and transformations applied:
            (X_train, X_test, transform_params)
          - If full_data=False and no transformations:
            (X_train, X_test)

        Where transform_params is a dict with keys:
          - "normalization": dict of normalization parameters (or empty dict)
          - "log_transform": dict of log transform parameters (or empty dict)
          - "asinh_transform": dict of asinh transform parameters (or empty dict)

    Raises:
        ValueError: If data is empty or invalid, or if multiple transformations
            would apply to the same columns, or if log_transform is applied
            to data with non-positive values, or if specified columns don't
            exist in data.
        RuntimeError: If data preprocessing fails
    """

    logger.debug(
        f"Preprocessing data with shape {data.shape}, full_data={full_data}"
    )

    if data.empty:
        raise ValueError("Data must not be None or empty")

    # Check which transformations are requested
    normalize_requested = normalize is not False and normalize != []
    log_transform_requested = (
        log_transform is not False and log_transform != []
    )
    asinh_transform_requested = (
        asinh_transform is not False and asinh_transform != []
    )

    # Collect transformation settings for conflict checking
    transforms = []
    if normalize_requested:
        transforms.append(("normalize", normalize))
    if log_transform_requested:
        transforms.append(("log_transform", log_transform))
    if asinh_transform_requested:
        transforms.append(("asinh_transform", asinh_transform))

    # Validate that multiple transformations don't conflict
    if len(transforms) > 1:
        # Check if any are True (apply to all columns)
        any_true = any(t[1] is True for t in transforms)
        if any_true:
            names = [t[0] for t in transforms]
            error_msg = (
                f"Cannot apply multiple transformations ({', '.join(names)}) "
                f"to all columns. Please specify column lists for each to "
                f"ensure they apply to different variables."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # All are lists - check for overlaps between any pair
        col_sets = {}
        for name, cols in transforms:
            col_sets[name] = set(cols) if isinstance(cols, list) else set()

        # Check all pairs for overlap
        transform_names = list(col_sets.keys())
        for i, name1 in enumerate(transform_names):
            for name2 in transform_names[i + 1 :]:
                overlap = col_sets[name1] & col_sets[name2]
                if overlap:
                    error_msg = (
                        f"Cannot apply both {name1} and {name2} to the same "
                        f"columns: {overlap}. Each column can only have one "
                        f"transformation applied."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)

    # Check for missing values
    missing_count = data.isna().sum().sum()
    if missing_count > 0:
        logger.warning(f"Data contains {missing_count} missing values")

    # Apply normalization if requested
    normalization_params = {}
    if normalize_requested:
        if isinstance(normalize, bool):
            # normalize=True means normalize all numeric columns
            data, normalization_params = normalize_data(data)
        else:
            # normalize is a list of specific columns
            data, normalization_params = normalize_data(
                data, columns_to_normalize=normalize
            )

    # Apply log transformation if requested
    log_transform_params = {}
    if log_transform_requested:
        if isinstance(log_transform, bool):
            # log_transform=True means transform all numeric columns
            data, log_transform_params = log_transform_data(data)
        else:
            # log_transform is a list of specific columns
            data, log_transform_params = log_transform_data(
                data, columns_to_transform=log_transform
            )

    # Apply asinh transformation if requested
    asinh_transform_params = {}
    if asinh_transform_requested:
        if isinstance(asinh_transform, bool):
            # asinh_transform=True means transform all numeric columns
            data, asinh_transform_params = asinh_transform_data(data)
        else:
            # asinh_transform is a list of specific columns
            data, asinh_transform_params = asinh_transform_data(
                data, columns_to_transform=asinh_transform
            )

    # Prepare transformation parameters to return
    has_transformations = (
        normalize_requested
        or log_transform_requested
        or asinh_transform_requested
    )
    if has_transformations:
        # Merge parameter dicts, with a key to distinguish them
        transform_params = {
            "normalization": normalization_params,
            "log_transform": log_transform_params,
            "asinh_transform": asinh_transform_params,
        }

    if full_data:
        if has_transformations:
            logger.info(
                "Returning full preprocessed dataset with transformations"
            )
            return (data, transform_params)
        else:
            logger.info("Returning full preprocessed dataset")
            return data
    else:
        logger.debug(
            f"Splitting data with train_size={train_size}, test_size={test_size}"
        )
        try:
            X_train, X_test = train_test_split(
                data,
                test_size=test_size,
                train_size=train_size,
                random_state=random_state,
            )
            logger.info(
                f"Data split into train ({X_train.shape}) and test ({X_test.shape}) sets"
            )
            if has_transformations:
                return (X_train, X_test, transform_params)
            else:
                return (X_train, X_test)

        except (ValueError, TypeError) as e:
            logger.error(f"Error in processing data: {str(e)}")
            raise


@validate_call(config=VALIDATE_CONFIG)
def unnormalize_predictions(
    imputations: dict, normalization_params: dict
) -> dict:
    """Unnormalize predictions using stored normalization parameters.

    Args:
        imputations: Dictionary mapping quantiles to DataFrames of predictions.
        normalization_params: Dictionary with mean and std for each column.

    Returns:
        Dictionary with same structure as imputations but with unnormalized values.

    Raises:
        ValueError: If columns in imputations don't match normalization parameters.
    """
    logger.debug(f"Unnormalizing predictions for {len(imputations)} quantiles")

    # Extract mean and std from normalization parameters
    mean = pd.Series(
        {col: p["mean"] for col, p in normalization_params.items()}
    )
    std = pd.Series({col: p["std"] for col, p in normalization_params.items()})

    unnormalized = {}
    for q, df in imputations.items():
        cols = df.columns

        # Check that all columns have normalization parameters
        missing_params = [col for col in cols if col not in mean.index]
        if missing_params:
            error_msg = f"Missing normalization parameters for columns: {missing_params}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Unnormalize: x_original = x_normalized * std + mean
        df_unnorm = df.mul(std[cols], axis=1).add(mean[cols], axis=1)
        unnormalized[q] = df_unnorm

        logger.debug(f"Unnormalized quantile {q} with shape {df_unnorm.shape}")

    return unnormalized


@validate_call(config=VALIDATE_CONFIG)
def unlog_transform_predictions(
    imputations: dict, log_transform_params: dict
) -> dict:
    """Reverse log transformation on predictions using stored parameters.

    Args:
        imputations: Dictionary mapping quantiles to DataFrames of predictions.
        log_transform_params: Dictionary with column names that were
            log-transformed.

    Returns:
        Dictionary with same structure as imputations but with
        un-log-transformed values.

    Raises:
        ValueError: If columns in imputations don't match log transformation
            parameters.
    """
    logger.debug(
        f"Reversing log transformation for {len(imputations)} quantiles"
    )

    untransformed = {}
    for q, df in imputations.items():
        cols = df.columns

        # Check that all columns have log transformation parameters
        missing_params = [
            col for col in cols if col not in log_transform_params
        ]
        if missing_params:
            error_msg = (
                f"Missing log transformation parameters for columns: "
                f"{missing_params}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Reverse log transformation: x_original = exp(x_log)
        df_untransformed = df.copy()
        for col in cols:
            df_untransformed[col] = np.exp(df[col])
        untransformed[q] = df_untransformed

        logger.debug(
            f"Reversed log transformation for quantile {q} with shape "
            f"{df_untransformed.shape}"
        )

    return untransformed


@validate_call(config=VALIDATE_CONFIG)
def un_asinh_transform_predictions(
    imputations: dict, asinh_transform_params: dict
) -> dict:
    """Reverse asinh transformation on predictions using stored parameters.

    Args:
        imputations: Dictionary mapping quantiles to DataFrames of predictions.
        asinh_transform_params: Dictionary with column names that were
            asinh-transformed.

    Returns:
        Dictionary with same structure as imputations but with
        un-asinh-transformed values (sinh applied).

    Raises:
        ValueError: If columns in imputations don't match asinh transformation
            parameters.
    """
    logger.debug(
        f"Reversing asinh transformation for {len(imputations)} quantiles"
    )

    untransformed = {}
    for q, df in imputations.items():
        cols = df.columns

        # Check that all columns have asinh transformation parameters
        missing_params = [
            col for col in cols if col not in asinh_transform_params
        ]
        if missing_params:
            error_msg = (
                f"Missing asinh transformation parameters for columns: "
                f"{missing_params}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Reverse asinh transformation: x_original = sinh(x_asinh)
        df_untransformed = df.copy()
        for col in cols:
            df_untransformed[col] = np.sinh(df[col])
        untransformed[q] = df_untransformed

        logger.debug(
            f"Reversed asinh transformation for quantile {q} with shape "
            f"{df_untransformed.shape}"
        )

    return untransformed
