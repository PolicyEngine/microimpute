"""Data preparation and transformation utilities

This module provides comprehensive data preparation functions for imputation workflows,
including data splitting, normalization, unnormalization, and categorical variable handling.
These utilities ensure consistent data preprocessing across different imputation methods.

Key functions:
    - preprocess_data: split and optionally normalize data for training/testing
    - unnormalize_predictions: convert normalized predictions back to original scale
    - Handle categorical variables through one-hot encoding
"""

import logging
from typing import Optional, Tuple, Union

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
def preprocess_data(
    data: pd.DataFrame,
    full_data: Optional[bool] = False,
    train_size: Optional[float] = TRAIN_SIZE,
    test_size: Optional[float] = TEST_SIZE,
    random_state: Optional[int] = RANDOM_STATE,
    normalize: Optional[bool] = False,
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
        normalize: Whether to normalize the data.

    Returns:
        Different tuple formats depending on the value of full_data:
          - If full_data=True: (data, dummy_info)
          - If full_data=False: (X_train, X_test, dummy_info)

        Where dummy_info is a dictionary mapping original columns to their resulting dummy columns

    Raises:
        ValueError: If data is empty or invalid
        RuntimeError: If data preprocessing fails
    """

    logger.debug(
        f"Preprocessing data with shape {data.shape}, full_data={full_data}"
    )

    if data.empty:
        raise ValueError("Data must not be None or empty")
    # Check for missing values
    missing_count = data.isna().sum().sum()
    if missing_count > 0:
        logger.warning(f"Data contains {missing_count} missing values")

    if normalize:
        logger.debug("Normalizing data")
        try:
            mean = data.mean(axis=0)
            std = data.std(axis=0)

            # Check for constant columns (std=0)
            constant_cols = std[std == 0].index.tolist()
            if constant_cols:
                logger.warning(
                    f"Found constant columns (std=0): {constant_cols}"
                )
                # Handle constant columns by setting std to 1 to avoid division by zero
                for col in constant_cols:
                    std[col] = 1

            # Apply normalization
            data = (data - mean) / std
            logger.debug("Data normalized successfully")

            # Store normalization parameters
            normalization_params = {
                col: {"mean": mean[col], "std": std[col]}
                for col in data.columns
            }

            logger.debug(f"Normalization parameters: {normalization_params}")

        except (TypeError, AttributeError) as e:
            logger.error(f"Error during data normalization: {str(e)}")
            raise RuntimeError("Failed to normalize data") from e

    if full_data and normalize:
        logger.info("Returning full preprocessed dataset")
        return (
            data,
            normalization_params,
        )
    elif full_data:
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
            if normalize:
                return (
                    X_train,
                    X_test,
                    normalization_params,
                )
            else:
                return (
                    X_train,
                    X_test,
                )

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
