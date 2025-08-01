"""Data preparation utilities for imputation benchmarking.

This module provides functions for acquiring, preprocessing, and splitting data for imputation benchmarking.
It includes utilities for downloading Survey of Consumer Finances
(SCF) data, normalizing features, and creating train-test splits with consistent parameters.
"""

import io
import logging
import zipfile
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import requests
from pydantic import validate_call
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from microimpute.config import (
    RANDOM_STATE,
    TEST_SIZE,
    TRAIN_SIZE,
    VALID_YEARS,
    VALIDATE_CONFIG,
)

logger = logging.getLogger(__name__)


@validate_call(config=VALIDATE_CONFIG)
def scf_url(year: int) -> str:
    """Return the URL of the SCF summary microdata zip file for a year.

    Args:
        year: Year of SCF summary microdata to retrieve.

    Returns:
        URL of summary microdata zip file for the given year.

    Raises:
        ValueError: If the year is not in VALID_YEARS.
    """
    logger.debug(f"Generating SCF URL for year {year}")

    if year not in VALID_YEARS:
        logger.error(
            f"Invalid SCF year: {year}. Valid years are {VALID_YEARS}"
        )
        raise ValueError(
            f"The SCF is not available for {year}. Valid years are {VALID_YEARS}"
        )

    url = f"https://www.federalreserve.gov/econres/files/scfp{year}s.zip"
    logger.debug(f"Generated URL: {url}")
    return url


@validate_call(config=VALIDATE_CONFIG)
def _load(
    years: Optional[Union[int, List[int]]] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load Survey of Consumer Finances data for specified years and columns.

    Args:
        years: Year or list of years to load data for.
        columns: List of column names to load.

    Returns:
        DataFrame containing the requested data.

    Raises:
        ValueError: If no Stata files are found in the downloaded zip
            or invalid parameters
        RuntimeError: If there's a network error or a problem processing
            the downloaded data
    """

    logger.info(f"Loading SCF data with years={years}")

    try:
        # Identify years for download
        if years is None:
            years = VALID_YEARS
            logger.warning(f"Using default years: {years}")

        if isinstance(years, int):
            years = [years]

        # Validate all years are valid
        invalid_years = [year for year in years if year not in VALID_YEARS]
        if invalid_years:
            logger.error(f"Invalid years specified: {invalid_years}")
            raise ValueError(
                f"Invalid years: {invalid_years}. Valid years are {VALID_YEARS}"
            )

        all_data: List[pd.DataFrame] = []

        for year in tqdm(years):
            logger.info(f"Processing data for year {year}")
            try:
                # Download zip file
                logger.debug(f"Downloading SCF data for year {year}")
                url = scf_url(year)
                try:
                    response = requests.get(url, timeout=60)
                    response.raise_for_status()  # Raise an error for bad responses
                except requests.exceptions.RequestException as e:
                    logger.error(
                        f"Network error downloading SCF data for year {year}: {str(e)}"
                    )
                    raise RuntimeError(
                        f"Failed to download SCF data for year {year}"
                    ) from e

                # Process zip file
                try:
                    logger.debug("Creating zipfile from downloaded content")
                    z = zipfile.ZipFile(io.BytesIO(response.content))

                    # Find the .dta file in the zip
                    dta_files: List[str] = [
                        f for f in z.namelist() if f.endswith(".dta")
                    ]
                    if not dta_files:
                        logger.error(
                            f"No Stata files found in zip for year {year}"
                        )
                        raise ValueError(
                            f"No Stata files found in zip for year {year}"
                        )

                    logger.debug(f"Found Stata files: {dta_files}")

                    # Read the Stata file
                    try:
                        logger.debug(f"Reading Stata file: {dta_files[0]}")
                        with z.open(dta_files[0]) as f:
                            df = pd.read_stata(
                                io.BytesIO(f.read()), columns=columns
                            )
                            logger.debug(
                                f"Read DataFrame with shape {df.shape}"
                            )

                        # Ensure 'wgt' is included
                        if (
                            columns is not None
                            and "wgt" not in df.columns
                            and "wgt" not in columns
                        ):
                            logger.debug("Re-reading with 'wgt' column added")
                            # Re-read to include weights
                            with z.open(dta_files[0]) as f:
                                cols_with_weight: List[str] = list(
                                    set(columns) | {"wgt"}
                                )
                                df = pd.read_stata(
                                    io.BytesIO(f.read()),
                                    columns=cols_with_weight,
                                )
                                logger.debug(
                                    f"Re-read DataFrame with shape {df.shape}"
                                )
                    except Exception as e:
                        logger.error(
                            f"Error reading Stata file for year {year}: {str(e)}"
                        )
                        raise RuntimeError(
                            f"Failed to process Stata file for year {year}"
                        ) from e

                except zipfile.BadZipFile as e:
                    logger.error(f"Bad zip file for year {year}: {str(e)}")
                    raise RuntimeError(
                        f"Downloaded zip file is corrupt for year {year}"
                    ) from e

                # Add year column
                df["year"] = year
                logger.info(
                    f"Successfully processed data for year {year}, shape: {df.shape}"
                )
                all_data.append(df)

            except Exception as e:
                logger.error(f"Error processing year {year}: {str(e)}")
                raise

        # Combine all years
        logger.debug(f"Combining data from {len(all_data)} years")
        if len(all_data) > 1:
            result = pd.concat(all_data)
            logger.info(
                f"Combined data from {len(years)} years, final shape: {result.shape}"
            )
            return result
        else:
            logger.info(
                f"Returning data for single year, shape: {all_data[0].shape}"
            )
            return all_data[0]

    except Exception as e:
        logger.error(f"Error in _load: {str(e)}")
        raise


@validate_call(config=VALIDATE_CONFIG)
def prepare_scf_data(
    full_data: bool = False, years: Optional[Union[int, List[int]]] = None
) -> Union[
    Tuple[pd.DataFrame, List[str], List[str], dict],  # when full_data=True
    Tuple[
        pd.DataFrame, pd.DataFrame, List[str], List[str], dict
    ],  # when full_data=False
]:
    """Preprocess the Survey of Consumer Finances data for model training and testing.

    Args:
        full_data: Whether to return the complete dataset without splitting.
        years: Year or list of years to load data for.

    Returns:
        Different tuple formats depending on the value of full_data:
          - If full_data=True: (data, predictor_columns, imputed_columns, dummy_info)
          - If full_data=False: (train_data, test_data,
                predictor_columns, imputed_columns, dummy_info)

        Where dummy_info is a dictionary with information about dummy variables created from string columns.

    Raises:
        ValueError: If required columns are missing from the data
        RuntimeError: If data processing fails
    """
    logger.info(
        f"Preparing SCF data with full_data={full_data}, years={years}"
    )

    try:
        # Load the raw data
        logger.debug("Loading SCF data")
        data = _load(years=years)

        # Define columns needed for analysis
        # predictors shared with cps data
        PREDICTORS: List[str] = [
            "hhsex",  # sex of head of household
            "age",  # age of respondent
            "married",  # marital status of respondent
            # "kids",  # number of children in household
            "race",  # race of respondent
            "income",  # total annual income of household
            "wageinc",  # income from wages and salaries
            "bussefarminc",  # income from business, self-employment or farm
            "intdivinc",  # income from interest and dividends
            "ssretinc",  # income from social security and retirement accounts
            "lf",  # labor force status
        ]

        IMPUTED_VARIABLES: List[str] = ["networth"]

        # Validate that all required columns exist in the data
        missing_columns = [
            col
            for col in PREDICTORS + IMPUTED_VARIABLES
            if col not in data.columns
        ]
        if missing_columns:
            logger.error(
                f"Required columns missing from SCF data: {missing_columns}"
            )
            raise ValueError(
                f"Required columns missing from SCF data: {missing_columns}"
            )

        logger.debug(
            f"Selecting {len(PREDICTORS)} predictors and {len(IMPUTED_VARIABLES)} imputation variables"
        )
        data = data[PREDICTORS + IMPUTED_VARIABLES]
        logger.debug(f"Data shape after column selection: {data.shape}")

        if full_data:
            logger.info("Processing full dataset without splitting")
            data = preprocess_data(data, full_data=True)
            logger.info(
                f"Returning full processed dataset with shape {data.shape}"
            )
            return data, PREDICTORS, IMPUTED_VARIABLES
        else:
            logger.info("Splitting data into train and test sets")
            X_train, X_test = preprocess_data(data)
            logger.info(
                f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}"
            )
            return X_train, X_test, PREDICTORS, IMPUTED_VARIABLES

    except Exception as e:
        logger.error(f"Error in prepare_scf_data: {str(e)}")
        raise RuntimeError(f"Failed to prepare SCF data: {str(e)}") from e


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

        except Exception as e:
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

        except Exception as e:
            logger.error(f"Error in processing data: {str(e)}")
            raise
