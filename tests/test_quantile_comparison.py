"""Tests for the end-to-end quantile loss comparison workflow.

This module tests the complete workflow of:
1. Preparing data
2. Training different imputation models
3. Generating predictions
4. Comparing models using quantile loss metrics
5. Visualizing the results
"""

from typing import List, Type

import io
import pandas as pd
import requests
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import zipfile

from microimpute.comparisons import *
from microimpute.config import RANDOM_STATE, VALID_YEARS
from microimpute.models import *
from microimpute.visualizations.plotting import *
from microimpute.utils.data import preprocess_data


def test_quantile_comparison_diabetes() -> None:
    """Test the end-to-end quantile loss comparison workflow."""
    diabetes_data = load_diabetes()
    diabetes_df = pd.DataFrame(
        diabetes_data.data, columns=diabetes_data.feature_names
    )

    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1", "s4"]

    diabetes_df = diabetes_df[predictors + imputed_variables]
    X_train, X_test = train_test_split(
        diabetes_df, test_size=0.2, random_state=RANDOM_STATE
    )

    Y_test: pd.DataFrame = X_test[imputed_variables]

    model_classes: List[Type[Imputer]] = [QRF, OLS, QuantReg, Matching]
    method_imputations = get_imputations(
        model_classes, X_train, X_test, predictors, imputed_variables
    )

    loss_comparison_df = compare_quantile_loss(
        Y_test, method_imputations, imputed_variables
    )

    assert not loss_comparison_df.isna().any().any()

    loss_comparison_df.to_csv("diabetes_comparison_results.csv")

    comparison_viz = method_comparison_results(
        data=loss_comparison_df,
        metric_name="Test Quantile Loss",
        data_format="long",  # Explicitly using wide format
    )
    fig = comparison_viz.plot(
        title="Method Comparison for Diabetes Dataset",
        show_mean=True,
        save_path="diabetes_model_comparison.jpg",
    )


def test_quantile_comparison_scf() -> None:
    """Test the end-to-end quantile loss comparison workflow on the scf data set."""
    scf_data = load_scf(2022)
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

    X_train, X_test = preprocess_data(
        data=scf_data,
        full_data=False,
        normalize=False,
    )

    # Shrink down the data by sampling
    X_train = X_train.sample(frac=0.01, random_state=RANDOM_STATE)
    X_test = X_test.sample(frac=0.01, random_state=RANDOM_STATE)

    Y_test: pd.DataFrame = X_test[IMPUTED_VARIABLES]

    model_classes: List[Type[Imputer]] = [QRF, OLS, QuantReg, Matching]
    method_imputations = get_imputations(
        model_classes, X_train, X_test, PREDICTORS, IMPUTED_VARIABLES
    )

    loss_comparison_df = compare_quantile_loss(
        Y_test, method_imputations, IMPUTED_VARIABLES
    )

    assert not loss_comparison_df.isna().any().any()

    loss_comparison_df.to_csv("scf_comparison_results.csv")

    comparison_viz = method_comparison_results(
        data=loss_comparison_df,
        metric_name="Test Quantile Loss",
        data_format="long",  # Explicitly using wide format
    )
    fig = comparison_viz.plot(
        title="Method Comparison for SCF Dataset",
        show_mean=True,
        save_path="scf_model_comparison.jpg",
    )


@validate_call(config=VALIDATE_CONFIG)
def load_scf(
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

    def scf_url(year: int) -> str:
        """Return the URL of the SCF summary microdata zip file for a year."""
        logger.debug(f"Generating SCF URL for year {year}")

        if year not in VALID_YEARS:
            logger.error(
                f"Invalid SCF year: {year}. Valid years are {VALID_YEARS}"
            )
            raise

        url = f"https://www.federalreserve.gov/econres/files/scfp{year}s.zip"
        logger.debug(f"Generated URL: {url}")
        return url

    logger.info(f"Loading SCF data with years={years}")

    try:
        # Identify years for download
        if years is None:
            years = VALID_YEARS
            logger.warning(f"Using default years: {years}")

        if isinstance(years, int):
            years = [years]

        all_data: List[pd.DataFrame] = []

        for year in years:
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
                    raise

                # Process zip file
                z = zipfile.ZipFile(io.BytesIO(response.content))
                # Find the .dta file in the zip
                dta_files: List[str] = [
                    f for f in z.namelist() if f.endswith(".dta")
                ]
                if not dta_files:
                    logger.error(
                        f"No Stata files found in zip for year {year}"
                    )
                    raise

                # Read the Stata file
                try:
                    logger.debug(f"Reading Stata file: {dta_files[0]}")
                    with z.open(dta_files[0]) as f:
                        df = pd.read_stata(
                            io.BytesIO(f.read()), columns=columns
                        )
                        logger.debug(f"Read DataFrame with shape {df.shape}")
                except Exception as e:
                    logger.error(
                        f"Error reading Stata file for year {year}: {str(e)}"
                    )
                    raise

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
