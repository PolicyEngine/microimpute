"""Tests for the end-to-end quantile loss comparison workflow.

This module tests the complete workflow of:
1. Preparing data
2. Training different imputation models
3. Generating predictions
4. Comparing models using quantile loss metrics
5. Visualizing the results
"""

from typing import List, Type

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from microimpute.comparisons import *
from microimpute.config import RANDOM_STATE
from microimpute.models import *
from microimpute.visualizations.plotting import *


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
    X_train, X_test, PREDICTORS, IMPUTED_VARIABLES = prepare_scf_data(
        full_data=False, years=2019
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
