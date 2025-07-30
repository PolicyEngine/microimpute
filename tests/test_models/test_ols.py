"""Tests for the OLS (Ordinary Least Squares) imputation model."""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

from microimpute.comparisons.data import preprocess_data
from microimpute.config import QUANTILES
from microimpute.evaluations import *
from microimpute.models.ols import OLS
from microimpute.visualizations.plotting import *

# Test Method on diabetes dataset
diabetes_data = load_diabetes()
diabetes_df = pd.DataFrame(
    diabetes_data.data, columns=diabetes_data.feature_names
)

predictors = ["age", "sex", "bmi", "bp"]
imputed_variables = ["s1", "s4"]

diabetes_df = diabetes_df[predictors + imputed_variables]


def test_ols_cross_validation(
    data: pd.DataFrame = diabetes_df,
    predictors: List[str] = predictors,
    imputed_variables: List[str] = imputed_variables,
    quantiles: List[float] = QUANTILES,
) -> None:
    """
    Test the OLS model on a specific dataset.

    Args:
            data: DataFrame with the dataset of interest.
            predictors: List of predictor variables.
            imputed_variables: List of variables to impute.
            quantiles: List of quantiles to predict.
    """
    ols_results = cross_validate_model(
        OLS, data, predictors, imputed_variables
    )

    ols_results.to_csv("ols_cv_results.csv")

    assert not ols_results.isna().any().any()

    perf_results_viz = model_performance_results(
        results=ols_results,
        model_name="OLS",
        method_name="Cross-Validation Quantile Loss Average",
    )
    fig = perf_results_viz.plot(
        title="OLS Cross-Validation Performance",
        save_path="ols_cv_performance.jpg",
    )


def test_ols_example(
    data: pd.DataFrame = diabetes_df,
    predictors: List[str] = predictors,
    imputed_variables: List[str] = imputed_variables,
    quantiles: List[float] = QUANTILES,
) -> None:
    """
    Example of how to use the OLS imputer model.

    This example demonstrates:
    - Initializing an OLS model
    - Fitting the model to training data
    - Predicting quantiles on test data
    - How OLS models assume normally distributed residuals

    Args:
        data: DataFrame with the dataset to use.
        predictors: List of predictor column names.
        imputed_variables: List of target column names.
        quantiles: List of quantiles to predict.
    """
    X_train, X_test = preprocess_data(data)

    # Initialize OLS model
    model = OLS()

    # Fit the model
    fitted_model = model.fit(X_train, predictors, imputed_variables)

    # Predict at multiple quantiles
    predictions: Dict[float, pd.DataFrame] = fitted_model.predict(
        X_test,
        quantiles,
        random_quantile_sample=False,
    )

    # Check structure of predictions
    assert isinstance(predictions, dict)
    assert set(predictions.keys()) == set(quantiles)

    # Demonstrate how OLS uses normal distribution assumption
    median_pred = predictions[0.5]
    q10_pred = predictions[0.1]
    q90_pred = predictions[0.9]

    # The difference between q90 and median should approximately equal
    # the difference between median and q10 for OLS (symmetric distribution)
    upper_diff = q90_pred - median_pred
    lower_diff = median_pred - q10_pred

    # Allow some numerical error
    np.testing.assert_allclose(
        upper_diff.mean(),
        lower_diff.mean(),
        rtol=0.1,
        err_msg="OLS should have symmetric quantile predictions around the median",
    )

    transformed_df = pd.DataFrame()
    for quantile, pred_df in predictions.items():
        # For each quantile and its predictions DataFrame
        for variable in imputed_variables:
            # Calculate the mean of predictions for this variable at this quantile
            mean_value = pred_df[variable].mean()
            # Create or update the value in our transformed DataFrame
            if variable not in transformed_df.columns:
                transformed_df[variable] = pd.Series(dtype="float64")
            transformed_df.loc[quantile, variable] = mean_value

    # Save to CSV for further analysis
    transformed_df.to_csv("ols_predictions_by_quantile.csv")
