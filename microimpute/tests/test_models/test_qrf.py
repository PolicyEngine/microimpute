"""Tests for the Quantile Regression Forest imputation model."""

from typing import Dict, List

import pandas as pd
from sklearn.datasets import load_diabetes

from microimpute.comparisons.data import preprocess_data
from microimpute.config import QUANTILES
from microimpute.evaluations import *
from microimpute.models.qrf import QRF

# Test Method on diabetes dataset
diabetes_data = load_diabetes()
diabetes_df = pd.DataFrame(
    diabetes_data.data, columns=diabetes_data.feature_names
)

predictors = ["age", "sex", "bmi", "bp"]
imputed_variables = ["s1", "s4"]

diabetes_df = diabetes_df[predictors + imputed_variables]


def test_qrf_cross_validation(
    data: pd.DataFrame = diabetes_df,
    predictors: List[str] = predictors,
    imputed_variables: List[str] = imputed_variables,
    quantiles: List[float] = QUANTILES,
) -> None:
    """
    Test the QRF model on a specific dataset.

    Args:
            data: DataFrame with the dataset of interest.
            predictors: List of predictor variables.
            imputed_variables: List of variables to impute.
            quantiles: List of quantiles to predict.
    """
    qrf_results = cross_validate_model(
        QRF, data, predictors, imputed_variables
    )

    qrf_results.to_csv("qrf_cv_results.csv")

    assert not qrf_results.isna().any().any()

    plot_train_test_performance(
        qrf_results, save_path="qrf_train_test_performance.jpg"
    )


def test_qrf_example(
    data: pd.DataFrame = diabetes_df,
    predictors: List[str] = predictors,
    imputed_variables: List[str] = imputed_variables,
    quantiles: List[float] = QUANTILES,
) -> None:
    """
    Example of how to use the Quantile Random Forest imputer model.

    This example demonstrates:
    - Initializing a QRF model
    - Fitting the model with optional hyperparameters
    - Predicting quantiles on test data
    - How QRF can capture complex nonlinear relationships

    Args:
        data: DataFrame with the dataset to use.
        predictors: List of predictor column names.
        imputed_variables: List of target column names.
        quantiles: List of quantiles to predict.
    """
    X_train, X_test, dummy_info = preprocess_data(data)

    # Initialize QRF model
    model = QRF()

    # Fit the model with RF hyperparameters
    fitted_model = model.fit(
        X_train,
        predictors,
        imputed_variables,
        n_estimators=100,  # Number of trees
        min_samples_leaf=5,  # Min samples in leaf nodes
    )

    # Predict at multiple quantiles
    predictions: Dict[float, pd.DataFrame] = fitted_model.predict(
        X_test, quantiles
    )

    # Check structure of predictions
    assert isinstance(predictions, dict)
    assert set(predictions.keys()) == set(quantiles)

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
    transformed_df.to_csv("qrf_predictions_by_quantile.csv")
