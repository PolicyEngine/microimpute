"""Tests for the Quantile Regression imputation model."""

from typing import Dict, List

import pandas as pd
from sklearn.datasets import load_diabetes

from microimpute.comparisons.data import preprocess_data
from microimpute.config import QUANTILES
from microimpute.evaluations import *
from microimpute.models.quantreg import QuantReg

# Test Method on diabetes dataset
diabetes_data = load_diabetes()
diabetes_df = pd.DataFrame(
    diabetes_data.data, columns=diabetes_data.feature_names
)

predictors = ["age", "sex", "bmi", "bp"]
imputed_variables = ["s1"]

diabetes_df = diabetes_df[predictors + imputed_variables]


def test_matching_cross_validation(
    data: pd.DataFrame = diabetes_df,
    predictors: List[str] = predictors,
    imputed_variables: List[str] = imputed_variables,
    quantiles: List[float] = QUANTILES,
) -> None:
    """
    Test the QuantReg model on a specific dataset.

    Args:
            data: DataFrame with the dataset of interest.
            predictors: List of predictor variables.
            imputed_variables: List of variables to impute.
            quantiles: List of quantiles to predict.
    """
    quantreg_results = cross_validate_model(
        QuantReg, data, predictors, imputed_variables
    )

    quantreg_results.to_csv("quantreg_results.csv")

    assert not quantreg_results.isna().any().any()

    plot_train_test_performance(
        quantreg_results, save_path="quantreg_train_test_performance.png"
    )


def test_quantreg_example(
    data: pd.DataFrame = diabetes_df,
    predictors: List[str] = predictors,
    imputed_variables: List[str] = imputed_variables,
    quantiles: List[float] = QUANTILES,
) -> None:
    """
    Example of how to use the Quantile Regression imputer model.

    This example demonstrates:
    - Initializing a QuantReg model
    - Fitting the model to specific quantiles
    - Predicting quantiles on test data
    - How QuantReg can capture non-symmetric distributions

    Args:
        data: DataFrame with test data
        predictors: List of predictor column names
        imputed_variables: List of target column names
        quantiles: List of quantiles to predict
    """
    X_train, X_test = preprocess_data(data)

    # Initialize QuantReg model
    model = QuantReg()

    # Fit the model to specific quantiles
    fitted_model = model.fit(
        X_train, predictors, imputed_variables, quantiles=quantiles
    )

    # Predict at the fitted quantiles
    predictions: Dict[float, pd.DataFrame] = fitted_model.predict(X_test)

    # Check structure of predictions
    assert isinstance(predictions, dict)
    assert set(predictions.keys()) == set(quantiles)

    # The advantage of QuantReg is that it can capture asymmetric distributions
    # No assertion here since asymmetry depends on the data

    # Basic checks
    for q, pred in predictions.items():
        assert pred is not None
        assert len(pred) == len(X_test)
