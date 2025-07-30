"""Tests for the Quantile Regression imputation model."""

from typing import Dict, List

import pandas as pd
from sklearn.datasets import load_diabetes

from microimpute.comparisons.data import preprocess_data
from microimpute.config import QUANTILES, RANDOM_STATE
from microimpute.evaluations import *
from microimpute.models.quantreg import QuantReg
from microimpute.visualizations.plotting import *

# Test Method on diabetes dataset
diabetes_data = load_diabetes()
diabetes_df = pd.DataFrame(
    diabetes_data.data, columns=diabetes_data.feature_names
)

predictors = ["age", "sex", "bmi", "bp"]
imputed_variables = ["s1", "s4"]

diabetes_df = diabetes_df[predictors + imputed_variables]

random_generator = np.random.default_rng(RANDOM_STATE)
count_samples = 10
mean_quantile = 0.5
# Calculate alpha parameter for beta distribution
a = mean_quantile / (1 - mean_quantile)
# Generate count_samples beta distributed values with parameter a
beta_samples = random_generator.beta(a, 1, size=count_samples)
quantiles = list(set(beta_samples))


def test_quantreg_cross_validation(
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

    quantreg_results.to_csv("quantreg_cv_results.csv")

    assert not quantreg_results.isna().any().any()

    perf_results_viz = model_performance_results(
        results=quantreg_results,
        model_name="QuantReg",
        method_name="Cross-Validation Quantile Loss Average",
    )
    fig = perf_results_viz.plot(
        title="QuantReg Cross-Validation Performance",
        save_path="quantreg_cv_performance.jpg",
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
    predictions: Dict[float, pd.DataFrame] = fitted_model.predict(
        X_test, random_quantile_sample=False
    )

    # Check structure of predictions
    assert isinstance(predictions, dict)

    # Basic checks
    for q, pred in predictions.items():
        assert pred is not None
        assert len(pred) == len(X_test)

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
    transformed_df.to_csv("quantreg_predictions_by_quantile.csv")
