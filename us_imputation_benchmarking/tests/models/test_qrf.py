from typing import Dict, List, Tuple

import pandas as pd
from sklearn.datasets import load_iris

from us_imputation_benchmarking.comparisons.data import preprocess_data
from us_imputation_benchmarking.config import QUANTILES
from us_imputation_benchmarking.evaluations.cross_validation import \
    cross_validate_model
from us_imputation_benchmarking.evaluations.train_test_performance import \
    plot_train_test_performance
from us_imputation_benchmarking.models.qrf import QRF

# Test Method on iris dataset
iris_data = load_iris()
iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)

predictors = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)"]
imputed_variables = ["petal width (cm)"]

iris_df = iris_df[predictors + imputed_variables]


def test_qrf_cross_validation(
    data: pd.DataFrame = iris_df,
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
    qrf_results = cross_validate_model(QRF, data, predictors, imputed_variables)

    # qrf_results.to_csv("matching_results.csv")

    assert not qrf_results.isna().any().any()

    # plot_train_test_performance(qrf_results, save_path="qrf_train_test_performance.png")


def test_qrf_example(
    data: pd.DataFrame = iris_df,
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
    X_train, X_test = preprocess_data(data)

    # Initialize QRF model
    model = QRF()

    # Fit the model with RF hyperparameters
    model.fit(
        X_train,
        predictors,
        imputed_variables,
        n_estimators=100,  # Number of trees
        min_samples_leaf=5,  # Min samples in leaf nodes
    )

    # Predict at multiple quantiles
    predictions: Dict[float, pd.DataFrame] = model.predict(X_test, quantiles)

    # Check structure of predictions
    assert isinstance(predictions, dict)
    assert set(predictions.keys()) == set(quantiles)

    # QRF should capture nonlinear relationships
    # We'd need more complex tests to verify this in detail
