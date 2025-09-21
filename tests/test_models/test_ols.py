"""Comprehensive tests for the OLS (Ordinary Least Squares) imputation model."""

from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes

from microimpute.config import QUANTILES
from microimpute.evaluations import *
from microimpute.models.ols import OLS
from microimpute.utils.data import preprocess_data
from microimpute.visualizations import *


# === Fixtures ===


@pytest.fixture
def diabetes_data() -> pd.DataFrame:
    """Load and prepare diabetes dataset for testing."""
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1", "s4"]
    return df[predictors + imputed_variables]


@pytest.fixture
def simple_data() -> pd.DataFrame:
    """Create simple synthetic data for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "x1": np.random.randn(100),
            "x2": np.random.randn(100),
            "y": np.random.randn(100),
        }
    )


# === Basic Functionality Tests ===


def test_ols_basic_fit_predict(diabetes_data: pd.DataFrame) -> None:
    """Test basic OLS model fitting and prediction."""
    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1", "s4"]

    X_train, X_test = preprocess_data(diabetes_data)

    # Initialize and fit model
    model = OLS()
    fitted_model = model.fit(X_train, predictors, imputed_variables)

    # Predict at multiple quantiles
    predictions = fitted_model.predict(
        X_test, QUANTILES, random_quantile_sample=False
    )

    # Validate predictions
    assert isinstance(predictions, dict)
    assert set(predictions.keys()) == set(QUANTILES)

    for q, pred_df in predictions.items():
        assert pred_df.shape == (len(X_test), len(imputed_variables))
        assert not pred_df.isna().any().any()


def test_ols_symmetric_quantiles(simple_data: pd.DataFrame) -> None:
    """Test that OLS produces symmetric quantile predictions due to normal distribution assumption."""
    X_train, X_test = preprocess_data(simple_data)

    model = OLS()
    fitted_model = model.fit(X_train, ["x1", "x2"], ["y"])

    # Predict at symmetric quantiles
    predictions = fitted_model.predict(
        X_test, quantiles=[0.1, 0.5, 0.9], random_quantile_sample=False
    )

    # Check symmetry around median
    median = predictions[0.5]["y"].values
    lower = predictions[0.1]["y"].values
    upper = predictions[0.9]["y"].values

    lower_diff = median - lower
    upper_diff = upper - median

    # OLS assumes normal distribution, so quantiles should be symmetric
    np.testing.assert_allclose(
        lower_diff.mean(),
        upper_diff.mean(),
        rtol=0.1,
        err_msg="OLS should have symmetric quantile predictions around the median",
    )


def test_ols_random_quantile_sampling(simple_data: pd.DataFrame) -> None:
    """Test OLS with random quantile sampling."""
    X_train, X_test = preprocess_data(simple_data)

    model = OLS()
    fitted_model = model.fit(X_train, ["x1", "x2"], ["y"])

    # Predict with random quantile sampling
    predictions = fitted_model.predict(
        X_test, quantiles=[0.5], random_quantile_sample=True
    )

    assert 0.5 in predictions
    assert len(predictions[0.5]) == len(X_test)
    assert not predictions[0.5]["y"].isna().any()


# === Edge Cases ===


def test_ols_perfect_collinearity() -> None:
    """Test OLS behavior with perfectly collinear predictors."""
    np.random.seed(42)
    n_samples = 100

    x1 = np.random.randn(n_samples)

    data = pd.DataFrame(
        {
            "x1": x1,
            "x2": x1 * 2,  # Perfect collinearity
            "x3": x1 * 3,  # Perfect collinearity
            "y": x1 + np.random.randn(n_samples) * 0.1,
        }
    )

    X_train, X_test = preprocess_data(data)

    model = OLS()
    # Should handle collinearity (might drop columns or use regularization)
    fitted_model = model.fit(X_train, ["x1", "x2", "x3"], ["y"])

    predictions = fitted_model.predict(X_test, quantiles=[0.5])

    assert 0.5 in predictions
    assert not predictions[0.5]["y"].isna().any()


# === Cross-Validation Test ===


def test_ols_cross_validation(diabetes_data: pd.DataFrame) -> None:
    """Test OLS model with cross-validation."""
    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1", "s4"]

    ols_results = cross_validate_model(
        OLS, diabetes_data, predictors, imputed_variables
    )

    # Validate cross-validation results - now a dict with dual metrics
    assert isinstance(ols_results, dict)
    assert "quantile_loss" in ols_results
    assert "log_loss" in ols_results

    # Check quantile_loss results (for numerical variables)
    ql_results = ols_results["quantile_loss"]
    assert "results" in ql_results
    assert isinstance(ql_results["results"], pd.DataFrame)
    assert "train" in ql_results["results"].index
    assert "test" in ql_results["results"].index
    assert not ql_results["results"].isna().all().all()
    assert ql_results["mean_test"] > 0


# === Extreme Values ===


def test_ols_extreme_values() -> None:
    """Test OLS with extreme values in data."""
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "x1": np.random.randn(100) * 1000,  # Large scale
            "x2": np.random.randn(100) * 0.001,  # Small scale
            "y": np.random.randn(100),
        }
    )

    X_train, X_test = preprocess_data(data)

    model = OLS()
    fitted_model = model.fit(X_train, ["x1", "x2"], ["y"])

    predictions = fitted_model.predict(X_test, quantiles=[0.5])

    assert 0.5 in predictions
    assert not predictions[0.5]["y"].isna().any()
    assert np.all(np.isfinite(predictions[0.5]["y"].values))


def test_ols_outliers() -> None:
    """Test OLS robustness to outliers."""
    np.random.seed(42)

    data = pd.DataFrame({"x": np.random.randn(100), "y": np.random.randn(100)})

    # Add outliers
    data.loc[0, "y"] = 100  # Large outlier
    data.loc[1, "y"] = -100  # Large outlier

    X_train, X_test = preprocess_data(data)

    model = OLS()
    fitted_model = model.fit(X_train, ["x"], ["y"])

    predictions = fitted_model.predict(X_test, quantiles=[0.5])

    assert 0.5 in predictions
    assert not predictions[0.5]["y"].isna().any()


# === Performance Tests ===


def test_ols_prediction_quality(diabetes_data: pd.DataFrame) -> None:
    """Test OLS prediction quality on real data."""
    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1"]

    data = diabetes_data[predictors + imputed_variables]

    # Split data
    np.random.seed(42)
    train_idx = np.random.choice(
        len(data), int(0.8 * len(data)), replace=False
    )
    test_idx = np.array([i for i in range(len(data)) if i not in train_idx])

    train_data = data.iloc[train_idx].reset_index(drop=True)
    test_data = data.iloc[test_idx].reset_index(drop=True)

    X_train = preprocess_data(
        train_data, full_data=True, train_size=1.0, test_size=0.0
    )
    X_test = preprocess_data(
        test_data, full_data=True, train_size=1.0, test_size=0.0
    )

    # Fit model
    model = OLS()
    fitted_model = model.fit(X_train, predictors, imputed_variables)

    # Get predictions
    predictions = fitted_model.predict(
        X_test, quantiles=[0.5], random_quantile_sample=False
    )

    # Calculate correlation with true values
    true_values = X_test["s1"].values
    pred_values = predictions[0.5]["s1"].values

    correlation = np.corrcoef(true_values, pred_values)[0, 1]

    # OLS should achieve reasonable correlation
    assert correlation > 0.2, f"OLS correlation too low: {correlation}"

    # Check prediction variance is reasonable
    assert np.var(pred_values) > 0, "OLS predictions have no variance"
