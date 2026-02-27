"""Comprehensive tests for the Statistical Matching imputation model."""

from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error

from microimpute.config import QUANTILES
from microimpute.evaluations import *
from microimpute.utils.data import preprocess_data
from microimpute.visualizations import *

try:
    from microimpute.models.matching import Matching

    MATCHING_AVAILABLE = True
except ImportError:
    MATCHING_AVAILABLE = False
    pytest.skip("Matching model not available", allow_module_level=True)


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


@pytest.fixture
def categorical_data() -> pd.DataFrame:
    """Create data with categorical variables."""
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame(
        {
            "numeric": np.random.randn(n_samples),
            "category": np.random.choice(["A", "B", "C"], n_samples),
            "target": np.random.randn(n_samples),
        }
    )


# === Basic Functionality Tests ===


def test_matching_basic_fit_predict(diabetes_data: pd.DataFrame) -> None:
    """Test basic Matching model fitting and prediction."""
    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1", "s4"]

    X_train, X_test = preprocess_data(diabetes_data)

    # Initialize and fit model
    model = Matching()
    fitted_model = model.fit(X_train, predictors, imputed_variables)

    # Predict (matching uses same value for all quantiles)
    predictions = fitted_model.predict(X_test, quantiles=[0.5])

    # Validate predictions
    assert isinstance(predictions, dict)
    assert 0.5 in predictions
    assert isinstance(predictions[0.5], pd.DataFrame)
    assert predictions[0.5].shape == (len(X_test), len(imputed_variables))
    assert not predictions[0.5].isna().any().any()


def test_matching_quantile_invariance(simple_data: pd.DataFrame) -> None:
    """Test that Matching returns same values for different quantiles."""
    X_train, X_test = preprocess_data(simple_data)

    model = Matching()
    fitted_model = model.fit(X_train, ["x1", "x2"], ["y"])

    # Get predictions at different quantiles
    predictions = fitted_model.predict(X_test, quantiles=[0.1, 0.5, 0.9])

    # Matching should return same values for all quantiles
    # (it doesn't model uncertainty)
    for i in range(len(X_test)):
        val_01 = predictions[0.1]["y"].iloc[i]
        val_05 = predictions[0.5]["y"].iloc[i]
        val_09 = predictions[0.9]["y"].iloc[i]
        assert (
            val_01 == val_05 == val_09
        ), "Matching should return same value for all quantiles"


def test_matching_donor_preservation(simple_data: pd.DataFrame) -> None:
    """Test that Matching preserves actual donor values."""
    X_train, X_test = preprocess_data(simple_data)

    model = Matching()
    fitted_model = model.fit(X_train, ["x1", "x2"], ["y"])

    predictions = fitted_model.predict(X_test[:1], quantiles=[0.5])

    # The predicted value should be from the training set
    predicted_value = predictions[0.5]["y"].iloc[0]
    assert (
        predicted_value in X_train["y"].values
    ), "Matched value should be from donor pool"


# === Distance Functions Tests ===


def test_matching_different_distance_functions() -> None:
    """Test Matching with different distance functions."""
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "x1": np.random.randn(100),
            "x2": np.random.randn(100),
            "y": np.random.randn(100),
        }
    )

    X_train, X_test = preprocess_data(data)

    distance_functions = ["Manhattan", "Euclidean"]

    for dist_fun in distance_functions:
        model = Matching()
        fitted_model = model.fit(
            X_train, ["x1", "x2"], ["y"], dist_fun=dist_fun
        )

        predictions = fitted_model.predict(X_test[:5], quantiles=[0.5])

        assert 0.5 in predictions
        assert not predictions[0.5]["y"].isna().any()


def test_matching_k_neighbors() -> None:
    """Test Matching with different k values."""
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "x1": np.random.randn(100),
            "x2": np.random.randn(100),
            "y": np.random.randn(100),
        }
    )

    X_train, X_test = preprocess_data(data)

    # Test different k values
    for k in [1, 3, 5]:
        model = Matching()
        fitted_model = model.fit(X_train, ["x1", "x2"], ["y"], k=k)

        predictions = fitted_model.predict(X_test[:5], quantiles=[0.5])

        assert 0.5 in predictions
        assert not predictions[0.5]["y"].isna().any()


# === Categorical Variables ===


def test_matching_mixed_types() -> None:
    """Test Matching with mixed data types."""
    np.random.seed(42)
    n_samples = 100

    data = pd.DataFrame(
        {
            "numeric": np.random.randn(n_samples),
            "category": np.random.choice(["A", "B", "C"], n_samples),
            "binary": np.random.choice([0, 1], n_samples),
            "target_numeric": np.random.randn(n_samples),
            "target_category": np.random.choice(["X", "Y"], n_samples),
        }
    )

    X_train, X_test = preprocess_data(data, normalize=False)

    model = Matching()
    fitted_model = model.fit(
        X_train,
        ["numeric", "category", "binary"],
        ["target_numeric", "target_category"],
    )

    predictions = fitted_model.predict(X_test, quantiles=[0.5])

    assert predictions[0.5]["target_numeric"].dtype == np.float64
    assert pd.api.types.is_string_dtype(predictions[0.5]["target_category"])


# === Edge Cases ===


def test_matching_single_donor(simple_data: pd.DataFrame) -> None:
    """Test Matching with very small donor pool."""
    # Use only 5 donors
    X_train = simple_data[:5]
    X_test = simple_data[90:]

    model = Matching()
    fitted_model = model.fit(X_train, ["x1", "x2"], ["y"])

    predictions = fitted_model.predict(X_test, quantiles=[0.5])

    assert 0.5 in predictions
    assert not predictions[0.5]["y"].isna().any()

    # All predictions should be from the small donor pool
    for val in predictions[0.5]["y"]:
        assert val in X_train["y"].values


def test_matching_exact_match() -> None:
    """Test Matching when exact matches exist."""
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [10, 20, 30, 40, 50],
        }
    )

    X_train = data
    # Test with exact match
    X_test = pd.DataFrame({"x1": [3.0], "x2": [3.0]})

    model = Matching()
    fitted_model = model.fit(X_train, ["x1", "x2"], ["y"])

    predictions = fitted_model.predict(X_test, quantiles=[0.5])

    # Check that predictions exist
    assert 0.5 in predictions
    assert not predictions[0.5].empty


# === Constrained Matching ===


def test_matching_constrained_mode() -> None:
    """Test Matching with constrained mode."""
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "x1": np.random.randn(100),
            "x2": np.random.randn(100),
            "y": np.random.randn(100),
        }
    )

    X_train, X_test = preprocess_data(data)

    model = Matching()
    fitted_model = model.fit(X_train, ["x1", "x2"], ["y"], constrained=True)

    predictions = fitted_model.predict(X_test, quantiles=[0.5])

    assert 0.5 in predictions
    assert not predictions[0.5]["y"].isna().any()


# === Cross-Validation ===


def test_matching_cross_validation(diabetes_data: pd.DataFrame) -> None:
    """Test Matching model with cross-validation."""
    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1", "s4"]

    # Preprocess without normalization for matching
    data = preprocess_data(diabetes_data, full_data=True, normalize=False)

    matching_results = cross_validate_model(
        Matching, data, predictors, imputed_variables
    )

    # Validate cross-validation results - now a dict with dual metrics
    assert isinstance(matching_results, dict)
    assert "quantile_loss" in matching_results
    assert "log_loss" in matching_results

    # Check quantile_loss results (for numerical variables)
    ql_results = matching_results["quantile_loss"]
    assert "results" in ql_results
    assert isinstance(ql_results["results"], pd.DataFrame)
    assert "train" in ql_results["results"].index
    assert "test" in ql_results["results"].index
    assert not ql_results["results"].isna().all().all()
    assert ql_results["mean_test"] > 0


# === Hyperparameter Tuning ===


def test_matching_hyperparameter_tuning(diabetes_data: pd.DataFrame) -> None:
    """Test hyperparameter tuning for Matching model."""
    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1", "s4"]

    # Split data
    np.random.seed(42)
    train_idx = np.random.choice(
        len(diabetes_data), int(0.7 * len(diabetes_data)), replace=False
    )
    valid_idx = np.array(
        [i for i in range(len(diabetes_data)) if i not in train_idx]
    )

    train_data = diabetes_data.iloc[train_idx].reset_index(drop=True)
    valid_data = diabetes_data.iloc[valid_idx].reset_index(drop=True)

    X_train = preprocess_data(train_data, full_data=True)
    X_valid = preprocess_data(valid_data, full_data=True)

    # Fit models with and without tuning
    default_model = Matching()
    default_fitted = default_model.fit(X_train, predictors, imputed_variables)

    tuned_model = Matching()
    tuned_fitted, best_params = tuned_model.fit(
        X_train, predictors, imputed_variables, tune_hyperparameters=True
    )

    # Make predictions
    default_preds = default_fitted.predict(X_valid, quantiles=[0.5])
    tuned_preds = tuned_fitted.predict(X_valid, quantiles=[0.5])

    # Calculate MSE
    default_mse = {}
    tuned_mse = {}

    for var in imputed_variables:
        default_mse[var] = mean_squared_error(
            X_valid[var], default_preds[0.5][var]
        )
        tuned_mse[var] = mean_squared_error(
            X_valid[var], tuned_preds[0.5][var]
        )

    # Both should produce valid results
    assert all(mse < np.inf for mse in default_mse.values())
    assert all(mse < np.inf for mse in tuned_mse.values())

    # Check hyperparameters if available
    if (
        hasattr(tuned_fitted, "hyperparameters")
        and tuned_fitted.hyperparameters
    ):
        if "dist_fun" in tuned_fitted.hyperparameters:
            assert tuned_fitted.hyperparameters["dist_fun"] in [
                "Manhattan",
                "Euclidean",
                "Mahalanobis",
                "exact",
                "Gower",
                "minimax",
            ]
        if "k" in tuned_fitted.hyperparameters:
            assert 1 <= tuned_fitted.hyperparameters["k"] <= 10


# === Performance Tests ===


def test_matching_multiple_targets(diabetes_data: pd.DataFrame) -> None:
    """Test Matching with multiple target variables."""
    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1", "s2", "s3", "s4"]

    diabetes = load_diabetes()
    full_data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    data = full_data[predictors + imputed_variables]

    X_train, X_test = preprocess_data(data)

    model = Matching()
    fitted_model = model.fit(X_train, predictors, imputed_variables)

    predictions = fitted_model.predict(X_test, quantiles=[0.5])

    assert predictions[0.5].shape[1] == len(imputed_variables)
    for var in imputed_variables:
        assert var in predictions[0.5].columns
        assert not predictions[0.5][var].isna().any()


def test_matching_preserves_relationships() -> None:
    """Test that Matching preserves relationships between variables."""
    np.random.seed(42)
    n_samples = 100

    # Create data with strong relationship between targets
    x = np.random.randn(n_samples)
    data = pd.DataFrame(
        {
            "x": x,
            "y1": 2 * x + np.random.randn(n_samples) * 0.1,
            "y2": 3 * x + np.random.randn(n_samples) * 0.1,
        }
    )

    X_train = data[:80]
    X_test = data[80:][["x"]]  # Only predictors for test

    model = Matching()
    fitted_model = model.fit(X_train, ["x"], ["y1", "y2"])

    predictions = fitted_model.predict(X_test, quantiles=[0.5])

    # Check that the relationship between y1 and y2 is preserved
    # Since we're matching entire rows, y1 and y2 should maintain their relationship
    pred_y1 = predictions[0.5]["y1"].values
    pred_y2 = predictions[0.5]["y2"].values

    # Each prediction should come from the same donor row
    for i in range(len(pred_y1)):
        # Find which donor row was matched
        donor_mask = (X_train["y1"] == pred_y1[i]) & (
            X_train["y2"] == pred_y2[i]
        )
        assert donor_mask.any(), "Predictions should come from same donor row"
