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

# The Matching class can load without R for injected Python callbacks;
# these integration tests specifically require the optional R bridge.
pytest.importorskip("rpy2.robjects")
from microimpute.models.matching import Matching
from microimpute.utils.statmatch_hotdeck import _get_statmatch
from rpy2.robjects.packages import PackageNotInstalledError

try:
    _get_statmatch()
except PackageNotInstalledError:
    pytest.skip("R StatMatch package not available", allow_module_level=True)


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

    # Predict a donor draw for each recipient.
    predictions = fitted_model.predict(X_test)

    # Validate predictions
    assert isinstance(predictions, pd.DataFrame)
    assert predictions.shape == (len(X_test), len(imputed_variables))
    assert not predictions.isna().any().any()


def test_matching_rejects_conditional_quantiles(simple_data: pd.DataFrame) -> None:
    """A donor draw must not be mislabeled as several conditional quantiles."""
    fitted = Matching().fit(simple_data, ["x1", "x2"], ["y"])
    with pytest.raises(NotImplementedError, match="conditional quantiles"):
        fitted.predict(simple_data, quantiles=[0.1, 0.5, 0.9])


def test_matching_donor_preservation(simple_data: pd.DataFrame) -> None:
    """Test that Matching preserves actual donor values."""
    X_train, X_test = preprocess_data(simple_data)

    model = Matching()
    fitted_model = model.fit(X_train, ["x1", "x2"], ["y"])

    predictions = fitted_model.predict(X_test[:1])

    # The predicted value should be from the training set
    predicted_value = predictions["y"].iloc[0]
    assert predicted_value in X_train["y"].values, (
        "Matched value should be from donor pool"
    )


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
        fitted_model = model.fit(X_train, ["x1", "x2"], ["y"], dist_fun=dist_fun)

        predictions = fitted_model.predict(X_test[:5])

        assert not predictions["y"].isna().any()


def test_matching_donor_reuse_limit() -> None:
    """NND k constrains donor reuse; it does not count nearest neighbors."""
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
        fitted_model = model.fit(
            X_train, ["x1", "x2"], ["y"], k=k, constrained=True, constr_alg="lpSolve"
        )

        predictions = fitted_model.predict(X_test[:5])

        assert not predictions["y"].isna().any()


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

    predictions = fitted_model.predict(X_test)

    assert predictions["target_numeric"].dtype == np.float64
    assert pd.api.types.is_string_dtype(predictions["target_category"])


# === Edge Cases ===


def test_matching_single_donor(simple_data: pd.DataFrame) -> None:
    """Test Matching with very small donor pool."""
    # Use only 5 donors
    X_train = simple_data[:5]
    X_test = simple_data[90:]

    model = Matching()
    fitted_model = model.fit(X_train, ["x1", "x2"], ["y"])

    predictions = fitted_model.predict(X_test)

    assert not predictions["y"].isna().any()

    # All predictions should be from the small donor pool
    for val in predictions["y"]:
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

    predictions = fitted_model.predict(X_test)

    # Check that predictions exist
    assert predictions["y"].iloc[0] == 30


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

    predictions = fitted_model.predict(X_test)

    assert not predictions["y"].isna().any()


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

    # Matching does not estimate conditional distributions and cannot be
    # ranked by the quantile/log-loss comparison API.
    for metric in ["quantile_loss", "log_loss"]:
        assert np.isnan(matching_results[metric]["mean_test"])


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
    valid_idx = np.array([i for i in range(len(diabetes_data)) if i not in train_idx])

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
    default_preds = default_fitted.predict(X_valid)
    tuned_preds = tuned_fitted.predict(X_valid)

    # Calculate MSE
    default_mse = {}
    tuned_mse = {}

    for var in imputed_variables:
        default_mse[var] = mean_squared_error(X_valid[var], default_preds[var])
        tuned_mse[var] = mean_squared_error(X_valid[var], tuned_preds[var])

    # Both should produce valid results
    assert all(mse < np.inf for mse in default_mse.values())
    assert all(mse < np.inf for mse in tuned_mse.values())

    # Check hyperparameters if available
    if hasattr(tuned_fitted, "hyperparameters") and tuned_fitted.hyperparameters:
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

    predictions = fitted_model.predict(X_test)

    assert predictions.shape[1] == len(imputed_variables)
    for var in imputed_variables:
        assert var in predictions.columns
        assert not predictions[var].isna().any()


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

    predictions = fitted_model.predict(X_test)

    # Check that the relationship between y1 and y2 is preserved
    # Since we're matching entire rows, y1 and y2 should maintain their relationship
    pred_y1 = predictions["y1"].values
    pred_y2 = predictions["y2"].values

    # Each prediction should come from the same donor row
    for i in range(len(pred_y1)):
        # Find which donor row was matched
        donor_mask = (X_train["y1"] == pred_y1[i]) & (X_train["y2"] == pred_y2[i])
        assert donor_mask.any(), "Predictions should come from same donor row"


def test_matching_weights_change_tied_donor_selection():
    """Optional live-R check of RANDwNND's documented weighted tie selection."""
    donor = pd.DataFrame({"x": [1.0, 1.0], "y": [10.0, 20.0], "w": [1000.0, 1.0]})
    fitted = Matching().fit(donor, ["x"], ["y"], weight_col="w")
    output = fitted.predict(pd.DataFrame({"x": np.ones(500)}))
    assert (output.y == 10.0).mean() > 0.97


def test_matching_seeded_draws_preserve_r_random_stream():
    """Optional live-R reproducibility and global RNG-isolation integration check."""
    import rpy2.robjects as ro

    donor = pd.DataFrame({"x": [1.0, 1.0], "y": [10.0, 20.0], "w": [1.0, 2.0]})
    receiver = pd.DataFrame({"x": np.ones(200)})
    first = Matching(seed=17).fit(donor, ["x"], ["y"], weight_col="w")
    second = Matching(seed=17).fit(donor, ["x"], ["y"], weight_col="w")
    ro.r["set.seed"](31)
    expected_next_draws = np.asarray(ro.r["runif"](3))
    ro.r["set.seed"](31)
    first_draw = first.predict(receiver)
    np.testing.assert_array_equal(np.asarray(ro.r["runif"](3)), expected_next_draws)
    pd.testing.assert_frame_equal(first_draw, second.predict(receiver))
    assert not first_draw.equals(first.predict(receiver))
