"""Comprehensive tests for the Quantile Regression imputation model."""

from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes

from microimpute.config import QUANTILES, RANDOM_STATE
from microimpute.evaluations import *
from microimpute.models.quantreg import QuantReg
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


@pytest.fixture
def skewed_data() -> pd.DataFrame:
    """Create data with skewed distribution."""
    np.random.seed(42)
    n_samples = 100

    # Create skewed target using exponential distribution
    return pd.DataFrame(
        {
            "x1": np.random.randn(n_samples),
            "x2": np.random.randn(n_samples),
            "y": np.random.exponential(scale=2.0, size=n_samples),
        }
    )


# === Basic Functionality Tests ===


def test_quantreg_basic_fit_predict(diabetes_data: pd.DataFrame) -> None:
    """Test basic QuantReg model fitting and prediction."""
    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1", "s4"]

    X_train, X_test = preprocess_data(diabetes_data)

    # Initialize and fit model
    model = QuantReg()
    fitted_model = model.fit(
        X_train, predictors, imputed_variables, quantiles=QUANTILES
    )

    # Predict at fitted quantiles
    predictions = fitted_model.predict(X_test, random_quantile_sample=False)

    # Validate predictions
    assert isinstance(predictions, dict)

    for q, pred_df in predictions.items():
        assert pred_df is not None
        assert len(pred_df) == len(X_test)
        assert not pred_df.isna().any().any()


def test_quantreg_specific_quantiles(simple_data: pd.DataFrame) -> None:
    """Test QuantReg with specific quantiles."""
    X_train, X_test = preprocess_data(simple_data)

    specific_quantiles = [0.1, 0.5, 0.9]

    model = QuantReg()
    fitted_model = model.fit(X_train, ["x1", "x2"], ["y"], quantiles=specific_quantiles)

    # Predict at the same quantiles
    predictions = fitted_model.predict(X_test, quantiles=specific_quantiles)

    assert set(predictions.keys()) == set(specific_quantiles)

    for q in specific_quantiles:
        assert not predictions[q]["y"].isna().any()


def test_quantreg_monotonic_quantiles(simple_data: pd.DataFrame) -> None:
    """Test that QuantReg produces monotonic quantile predictions."""
    X_train, X_test = preprocess_data(simple_data)

    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    model = QuantReg()
    fitted_model = model.fit(X_train, ["x1", "x2"], ["y"], quantiles=quantiles)

    predictions = fitted_model.predict(X_test, quantiles=quantiles)

    # Check monotonicity for most observations (allowing some crossing due to estimation)
    monotonic_count = 0
    for i in range(len(X_test)):
        values = [predictions[q]["y"].iloc[i] for q in quantiles]
        is_monotonic = all(
            values[j] <= values[j + 1] + 1e-6 for j in range(len(values) - 1)
        )
        if is_monotonic:
            monotonic_count += 1

    # At least 80% should be monotonic (some crossing is expected in quantile regression)
    assert monotonic_count / len(X_test) > 0.8, "Too many quantile crossings"


# === Edge Cases ===


def test_quantreg_skewed_distribution(skewed_data: pd.DataFrame) -> None:
    """Test QuantReg with skewed target distribution."""
    X_train, X_test = preprocess_data(skewed_data)

    quantiles = [0.25, 0.5, 0.75]

    model = QuantReg()
    fitted_model = model.fit(X_train, ["x1", "x2"], ["y"], quantiles=quantiles)

    predictions = fitted_model.predict(X_test, quantiles=quantiles)

    # For skewed distribution, quantiles should not be symmetric
    median = predictions[0.5]["y"].mean()
    q25 = predictions[0.25]["y"].mean()
    q75 = predictions[0.75]["y"].mean()

    lower_diff = median - q25
    upper_diff = q75 - median

    # For skewed distributions, quantiles should be different from symmetric
    # Just check they're not exactly equal (some asymmetry is captured)
    assert abs(upper_diff - lower_diff) > 0.01, "QuantReg should capture some asymmetry"


# === Cross-Validation Test ===


def test_quantreg_cross_validation(diabetes_data: pd.DataFrame) -> None:
    """Test QuantReg model with cross-validation."""
    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1", "s4"]

    quantreg_results = cross_validate_model(
        QuantReg, diabetes_data, predictors, imputed_variables
    )

    # Validate cross-validation results - now a dict with dual metrics
    assert isinstance(quantreg_results, dict)
    assert "quantile_loss" in quantreg_results
    assert "log_loss" in quantreg_results

    # Check quantile_loss results (for numerical variables)
    ql_results = quantreg_results["quantile_loss"]
    assert "results" in ql_results
    assert isinstance(ql_results["results"], pd.DataFrame)
    assert "train" in ql_results["results"].index
    assert "test" in ql_results["results"].index
    assert not ql_results["results"].isna().all().all()
    assert ql_results["mean_test"] > 0


# === Robustness Tests ===


def test_quantreg_outliers() -> None:
    """Test QuantReg robustness to outliers (should be more robust than OLS)."""
    np.random.seed(42)

    data = pd.DataFrame({"x": np.random.randn(100), "y": np.random.randn(100)})

    # Add extreme outliers
    data.loc[0, "y"] = 1000
    data.loc[1, "y"] = -1000

    X_train, X_test = preprocess_data(data)

    model = QuantReg()
    fitted_model = model.fit(
        X_train,
        ["x"],
        ["y"],
        quantiles=[0.5],  # Median regression is robust to outliers
    )

    predictions = fitted_model.predict(X_test, quantiles=[0.5])

    # When quantiles specified, returns dict
    assert isinstance(predictions, dict)
    assert 0.5 in predictions
    assert not predictions[0.5]["y"].isna().any()

    # Median predictions should not be heavily influenced by outliers
    median_pred = predictions[0.5]["y"].median()
    assert -10 < median_pred < 10, "Median regression affected by outliers"


def test_quantreg_heteroscedasticity() -> None:
    """Test QuantReg with heteroscedastic errors."""
    np.random.seed(42)
    n_samples = 200

    x = np.random.uniform(-3, 3, n_samples)
    # Variance increases with x (heteroscedasticity)
    y = 2 * x + np.random.randn(n_samples) * (1 + np.abs(x))

    data = pd.DataFrame({"x": x, "y": y})

    X_train, X_test = preprocess_data(data)

    model = QuantReg()
    fitted_model = model.fit(X_train, ["x"], ["y"], quantiles=[0.1, 0.5, 0.9])

    predictions = fitted_model.predict(X_test, quantiles=[0.1, 0.5, 0.9])

    # Quantile regression should capture heteroscedasticity
    # Prediction intervals should be wider for larger |x|
    X_test_sorted = X_test.sort_values("x")
    low_x_idx = X_test_sorted.index[:10]
    high_x_idx = X_test_sorted.index[-10:]

    low_x_spread = (
        predictions[0.9].loc[low_x_idx, "y"] - predictions[0.1].loc[low_x_idx, "y"]
    ).mean()
    high_x_spread = (
        predictions[0.9].loc[high_x_idx, "y"] - predictions[0.1].loc[high_x_idx, "y"]
    ).mean()

    # Check that there's some difference in spread (heteroscedasticity is partially captured)
    assert abs(high_x_spread - low_x_spread) > 0.01, (
        "QuantReg should show some heteroscedasticity effect"
    )


# === Comparison with OLS ===


def test_quantreg_vs_ols_median(simple_data: pd.DataFrame) -> None:
    """Test that QuantReg median predictions are similar to OLS mean predictions for normal data."""
    from microimpute.models.ols import OLS

    X_train, X_test = preprocess_data(simple_data)

    # Fit QuantReg for median
    quantreg = QuantReg()
    quantreg_fitted = quantreg.fit(X_train, ["x1", "x2"], ["y"], quantiles=[0.5])
    quantreg_pred = quantreg_fitted.predict(X_test, quantiles=[0.5])

    # Fit OLS
    ols = OLS()
    ols_fitted = ols.fit(X_train, ["x1", "x2"], ["y"])
    ols_pred = ols_fitted.predict(X_test, quantiles=[0.5], random_quantile_sample=False)

    # For normally distributed errors, median and mean should be similar
    # QuantReg returns DataFrame for single quantile, OLS returns dict
    quantreg_median = quantreg_pred[0.5]["y"].values
    ols_mean = ols_pred[0.5]["y"].values

    correlation = np.corrcoef(quantreg_median, ols_mean)[0, 1]
    assert correlation > 0.85, (
        "QuantReg median and OLS mean should be reasonably similar for normal data"
    )


# === Performance Tests ===


def test_quantreg_prediction_quality(diabetes_data: pd.DataFrame) -> None:
    """Test QuantReg prediction quality on real data."""
    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1"]

    data = diabetes_data[predictors + imputed_variables]

    # Split data
    np.random.seed(42)
    train_idx = np.random.choice(len(data), int(0.8 * len(data)), replace=False)
    test_idx = np.array([i for i in range(len(data)) if i not in train_idx])

    train_data = data.iloc[train_idx].reset_index(drop=True)
    test_data = data.iloc[test_idx].reset_index(drop=True)

    X_train = preprocess_data(train_data, full_data=True, train_size=1.0, test_size=0.0)
    X_test = preprocess_data(test_data, full_data=True, train_size=1.0, test_size=0.0)

    # Fit model
    model = QuantReg()
    fitted_model = model.fit(X_train, predictors, imputed_variables, quantiles=[0.5])

    # Get predictions
    predictions = fitted_model.predict(X_test, quantiles=[0.5])

    # Calculate correlation with true values
    # When quantiles specified, returns dict
    true_values = X_test["s1"].values
    pred_values = predictions[0.5]["s1"].values

    correlation = np.corrcoef(true_values, pred_values)[0, 1]

    # QuantReg should achieve reasonable correlation
    assert correlation > 0.15, f"QuantReg correlation too low: {correlation}"

    # Check that predictions have reasonable variance
    assert np.var(pred_values) > 0, "QuantReg predictions have no variance"


def test_quantreg_random_quantile_sample_returns_numeric_dtype() -> None:
    """Regression test for #11: the per-row random-quantile path previously
    allocated an object-dtype DataFrame and wrote numeric predictions
    into it via .loc, silently demoting the result to object. After
    vectorisation the output must be a numeric dtype."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 100
    data = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": rng.normal(size=n) + rng.normal(size=n),
        }
    )

    model = QuantReg()
    fitted = model.fit(data, ["x"], ["y"], quantiles=[0.1, 0.5, 0.9])

    test = pd.DataFrame({"x": rng.normal(size=20)})
    preds = fitted.predict(test, random_quantile_sample=True)

    # With random_quantile_sample=True and no quantiles at predict time,
    # the implementation keys the result by the mean of the fitted
    # quantiles.
    mean_q = np.mean([0.1, 0.5, 0.9])
    assert mean_q in preds
    out = preds[mean_q]
    assert pd.api.types.is_numeric_dtype(out["y"]), (
        f"Vectorised random-quantile path must return a numeric dtype; "
        f"got {out['y'].dtype}"
    )
    assert np.all(np.isfinite(out["y"].values))
