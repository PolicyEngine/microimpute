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


def test_ols_quantile_uses_full_prediction_se() -> None:
    """Regression test for #6: OLS quantile prediction must use the full
    prediction SE (leverage + residual) rather than sqrt(scale). The
    prediction SE for a new row is strictly greater than the residual
    std; the gap grows for rows far from the training centroid.

    We verify the fix by checking that (a) at a fixed quantile, the
    prediction interval at an extreme x is WIDER than at the centroid
    (which the old implementation could not produce — both used the
    same residual std), and (b) at q=0.99 the quantile prediction for
    an extreme row is larger than the residual-std-only formulation
    would have given.
    """
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    y = 2.0 * x + rng.normal(size=n) * 0.3
    train = pd.DataFrame({"x": x, "y": y})

    model = OLS()
    fitted = model.fit(train, ["x"], ["y"])

    # One test row at the centroid, one far outside the support.
    x_test = pd.DataFrame({"x": [0.0, 10.0]})
    upper = fitted.predict(x_test, quantiles=[0.99])[0.99]["y"].values
    lower = fitted.predict(x_test, quantiles=[0.01])[0.01]["y"].values
    widths = upper - lower

    # The extrapolated point must have a wider prediction interval than
    # the centroid (leverage effect). With the old se = sqrt(scale),
    # both rows had identical widths.
    assert widths[1] > widths[0], (
        "Prediction SE must grow with leverage; widths were "
        f"{widths}, indicating residual-std-only (pre-fix) behaviour"
    )


def test_ols_quantile_clips_q_away_from_zero_and_one() -> None:
    """Regression test for #6: q=0 and q=1 previously produced ±inf via
    norm.ppf; the clipped implementation should return finite values."""
    rng = np.random.default_rng(0)
    n = 100
    x = rng.normal(size=n)
    y = 2.0 * x + rng.normal(size=n) * 0.3
    train = pd.DataFrame({"x": x, "y": y})

    model = OLS()
    fitted = model.fit(train, ["x"], ["y"])

    x_test = pd.DataFrame({"x": [0.0]})
    preds_0 = fitted.predict(x_test, quantiles=[0.0])[0.0]["y"].values
    preds_1 = fitted.predict(x_test, quantiles=[1.0])[1.0]["y"].values

    assert np.all(np.isfinite(preds_0)), "q=0 produced non-finite predictions"
    assert np.all(np.isfinite(preds_1)), "q=1 produced non-finite predictions"


def test_ols_mixed_targets_preserve_test_index() -> None:
    """Regression test: when OLS imputes a DataFrame containing both a
    numeric target (returned via the OLS path) and a categorical/boolean
    target (returned via the logistic-regression path), the predictions
    assembled into the output DataFrame must align by the X_test index.

    The bug: the numeric path previously returned a bare ndarray. When
    that ndarray was assigned as the first column of a fresh empty
    DataFrame, the DataFrame took on a default RangeIndex (0..N). The
    subsequent categorical prediction (a pd.Series with the real
    X_test index, e.g. 160..199) then failed to align on assignment
    and the whole column came back as NaN — which later produced
    ``Input contains NaN`` in sklearn's log_loss.
    """
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "num_pred1": rng.normal(size=n),
            "num_pred2": rng.normal(size=n) * 2 + 1,
            "num_target": rng.normal(size=n) * 3,
            "binary_target": rng.choice([0, 1], size=n),
        }
    )
    # Split so the test slice has a non-zero-based index.
    train_data = df.iloc[:160].copy()
    test_data = df.iloc[160:].copy()

    model = OLS()
    fitted = model.fit(
        train_data,
        predictors=["num_pred1", "num_pred2"],
        imputed_variables=["num_target", "binary_target"],
    )
    predictions = fitted.predict(test_data, quantiles=[0.5])
    out = predictions[0.5]

    assert (
        not out["num_target"].isna().any()
    ), "num_target should not contain NaN"
    assert not out["binary_target"].isna().any(), (
        "binary_target should not be NaN; the output DataFrame index must "
        "align with the X_test index so the categorical Series assignment "
        "lines up with the numeric column"
    )
    # Index must match the test slice, not a fresh RangeIndex.
    assert list(out.index) == list(test_data.index)


def test_logistic_l1_ratio_activates_elasticnet() -> None:
    """Regression test for #8: passing l1_ratio must activate the
    elasticnet penalty (and saga solver). Previously l1_ratio was
    passed through with the default L2 penalty and was silently ignored.
    """
    import logging

    from microimpute.models.ols import _LogisticRegressionModel

    rng = np.random.default_rng(0)
    n = 100
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = pd.Series((X["a"] + rng.normal(size=n) > 0).astype(int), name="y")

    model = _LogisticRegressionModel(seed=0, logger=logging.getLogger("test"))
    model.fit(X, y, var_type="boolean", l1_ratio=0.5)

    # When l1_ratio=0.5, penalty must be "elasticnet" and solver "saga"
    # so l1_ratio actually has an effect.
    assert model.classifier.penalty == "elasticnet", (
        f"Expected penalty='elasticnet' with l1_ratio=0.5, got "
        f"penalty={model.classifier.penalty!r} (l1_ratio silently ignored)"
    )
    assert (
        model.classifier.solver == "saga"
    ), f"Expected solver='saga' for elasticnet, got {model.classifier.solver!r}"
