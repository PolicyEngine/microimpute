"""Regression API and stochastic correctness checks for submission issue #206/#208."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from microimpute.models.ols import OLS
from microimpute.models.quantreg import QuantReg


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    x = rng.normal(size=200)
    return pd.DataFrame({"x": x, "y": 3 + 2 * x + rng.normal(size=200)})


@pytest.mark.parametrize("model_class", [OLS, QuantReg])
@pytest.mark.parametrize("x", [[2.0], [2.0, 2.0, 2.0]])
def test_homogeneous_receivers_keep_intercept(model_class, x, regression_data):
    fitted = model_class().fit(regression_data, ["x"], ["y"])
    receiver = pd.DataFrame({"x": x}, index=np.arange(len(x)) + 1000)
    actual = fitted.predict(receiver, quantiles=[0.5])[0.5]
    estimator = sm.OLS if model_class is OLS else sm.QuantReg
    reference = estimator(
        regression_data.y, sm.add_constant(regression_data[["x"]], has_constant="add")
    ).fit()
    expected = reference.predict(sm.add_constant(receiver, has_constant="add"))
    np.testing.assert_allclose(actual.y, expected)
    assert actual.index.equals(receiver.index)


@pytest.mark.parametrize("model_class", [OLS, QuantReg])
def test_constant_training_predictor_keeps_stable_design(model_class):
    data = pd.DataFrame({"x": np.ones(30), "y": np.arange(30.0)})
    fitted = model_class().fit(data, ["x"], ["y"])
    actual = fitted.predict(pd.DataFrame({"x": [1.0]}), quantiles=[0.5])[0.5]
    np.testing.assert_allclose(actual.y, [14.5], atol=1e-5)


@pytest.mark.parametrize("model_class", [OLS, QuantReg])
@pytest.mark.parametrize("column", ["x", "y"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_training_values_fail(
    model_class, column, bad_value, regression_data
):
    regression_data.loc[0, column] = bad_value
    with pytest.raises((ValueError, RuntimeError), match="finite|missing|NaN|inf"):
        model_class().fit(regression_data, ["x"], ["y"])


@pytest.mark.parametrize("model_class", [OLS, QuantReg])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_prediction_values_fail(model_class, bad_value, regression_data):
    fitted = model_class().fit(regression_data, ["x"], ["y"])
    with pytest.raises((ValueError, RuntimeError), match="finite|missing|NaN|inf"):
        fitted.predict(pd.DataFrame({"x": [bad_value, 1.0]}))


def test_quantreg_fits_new_prediction_quantiles_on_donor_data(regression_data):
    fitted = QuantReg().fit(regression_data, ["x"], ["y"])
    receiver = pd.DataFrame({"x": [-1.0, 0.0, 1.0]})
    initial_default = fitted.predict(receiver)
    grid = [0.13, 0.5, 0.87]
    actual = fitted.predict(receiver, quantiles=grid)
    for q in grid:
        reference = sm.QuantReg(
            regression_data.y,
            sm.add_constant(regression_data[["x"]], has_constant="add"),
        ).fit(q=q)
        expected = reference.predict(sm.add_constant(receiver, has_constant="add"))
        np.testing.assert_allclose(actual[q].y, expected)
    pd.testing.assert_frame_equal(fitted.predict(receiver), initial_default)


@pytest.mark.parametrize("q", [0.0, 1.0])
def test_quantreg_rejects_unsupported_endpoints(q, regression_data):
    fitted = QuantReg().fit(regression_data, ["x"], ["y"])
    with pytest.raises(ValueError, match="strictly between"):
        fitted.predict(pd.DataFrame({"x": [0.0]}), quantiles=[q])


def test_ols_sampling_has_one_independent_shock_per_row_and_advances(regression_data):
    # Identical target fits expose accidental re-use of the same residual shock.
    regression_data["z"] = regression_data.y
    fitted = OLS().fit(regression_data, ["x"], ["y", "z"])
    receiver = pd.DataFrame({"x": np.zeros(2000)})
    first = fitted.predict(receiver, random_quantile_sample=True)
    second = fitted.predict(receiver, random_quantile_sample=True)
    assert first.y.nunique() == len(receiver)
    assert not np.array_equal(first.y, second.y)
    assert abs(first.y.corr(first.z)) < 0.1
    fresh = OLS().fit(regression_data, ["x"], ["y", "z"])
    pd.testing.assert_frame_equal(
        first, fresh.predict(receiver, random_quantile_sample=True)
    )


def test_quantreg_sampling_advances_independently_across_targets(regression_data):
    regression_data["z"] = regression_data.y
    fitted = QuantReg().fit(
        regression_data, ["x"], ["y", "z"], quantiles=[0.1, 0.5, 0.9]
    )
    receiver = pd.DataFrame({"x": np.zeros(1000)})
    first = fitted.predict(receiver, random_quantile_sample=True)[0.5]
    second = fitted.predict(receiver, random_quantile_sample=True)[0.5]
    assert not np.array_equal(first.y, second.y)
    assert abs(first.y.corr(first.z)) < 0.1


def test_ols_survey_weight_units_do_not_change_predictive_quantiles(regression_data):
    # Survey weights express relative population mass, so changing their units
    # must leave both the conditional mean and residual distribution invariant.
    weights = np.linspace(0.5, 4.0, len(regression_data))
    receiver = pd.DataFrame({"x": [-1.0, 0.0, 4.0]})
    grid = [0.1, 0.5, 0.9]
    first = OLS().fit(regression_data, ["x"], ["y"], weight_col=weights)
    scaled = OLS().fit(regression_data, ["x"], ["y"], weight_col=100 * weights)
    first_predictions = first.predict(receiver, quantiles=grid)
    scaled_predictions = scaled.predict(receiver, quantiles=grid)
    for q in grid:
        np.testing.assert_allclose(first_predictions[q], scaled_predictions[q])

    from scipy.stats import norm

    reference = sm.WLS(
        regression_data.y,
        sm.add_constant(regression_data[["x"]], has_constant="add"),
        weights=weights / weights.mean(),
    ).fit()
    prediction = reference.get_prediction(sm.add_constant(receiver, has_constant="add"))
    expected = prediction.predicted_mean + norm.ppf(0.9) * np.sqrt(
        prediction.var_pred_mean + reference.scale
    )
    np.testing.assert_allclose(first_predictions[0.9].y, expected)


def test_ols_explicit_quantiles_are_deterministic_even_if_sampling_flag_set(
    regression_data,
):
    fitted = OLS().fit(regression_data, ["x"], ["y"])
    receiver = pd.DataFrame({"x": [0.0, 1.0]})
    exact = fitted.predict(receiver, quantiles=[0.1, 0.9])
    actual = fitted.predict(receiver, quantiles=[0.1, 0.9], random_quantile_sample=True)
    for q in exact:
        pd.testing.assert_frame_equal(actual[q], exact[q])
