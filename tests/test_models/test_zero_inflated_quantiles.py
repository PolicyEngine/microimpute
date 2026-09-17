"""Exact mixture inversions and survey weights for issues #203 and #206."""

import numpy as np
import pandas as pd
import pytest

from microimpute.models.zero_inflated import (
    ZeroInflatedImputer,
    ZeroInflatedImputerResults,
)


class _KnownComponent:
    def __init__(self, lower, upper):
        self.lower, self.upper = lower, upper

    def predict(self, X, quantiles=None, **kwargs):
        if quantiles is None:
            return pd.DataFrame({"y": np.repeat(self.lower, len(X))}, index=X.index)
        return {
            q: pd.DataFrame(
                {"y": np.repeat(self.lower + (self.upper - self.lower) * q, len(X))},
                index=X.index,
            )
            for q in quantiles
        }


class _KnownGate:
    def __init__(self, classes, probabilities):
        self.classes_ = np.asarray(classes)
        self.probabilities = np.asarray(probabilities)

    def predict_proba(self, X):
        return self.probabilities[np.asarray(X[:, 0], dtype=int)]


def _mixture(kind, classes, probabilities):
    return ZeroInflatedImputerResults(
        predictors=["x"],
        imputed_variables=["y"],
        seed=42,
        regimes={"y": kind},
        per_variable={
            "y": {
                "kind": kind,
                "classifier": _KnownGate(classes, probabilities),
                "positive_base": _KnownComponent(2, 6),
                "negative_base": _KnownComponent(-4, -2),
            }
        },
    )


@pytest.mark.parametrize(
    "kind,classes,probabilities,expected",
    [
        ("zi_positive", [0, 1], [[0.6, 0.4]], [0, 0, 0, 3, 5]),
        ("zi_negative", [0, 1], [[0.6, 0.4]], [-3.5, -2.5, 0, 0, 0]),
        ("sign_only", [0, 1], [[0.4, 0.6]], [-3.5, -2.5, 2 + 4 / 6, 4, 2 + 10 / 3]),
        ("three_sign", [0, 1, 2], [[0.2, 0.4, 0.4]], [-3, 0, 0, 3, 5]),
    ],
)
def test_quantiles_invert_ordered_mixture(kind, classes, probabilities, expected):
    fitted = _mixture(kind, classes, probabilities)
    query = pd.DataFrame({"x": [0] * 12}, index=np.arange(12) + 100)
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    first = fitted.predict(query, quantiles=quantiles)
    second = fitted.predict(query, quantiles=quantiles)
    for q, value in zip(quantiles, expected):
        np.testing.assert_allclose(first[q]["y"], value)
        pd.testing.assert_frame_equal(first[q], second[q])
    assert (
        np.diff(np.column_stack([first[q]["y"] for q in quantiles]), axis=1) >= 0
    ).all()


def test_mixture_rescaling_varies_by_row_and_handles_zero_mass():
    fitted = _mixture(
        "three_sign", [2, 0, 1], [[0.5, 0.25, 0.25], [0, 1, 0], [0, 0, 1], [1, 0, 0]]
    )
    query = pd.DataFrame({"x": [0, 1, 2, 3]}, index=[9, 7, 3, 1])
    output = fitted.predict(query, quantiles=[0, 0.25, 0.5, 1])
    np.testing.assert_allclose(output[0]["y"], [-4, -4, 0, 2])
    np.testing.assert_allclose(output[0.25]["y"], [-2, -3.5, 0, 3])
    np.testing.assert_allclose(output[0.5]["y"], [0, -3, 0, 4])
    np.testing.assert_allclose(output[1]["y"], [6, -2, 0, 6])


@pytest.mark.parametrize("weights_kind", ["column", "array", "series"])
def test_numeric_weights_reach_gate_and_component(weights_kind):
    # With no predictor signal, the gate must estimate the weighted prevalence;
    # the positive component's median must reflect its own conditional weights.
    values = np.r_[np.zeros(40), np.arange(1, 61, dtype=float)]
    weights = np.r_[np.repeat(30.0, 40), np.repeat(25.0, 10), np.ones(50)]
    data = pd.DataFrame(
        {"x": np.zeros(100), "y": values, "w": weights}, index=np.arange(100) * 2
    )
    weight_arg = {"column": "w", "array": weights, "series": data["w"].iloc[::-1]}[
        weights_kind
    ]
    fitted = ZeroInflatedImputer().fit(data, ["x"], ["y"], weight_col=weight_arg)
    gate = fitted._per_variable["y"]["classifier"]
    np.testing.assert_allclose(gate.predict_proba([[0]])[0], [0.8, 0.2], atol=1e-8)
    # q=.9 is the positive component's median because p0=.8.
    output = fitted.predict(pd.DataFrame({"x": [0.0]}), quantiles=[0.5, 0.9])
    assert output[0.5]["y"].iloc[0] == 0
    assert 1 <= output[0.9]["y"].iloc[0] <= 10


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0, 0.0])
def test_numeric_weights_are_validated(bad):
    data = pd.DataFrame({"x": [0.0] * 30, "y": np.arange(30, dtype=float)})
    weights = np.ones(30)
    weights[3] = bad
    with pytest.raises(ValueError, match="[Ww]eight"):
        ZeroInflatedImputer().fit(data, ["x"], ["y"], weight_col=weights)


def test_sequential_multi_target_quantiles_fail_explicitly():
    data = pd.DataFrame(
        {
            "x": np.arange(60, dtype=float),
            "a": np.arange(60, dtype=float) + 1,
            "b": np.arange(60, dtype=float) + 2,
        }
    )
    fitted = ZeroInflatedImputer().fit(data, ["x"], ["a", "b"], n_estimators=3)
    with pytest.raises(NotImplementedError, match="sequential=False"):
        fitted.predict(data[["x"]], quantiles=[0.5])
    assert fitted.predict(data[["x"]]).shape == (60, 2)


def test_components_with_incompatible_sign_support_fail_explicitly():
    fitted = _mixture("zi_positive", [0, 1], [[0.1, 0.9]])
    fitted._per_variable["y"]["positive_base"] = _KnownComponent(-5, 5)
    with pytest.raises(ValueError, match="outside its sign support"):
        fitted.predict(pd.DataFrame({"x": [0]}), quantiles=[0.2])


def test_explicit_quantiles_do_not_advance_stochastic_rng():
    fitted = _mixture("zi_positive", [0, 1], [[0.6, 0.4]])
    reference = _mixture("zi_positive", [0, 1], [[0.6, 0.4]])
    query = pd.DataFrame({"x": [0] * 50})
    fitted.predict(query, quantiles=[0.1, 0.9])
    pd.testing.assert_frame_equal(fitted.predict(query), reference.predict(query))


@pytest.mark.parametrize("quantiles", [[], [-0.1], [1.1], [np.nan]])
def test_invalid_quantile_requests_fail_explicitly(quantiles):
    fitted = _mixture("zi_positive", [0, 1], [[0.6, 0.4]])
    with pytest.raises(ValueError, match="[Qq]uantile"):
        fitted.predict(pd.DataFrame({"x": [0]}), quantiles=quantiles)


def test_independent_numeric_components_have_distinct_reproducible_draws():
    rng = np.random.default_rng(4)
    data = pd.DataFrame(
        {"x": np.zeros(300), "a": rng.uniform(1, 3, 300), "b": rng.uniform(1, 3, 300)}
    )
    query = pd.DataFrame({"x": np.zeros(500)})
    first = (
        ZeroInflatedImputer(sequential=False, seed=17)
        .fit(data, ["x"], ["a", "b"], n_estimators=10)
        .predict(query)
    )
    second = (
        ZeroInflatedImputer(sequential=False, seed=17)
        .fit(data, ["x"], ["a", "b"], n_estimators=10)
        .predict(query)
    )
    pd.testing.assert_frame_equal(first, second)
    assert abs(first.corr().loc["a", "b"]) < 0.15


def test_explicit_categorical_target_routes_to_auxiliary_model():
    data = pd.DataFrame({"x": np.arange(30, dtype=float), "y": np.tile([0, 1, 2], 10)})
    fitted = ZeroInflatedImputer().fit(
        data, ["x"], ["y"], target_types={"y": "categorical"}, n_estimators=3
    )
    assert "y" not in fitted._regimes
    assert fitted.predict(data[["x"]])["y"].isin([0, 1, 2]).all()


def test_quantile_endpoints_keep_tiny_nonzero_components():
    fitted = _mixture("three_sign", [0, 1, 2], [[1e-20, 1.0, 1e-20]])
    predicted = fitted.predict(pd.DataFrame({"x": [0]}), quantiles=[0, 0.5, 1])
    assert predicted[0]["y"].iloc[0] == -4
    assert predicted[0.5]["y"].iloc[0] == 0
    assert predicted[1]["y"].iloc[0] == 6
