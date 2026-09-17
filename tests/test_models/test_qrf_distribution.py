"""Publication regression checks for QRF distribution defaults and weights."""

import logging

import numpy as np
import pandas as pd

from microimpute.config import DEFAULT_MODEL_PARAMS
from microimpute.models.qrf import _QRFModel


def test_qrf_defaults_keep_multiple_observations_per_leaf():
    X = pd.DataFrame({"x": np.arange(100, dtype=float)})
    model = _QRFModel(4, logging.getLogger(__name__))
    model.fit(X, pd.Series(np.sin(X.x), name="y"))
    assert (
        model.qrf.min_samples_leaf
        == DEFAULT_MODEL_PARAMS["qrf"]["min_samples_leaf"]
        >= 10
    )
    assert model.qrf.max_samples_leaf is None
    overridden = _QRFModel(4, logging.getLogger(__name__))
    overridden.fit(
        X, pd.Series(np.sin(X.x), name="y"), min_samples_leaf=3, n_estimators=7
    )
    assert overridden.qrf.min_samples_leaf == 3
    assert overridden.qrf.n_estimators == 7


def test_qrf_weighted_conditional_cdf_without_predictor_signal():
    # No splits are possible, so the weighted median is analytically known.
    X = pd.DataFrame({"x": np.zeros(100)})
    y = pd.Series(np.arange(100, dtype=float), name="y")
    weights = np.r_[np.repeat(100.0, 10), np.ones(90)]
    models = []
    for scale in [1, 50]:
        model = _QRFModel(4, logging.getLogger(__name__))
        model.fit(X, y, sample_weight=weights * scale, bootstrap=False, n_estimators=5)
        models.append(model)
    query = pd.DataFrame({"x": [0.0]})
    for model in models:
        assert model.predict(query, exact_quantile=0.5).iloc[0] == 5
        assert model.predict(query, exact_quantile=0.9).iloc[0] == 9


def test_qrf_default_quantile_coverage_on_normal_noise():
    # Independent holdout from Y|X ~ N(2X,1): default tails must approximate the
    # known .1/.9 probabilities. This checks this fixture, not universal calibration.
    rng = np.random.default_rng(22)
    X = pd.DataFrame({"x": rng.uniform(-2, 2, 3000)})
    y = pd.Series(2 * X.x + rng.normal(size=len(X)), name="y")
    query = pd.DataFrame({"x": rng.uniform(-2, 2, 2000)})
    truth = 2 * query.x + rng.normal(size=len(query))
    model = _QRFModel(42, logging.getLogger(__name__))
    model.fit(X, y)
    for q in [0.1, 0.5, 0.9]:
        observed = np.mean(truth <= model.predict(query, exact_quantile=q))
        assert abs(observed - q) < 0.055, (q, observed)


def test_qrf_weighted_cdf_preserves_bootstrap_multiplicity():
    X = pd.DataFrame({"x": np.zeros(50)})
    y = pd.Series(np.arange(50, dtype=float), name="y")
    weights = np.linspace(1, 10, 50)
    model = _QRFModel(11, logging.getLogger(__name__))
    model.fit(X, y, sample_weight=weights, n_estimators=4, min_samples_leaf=5)
    # One leaf per tree makes the bootstrap-weighted CDF directly calculable.
    tree_mass = []
    for samples in model.qrf.estimators_samples_:
        mass = np.bincount(samples, minlength=len(y)) * weights
        tree_mass.append(mass / mass.sum())
    cdf = np.cumsum(np.mean(tree_mass, axis=0))
    quantiles = np.array([0.1, 0.5, 0.9])
    expected = np.searchsorted(cdf, quantiles)
    observed = model.predict_quantiles_per_row(
        pd.DataFrame({"x": np.zeros(3)}), quantiles
    )
    np.testing.assert_array_equal(observed, expected)


def test_qrf_row_specific_quantiles_match_scalar_queries():
    rng = np.random.default_rng(19)
    X = pd.DataFrame({"x": rng.normal(size=100)})
    y = pd.Series(rng.normal(size=100), name="y")
    model = _QRFModel(4, logging.getLogger(__name__))
    model.fit(X, y, n_estimators=5)
    query = X.iloc[:5]
    quantiles = np.array([0, 0.21, 0.53, 0.97, 1])
    output = model.predict_quantiles_per_row(query, quantiles)
    expected = [
        model.predict(query.iloc[[i]], exact_quantile=float(q)).iloc[0]
        for i, q in enumerate(quantiles)
    ]
    np.testing.assert_allclose(output, expected)


def test_sequential_multi_target_quantiles_are_explicitly_unsupported():
    from microimpute.models import QRF
    import pytest

    rng = np.random.default_rng(36)
    data = pd.DataFrame(
        {"x": rng.normal(size=80), "a": rng.normal(size=80), "b": rng.normal(size=80)}
    )
    fitted = QRF().fit(data, ["x"], ["a", "b"], n_estimators=5)
    with pytest.raises(NotImplementedError, match="sequential=False"):
        fitted.predict(data[["x"]].iloc[:3], quantiles=[0.5])
    assert fitted.predict(data[["x"]].iloc[:3]).shape == (3, 2)


def test_independent_quantiles_are_invariant_to_target_order_and_batching():
    from microimpute.models import QRF

    rng = np.random.default_rng(6)
    data = pd.DataFrame(
        {
            "x": rng.normal(size=200),
            "a": rng.normal(size=200),
            "b": rng.normal(size=200),
        }
    )
    query = data[["x"]].iloc[:15]
    reference = None
    for batch_size in [None, 1]:
        for variables in [["a", "b"], ["b", "a"]]:
            fitted = QRF(sequential=False, seed=9, batch_size=batch_size).fit(
                data, ["x"], variables, n_estimators=7
            )
            for model in fitted.models.values():
                assert model.feature_columns == ["x"]
            prediction = fitted.predict(query, quantiles=[0.1, 0.5, 0.9])
            if reference is None:
                reference = prediction
            for q in prediction:
                pd.testing.assert_frame_equal(
                    reference[q].sort_index(axis=1), prediction[q].sort_index(axis=1)
                )


def test_independent_quantile_is_marginal_for_nonlinearly_related_targets():
    from microimpute.models import QRF

    # For A~Uniform(-2,2), median(A)=0 but median(A**2)=1. Plugging
    # median(A) into a conditional second-stage model incorrectly gives zero.
    rng = np.random.default_rng(95)
    a = rng.uniform(-2, 2, 1200)
    data = pd.DataFrame({"x": np.zeros(len(a)), "a": a, "b": a**2})
    fitted = QRF(sequential=False).fit(data, ["x"], ["a", "b"], n_estimators=15)
    medians = fitted.predict(pd.DataFrame({"x": [0.0]}), quantiles=[0.5])[0.5]
    assert abs(medians.a.iloc[0]) < 0.15
    assert abs(medians.b.iloc[0] - 1.0) < 0.2


def test_sequential_stochastic_draws_retain_joint_dependence():
    from microimpute.models import QRF

    rng = np.random.default_rng(11)
    a = rng.normal(size=1500)
    data = pd.DataFrame(
        {
            "x": rng.normal(size=len(a)),
            "a": a,
            "b": -a + rng.normal(scale=0.1, size=len(a)),
        }
    )
    fitted = QRF().fit(data, ["x"], ["a", "b"], n_estimators=20)
    draws = fitted.predict(data[["x"]].iloc[:600])
    assert draws.corr().loc["a", "b"] < -0.9


def test_independent_mode_is_preserved_after_tuning(monkeypatch):
    from microimpute.models import QRF

    rng = np.random.default_rng(6)
    data = pd.DataFrame(
        {"x": rng.normal(size=80), "a": rng.normal(size=80), "b": rng.normal(size=80)}
    )
    model = QRF(sequential=False)
    monkeypatch.setattr(
        model, "_tune_hyperparameters", lambda **kwargs: {"n_estimators": 3}
    )
    fitted, _ = model.fit(data, ["x"], ["a", "b"], tune_hyperparameters=True)
    assert not fitted.sequential
    assert fitted.predict(data[["x"]].iloc[:2], quantiles=[0.5])[0.5].shape == (2, 2)


def test_independent_inner_tuning_uses_original_predictors(monkeypatch):
    from microimpute.models import QRF
    from microimpute.models.qrf import _RandomForestClassifierModel

    rng = np.random.default_rng(19)
    data = pd.DataFrame(
        {
            "x": rng.normal(size=80),
            "a": rng.normal(size=80),
            "b": rng.normal(size=80),
            "c": np.tile(["A", "B"], 40),
        }
    )
    seen = []
    for model_class in [_QRFModel, _RandomForestClassifierModel]:
        original_fit = model_class.fit

        def recording_fit(self, X, y, *args, _fit=original_fit, **kwargs):
            seen.append((y.name, list(X.columns)))
            return _fit(self, X, y, *args, **kwargs)

        monkeypatch.setattr(model_class, "fit", recording_fit)
    model = QRF(sequential=False)
    model.imputed_variables = ["a", "b", "c"]
    model.categorical_targets = {"c": {"type": "categorical", "categories": ["A", "B"]}}
    model._tune_qrf_hyperparameters(data, ["x"], ["a", "b"], n_cv_folds=2, n_trials=1)
    model._tune_rfc_hyperparameters(data, ["x"], ["c"], n_cv_folds=2, n_trials=1)
    assert {name for name, _ in seen} == {"a", "b", "c"}
    assert all(columns == ["x"] for _, columns in seen)


def test_independent_batch_fit_supports_mixed_targets():
    from microimpute.models import QRF

    rng = np.random.default_rng(7)
    data = pd.DataFrame(
        {
            "x": rng.normal(size=60),
            "a": rng.normal(size=60),
            "label": np.tile(["A", "B"], 30),
        }
    )
    fitted = QRF(sequential=False, batch_size=1).fit(
        data, ["x"], ["a", "label"], n_estimators=3
    )
    result = fitted.predict(data[["x"]].iloc[:3], quantiles=[0.5], return_probs=True)
    assert result[0.5].shape == (3, 2)
    assert result["probabilities"]["label"]["probabilities"].shape == (3, 2)
