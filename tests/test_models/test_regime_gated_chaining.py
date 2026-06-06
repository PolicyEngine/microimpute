"""Sequential (chained-equations) imputation + fitted state in Imputer.

A correct chained imputer conditions each target on the previously imputed
ones, so it reproduces cross-variable correlation that is *not* explained
by the shared predictors. We construct two targets correlated through an
unobserved latent factor and confirm that imputing them together (one
chained list call) recovers that correlation, while imputing them in
separate per-variable calls (the old microplex per-column pattern) does
not. We also check the sklearn-style fitted attributes (``regimes_``,
``predictors_``, ``models_``) and the sub-estimators' standard
``feature_importances_`` / ``feature_names_in_``.
"""

import numpy as np
import pandas as pd

from microimpute.models.regime_gated import REGIME_NO_GATE, Imputer


def _make_latent_correlated_frame(n: int, seed: int) -> pd.DataFrame:
    """Two positive targets A, B that share a latent factor L *with
    opposite sign*, so they are strongly NEGATIVELY correlated.

    The shared predictor X explains almost none of the A-B relationship;
    it runs through L, which is never observed. An imputer that draws B
    independently of A cannot reproduce this dependence. Only one that
    conditions B on the already-imputed A recovers it, because A reveals
    L (given X, A pins down L, which then pins down B).
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    latent = rng.normal(size=n)  # unobserved
    a = 10.0 + 0.2 * x + 1.5 * latent + 0.3 * rng.normal(size=n)
    b = 20.0 + 0.2 * x - 1.5 * latent + 0.3 * rng.normal(size=n)
    return pd.DataFrame({"x": x, "a": a, "b": b})


def _chained_correlation() -> tuple[float, float]:
    """Impute [a, b] together (chained) and return (imputed, true) corr."""
    train = _make_latent_correlated_frame(n=4000, seed=0)
    test = _make_latent_correlated_frame(n=4000, seed=1)
    fitted = Imputer(seed=0).fit(
        X_train=train, predictors=["x"], imputed_variables=["a", "b"]
    )
    preds = fitted.predict(test[["x"]])
    return (
        float(np.corrcoef(preds["a"], preds["b"])[0, 1]),
        float(np.corrcoef(test["a"], test["b"])[0, 1]),
    )


def _independent_correlation() -> float:
    """Impute a and b in *separate* single-variable calls (the old
    microplex per-column pattern), then measure their correlation."""
    train = _make_latent_correlated_frame(n=4000, seed=0)
    test = _make_latent_correlated_frame(n=4000, seed=1)
    a = (
        Imputer(seed=0)
        .fit(X_train=train, predictors=["x"], imputed_variables=["a"])
        .predict(test[["x"]])["a"]
    )
    b = (
        Imputer(seed=0)
        .fit(X_train=train, predictors=["x"], imputed_variables=["b"])
        .predict(test[["x"]])["b"]
    )
    return float(np.corrcoef(a, b)[0, 1])


def test_chaining_recovers_joint_correlation():
    seq_corr, true_corr = _chained_correlation()
    indep_corr = _independent_correlation()

    # The true A-B correlation is strongly negative (opposite latent loads).
    assert true_corr < -0.85, true_corr
    # One chained list call conditions b on the already-imputed a (which
    # reveals the latent factor), recovering the true negative dependence.
    assert seq_corr < -0.7, seq_corr
    assert abs(seq_corr - true_corr) < 0.15, (seq_corr, true_corr)
    # Imputing a and b in separate calls never lets b see a, so it misses
    # most of the dependence.
    assert seq_corr < indep_corr - 0.4, (seq_corr, indep_corr)


def test_fitted_attributes_expose_regimes_models_predictors():
    train = _make_latent_correlated_frame(n=2000, seed=4)
    fitted = Imputer(seed=0).fit(
        X_train=train, predictors=["x"], imputed_variables=["a", "b"]
    )

    # Sklearn-style fitted state, no bespoke lineage object.
    assert set(fitted.regimes_) == {"a", "b"}

    # a is imputed first: only the original predictor; b is chained on a.
    assert fitted.predictors_["a"] == ["x"]
    assert fitted.predictors_["b"] == ["x", "a"]

    # models_ maps roles to fitted sub-estimators.
    roles = fitted.models_["b"]
    assert roles, "expected at least one fitted sub-estimator"
    assert all(v is not None for v in roles.values())

    # At least one QRF base sub-estimator exposes the standard sklearn
    # fitted attributes, keyed by b's chained predictors.
    qrf_roles = [
        roles[r] for r in ("single", "positive", "negative") if r in roles
    ]
    assert qrf_roles, "expected a QRF base sub-estimator by role"
    bases_with_importances = [
        base for base in qrf_roles if hasattr(base, "feature_importances_")
    ]
    assert (
        bases_with_importances
    ), "expected a base estimator exposing feature_importances_"
    base = bases_with_importances[0]
    assert hasattr(base, "feature_importances_")
    assert set(base.feature_names_in_).issubset({"x", "a"}), list(
        base.feature_names_in_
    )


def test_signregime_false_disables_gating():
    train = _make_latent_correlated_frame(n=1500, seed=6)
    fitted = Imputer(signregime=False, seed=0).fit(
        X_train=train, predictors=["x"], imputed_variables=["a", "b"]
    )
    assert fitted.regimes_["a"] == REGIME_NO_GATE
    # Still produces values and still chains predictors.
    preds = fitted.predict(train[["x"]])
    assert set(preds.columns) >= {"a", "b"}
    assert fitted.predictors_["b"] == ["x", "a"]
