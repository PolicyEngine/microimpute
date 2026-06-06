"""Sequential (chained-equations) imputation + lineage in Imputer.

A correct chained imputer conditions each target on the previously imputed
ones, so it reproduces cross-variable correlation that is *not* explained
by the shared predictors. We construct two targets correlated through an
unobserved latent factor and confirm that imputing them together (one
chained list call) recovers that correlation, while imputing them in
separate per-variable calls (the old microplex per-column pattern) does
not. We also check the lineage accessor.
"""

import numpy as np
import pandas as pd

from microimpute.models.regime_gated import (REGIME_NO_GATE, Imputer,
                                             VariableLineage)


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


def test_lineage_reports_chained_predictors_and_models():
    train = _make_latent_correlated_frame(n=2000, seed=4)
    fitted = Imputer(seed=0).fit(
        X_train=train, predictors=["x"], imputed_variables=["a", "b"]
    )
    lineage = fitted.lineage()

    assert set(lineage) == {"a", "b"}
    assert all(isinstance(v, VariableLineage) for v in lineage.values())

    # a is imputed first: only the original predictor.
    assert lineage["a"].predictors == ["x"]
    # b is chained on the already-imputed a.
    assert lineage["b"].predictors == ["x", "a"]

    # Fit metrics and at least one fitted model with importances are present.
    assert lineage["b"].fit_metrics["n_train"] == 2000
    assert lineage["b"].models, "expected at least one fitted model"
    imps = [v for v in lineage["b"].feature_importances.values() if v]
    assert imps, "expected QRF feature importances for at least one role"
    # b's base importances are keyed by its chained predictors.
    some = imps[0]
    assert set(some).issubset({"x", "a"}), some


def test_signregime_false_disables_gating():
    train = _make_latent_correlated_frame(n=1500, seed=6)
    fitted = Imputer(signregime=False, seed=0).fit(
        X_train=train, predictors=["x"], imputed_variables=["a", "b"]
    )
    lineage = fitted.lineage()
    assert lineage["a"].regime == REGIME_NO_GATE
    # Still produces values and still chains predictors.
    preds = fitted.predict(train[["x"]])
    assert set(preds.columns) >= {"a", "b"}
    assert lineage["b"].predictors == ["x", "a"]
