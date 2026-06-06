"""Sequential (chained-equations) imputation in ZeroInflatedImputer.

A correct chained imputer conditions each target on the previously
imputed ones, so it reproduces cross-variable correlation that is *not*
explained by the shared predictors. We construct two targets that are
correlated through an unobserved latent factor and confirm that the
sequential imputer recovers that correlation while the non-sequential
(per-variable independent) imputer does not.
"""

import numpy as np
import pandas as pd

from microimpute.models.zero_inflated import ZeroInflatedImputer


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


def _imputed_correlation(sequential: bool) -> tuple[float, float]:
    train = _make_latent_correlated_frame(n=4000, seed=0)
    test = _make_latent_correlated_frame(n=4000, seed=1)

    imp = ZeroInflatedImputer(sequential=sequential, seed=0)
    fitted = imp.fit(
        X_train=train,
        predictors=["x"],
        imputed_variables=["a", "b"],
    )
    preds = fitted.predict(test[["x"]])
    imputed_corr = float(np.corrcoef(preds["a"], preds["b"])[0, 1])
    true_corr = float(np.corrcoef(test["a"], test["b"])[0, 1])
    return imputed_corr, true_corr


def test_sequential_chaining_recovers_joint_correlation():
    seq_corr, true_corr = _imputed_correlation(sequential=True)
    indep_corr, _ = _imputed_correlation(sequential=False)

    # The true A-B correlation is strongly negative (opposite latent loads).
    assert true_corr < -0.85, true_corr
    # Chained imputation conditions b on the already-imputed a (which
    # reveals the latent factor), recovering the true negative dependence
    # almost exactly.
    assert seq_corr < -0.7, seq_corr
    assert abs(seq_corr - true_corr) < 0.15, (seq_corr, true_corr)
    # Non-sequential per-variable imputation never sees a when drawing b,
    # so it misses most of the dependence.
    assert seq_corr < indep_corr - 0.4, (seq_corr, indep_corr)


def test_single_target_is_unaffected_by_sequential_flag():
    """A one-variable list has no prior to chain on, so the flag is a no-op."""
    train = _make_latent_correlated_frame(n=1500, seed=2)
    test = _make_latent_correlated_frame(n=1500, seed=3)

    out = {}
    for sequential in (True, False):
        imp = ZeroInflatedImputer(sequential=sequential, seed=7)
        fitted = imp.fit(X_train=train, predictors=["x"], imputed_variables=["a"])
        out[sequential] = fitted.predict(test[["x"]])["a"].to_numpy()

    np.testing.assert_allclose(out[True], out[False])
