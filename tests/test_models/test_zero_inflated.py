"""Regime-aware zero-inflation wrapper around base imputers.

Tabular microdata variables fall into seven regimes based on which of
{negative, zero, positive} values appear in the training data:

- (+, 0, −): three-sign. Gate classifies neg/0/pos; separate base imputers
  on the positive and negative subsets.
- (+, 0):    standard zero-inflated positive. Binary gate (0 vs nonzero);
  base imputer on positives only.
- (−, 0):    mirror ZI. Binary gate; base imputer on negatives only.
- (+, −):    sign-only (no zero). Binary sign gate; two base imputers.
- (+):       positive-only. Single base imputer; no gate.
- (−):       negative-only. Single base imputer; no gate.
- (0):       degenerate. Constant-zero imputer; no gate.

The wrapper detects the regime automatically at fit time from the
training distribution (no per-variable hand configuration) and composes
the base imputer accordingly. This replaces ad-hoc per-variable zero
handling in downstream packages (policyengine-us-data, microplex-us).

Tests here pin the public contract. They deliberately use tiny fixtures
so they run in under a second each — the correctness axes are:

1. Regime detection from training data is correct.
2. Predictions respect the detected regime (no zero leaks, no
   sign-interpolation between positive and negative regimes).
3. Fit/predict lifecycle matches the base `Imputer` contract.
4. Pure support detection: any observed negative, zero, or positive
   support participates in regime selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from microimpute.models.qrf import QRF


def _deterministic_frame(
    n: int,
    y_values: np.ndarray,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a training DataFrame with two predictors and one target.

    Predictors are (age, income_bin) so both are in-range for random
    integers / floats; the target is whatever distribution the caller
    constructs via ``y_values``.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, size=n).astype(float),
            "income_bin": rng.integers(0, 4, size=n).astype(float),
            "y": y_values,
        }
    )


class TestRegimeDetection:
    """Regime auto-detection from the training distribution."""

    def test_regime_detection_is_importable(self) -> None:
        from microimpute.models.zero_inflated import ZeroInflatedImputer

        assert ZeroInflatedImputer is not None

    def test_positive_plus_zero_is_zi_positive(self) -> None:
        """97% zeros + 3% positives → ZI_POSITIVE regime."""
        from microimpute.models.zero_inflated import ZeroInflatedImputer

        rng = np.random.default_rng(0)
        y = np.where(rng.random(500) > 0.97, rng.exponential(100, 500), 0.0)
        data = _deterministic_frame(500, y)

        imputer = ZeroInflatedImputer(base_imputer_class=QRF)
        imputer.fit(data, predictors=["age", "income_bin"], imputed_variables=["y"])
        assert imputer.get_regime("y") == "ZI_POSITIVE"

    def test_negative_plus_zero_is_zi_negative(self) -> None:
        """97% zeros + 3% negatives → ZI_NEGATIVE regime (mirror)."""
        from microimpute.models.zero_inflated import ZeroInflatedImputer

        rng = np.random.default_rng(0)
        y = np.where(rng.random(500) > 0.97, -rng.exponential(100, 500), 0.0)
        data = _deterministic_frame(500, y)

        imputer = ZeroInflatedImputer(base_imputer_class=QRF)
        imputer.fit(data, predictors=["age", "income_bin"], imputed_variables=["y"])
        assert imputer.get_regime("y") == "ZI_NEGATIVE"

    def test_three_sign_mass_is_three_sign(self) -> None:
        """Non-trivial mass in each of {neg, 0, pos} → THREE_SIGN regime.

        Fixture models a capital-gains-like distribution: 70% zero,
        15% positive, 15% negative, with distinct pos/neg means.
        """
        from microimpute.models.zero_inflated import ZeroInflatedImputer

        rng = np.random.default_rng(0)
        n = 1000
        u = rng.random(n)
        y = np.zeros(n)
        pos_mask = u > 0.85
        neg_mask = (u > 0.70) & (u <= 0.85)
        y[pos_mask] = rng.exponential(500, size=pos_mask.sum())
        y[neg_mask] = -rng.exponential(300, size=neg_mask.sum())
        data = _deterministic_frame(n, y)

        imputer = ZeroInflatedImputer(base_imputer_class=QRF)
        imputer.fit(data, predictors=["age", "income_bin"], imputed_variables=["y"])
        assert imputer.get_regime("y") == "THREE_SIGN"

    def test_positive_only_is_positive_only(self) -> None:
        """All positive, no zeros → POSITIVE_ONLY (no gate, raw base imputer)."""
        from microimpute.models.zero_inflated import ZeroInflatedImputer

        rng = np.random.default_rng(0)
        y = rng.exponential(100, size=500)  # strictly positive
        data = _deterministic_frame(500, y)

        imputer = ZeroInflatedImputer(base_imputer_class=QRF)
        imputer.fit(data, predictors=["age", "income_bin"], imputed_variables=["y"])
        assert imputer.get_regime("y") == "POSITIVE_ONLY"

    def test_constant_zero_is_degenerate(self) -> None:
        """All zeros → DEGENERATE_ZERO, predictions are exactly 0."""
        from microimpute.models.zero_inflated import ZeroInflatedImputer

        data = _deterministic_frame(500, np.zeros(500))
        imputer = ZeroInflatedImputer(base_imputer_class=QRF)
        imputer.fit(data, predictors=["age", "income_bin"], imputed_variables=["y"])
        assert imputer.get_regime("y") == "DEGENERATE_ZERO"

    def test_rare_negative_tail_triggers_three_sign(self) -> None:
        """Any observed negative support participates in auto detection.

        Capital gains example: 97% zero, 2.9% positive, 0.1% negative.
        The negative mass is real and should trigger THREE_SIGN without
        caller-supplied support/sign metadata.
        """
        from microimpute.models.zero_inflated import ZeroInflatedImputer

        rng = np.random.default_rng(0)
        n = 500
        u = rng.random(n)
        y = np.zeros(n)
        pos_mask = u > 0.971
        y[pos_mask] = rng.exponential(100, size=pos_mask.sum())
        y[0] = -50.0
        assert (y < 0).sum() == 1, "fixture precondition"
        assert (y > 0).sum() > 0, "fixture precondition"
        assert (y == 0).sum() > 0, "fixture precondition"

        data = _deterministic_frame(n, y)
        imputer = ZeroInflatedImputer(base_imputer_class=QRF)
        imputer.fit(data, predictors=["age", "income_bin"], imputed_variables=["y"])
        assert imputer.get_regime("y") == "THREE_SIGN"


class TestPredictionsRespectRegime:
    """Predicted values must lie in the regime's support."""

    def _fit_predict(self, y: np.ndarray, n_pred: int = 200) -> pd.Series:
        from microimpute.models.zero_inflated import ZeroInflatedImputer

        data = _deterministic_frame(len(y), y, seed=1)
        imputer = ZeroInflatedImputer(base_imputer_class=QRF)
        result = imputer.fit(
            data, predictors=["age", "income_bin"], imputed_variables=["y"]
        )

        rng = np.random.default_rng(42)
        predict_data = pd.DataFrame(
            {
                "age": rng.integers(18, 80, size=n_pred).astype(float),
                "income_bin": rng.integers(0, 4, size=n_pred).astype(float),
            }
        )
        predictions = result.predict(predict_data)
        return predictions["y"]

    def test_zi_positive_predictions_are_zero_or_positive(self) -> None:
        rng = np.random.default_rng(0)
        y = np.where(rng.random(800) > 0.95, rng.exponential(200, 800), 0.0)
        preds = self._fit_predict(y, n_pred=500)
        assert (preds >= 0).all(), (
            f"ZI_POSITIVE predictions include {(preds < 0).sum()} negative "
            f"values; min={preds.min()}."
        )

    def test_zi_negative_predictions_are_zero_or_negative(self) -> None:
        rng = np.random.default_rng(0)
        y = np.where(rng.random(800) > 0.95, -rng.exponential(200, 800), 0.0)
        preds = self._fit_predict(y, n_pred=500)
        assert (preds <= 0).all(), (
            f"ZI_NEGATIVE predictions include {(preds > 0).sum()} positive "
            f"values; max={preds.max()}."
        )

    def test_positive_only_predictions_are_positive(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.exponential(100, size=800)  # strictly positive
        preds = self._fit_predict(y, n_pred=500)
        # Allow small fp nonneg; predictions must not be meaningfully negative.
        assert (preds >= -1e-6).all(), preds.min()

    def test_constant_zero_predictions_are_all_zero(self) -> None:
        preds = self._fit_predict(np.zeros(500), n_pred=300)
        np.testing.assert_array_equal(preds, 0.0)

    def test_three_sign_predictions_do_not_leak_across_zero(self) -> None:
        """A record whose gate says POS must draw from the positive base
        imputer only — never from the negative one. Easiest observable
        consequence: we can recover predictions that are strictly above
        `min(train_positives)` on records the gate calls positive, and
        strictly below `max(train_negatives)` on records the gate calls
        negative. No record should land in the interpolated band between
        max(neg) and min(pos)."""
        from microimpute.models.zero_inflated import ZeroInflatedImputer

        rng = np.random.default_rng(0)
        n = 2000
        u = rng.random(n)
        y = np.zeros(n)
        # Clear gap between regimes: positives in [100, ∞), negatives
        # in (-∞, -100]. Zeros exactly 0.
        pos_mask = u > 0.85
        neg_mask = (u > 0.70) & (u <= 0.85)
        y[pos_mask] = 100 + rng.exponential(500, size=pos_mask.sum())
        y[neg_mask] = -(100 + rng.exponential(300, size=neg_mask.sum()))
        data = _deterministic_frame(n, y, seed=2)

        imputer = ZeroInflatedImputer(base_imputer_class=QRF)
        result = imputer.fit(
            data, predictors=["age", "income_bin"], imputed_variables=["y"]
        )

        predict_rng = np.random.default_rng(42)
        predict_data = pd.DataFrame(
            {
                "age": predict_rng.integers(18, 80, size=2000).astype(float),
                "income_bin": predict_rng.integers(0, 4, size=2000).astype(float),
            }
        )
        preds = result.predict(predict_data)["y"]

        # The interpolation-free guarantee: no prediction lies strictly
        # between -100 and 100 except exactly zero. (If the gate routed a
        # record to the positive base, the QRF draws from training
        # positives which are all >= 100; similarly for negative.)
        interior = preds[(preds > -100) & (preds < 100)]
        nonzero_interior = interior[np.abs(interior) > 1e-6]
        assert len(nonzero_interior) == 0, (
            f"Three-sign regime leaked {len(nonzero_interior)} draws into "
            f"the (-100, 100) interior: samples={nonzero_interior.head(10).tolist()}"
        )


class TestBaseImputerParity:
    """Single-regime behavior must match the base Imputer (no gate overhead)."""

    def test_positive_only_matches_bare_qrf(self) -> None:
        """For a strictly-positive target, ZeroInflatedImputer should
        produce equivalent distributions to QRF directly (since regime
        detection returns POSITIVE_ONLY and no gate is applied)."""
        from microimpute.models.zero_inflated import ZeroInflatedImputer

        rng = np.random.default_rng(0)
        y = rng.exponential(100, size=500)  # strictly positive
        data = _deterministic_frame(500, y, seed=3)

        bare = QRF()
        bare_result = bare.fit(
            data, predictors=["age", "income_bin"], imputed_variables=["y"]
        )

        wrapped = ZeroInflatedImputer(base_imputer_class=QRF)
        wrapped_result = wrapped.fit(
            data, predictors=["age", "income_bin"], imputed_variables=["y"]
        )

        predict_rng = np.random.default_rng(99)
        predict_data = pd.DataFrame(
            {
                "age": predict_rng.integers(18, 80, size=300).astype(float),
                "income_bin": predict_rng.integers(0, 4, size=300).astype(float),
            }
        )

        bare_preds = bare_result.predict(predict_data)["y"]
        wrapped_preds = wrapped_result.predict(predict_data)["y"]

        # Both should produce strictly positive draws with similar
        # distributional statistics on this small fixture.
        assert (bare_preds >= 0).all()
        assert (wrapped_preds >= 0).all()
        # The wrapper adds a little randomness via its own rng, so we
        # don't require element-wise equality — just broadly similar
        # means.
        assert (
            abs(bare_preds.mean() - wrapped_preds.mean()) / max(bare_preds.mean(), 1.0)
            < 0.25
        )
