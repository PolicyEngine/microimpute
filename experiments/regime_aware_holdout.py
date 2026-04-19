"""Holdout experiment: tripartite regime-aware vs alternatives on neg/0/pos data.

Settles the design question: given a target variable that genuinely
spans {negative, zero, positive}, is the three-sign regime-aware
imputer (binary gate on signed class, separate base imputers per sign)
actually better than simpler alternatives?

Four approaches tested on the same held-out real-DGP-like fixture:

A. **Tripartite**: ``ZeroInflatedImputer`` with regime detection
   (expected regime: ``THREE_SIGN``). Gate routes to positive-QRF or
   negative-QRF; exact zeros come from the gate.

B. **Binary-nonzero + single QRF**: simulates a ``y != 0`` gate.
   Binary classifier (zero vs nonzero); single QRF trained on all
   nonzero rows, pos and neg mixed. The QRF interpolates between the
   two regimes — the failure mode we hypothesized.

C. **Positive-only + QRF (current microplex-us bug)**: ``y > 0`` gate.
   Negative training rows dropped; QRF only sees positives. Predicts
   no negative values at test time.

D. **No gate**: bare ``QRF`` on the full training set, no gate.
   Zeros come out as whatever the QRF happens to predict.

Metrics on the 20 % held-out partition:

- **Pinball loss at q=0.5** (median quantile loss): lower is better.
- **Zero-rate MAE**: absolute difference between the predicted and
  observed fraction of exact zeros. Approaches without a zero gate
  (D) are penalized here.
- **Sign-match rate**: fraction of held-out records where
  ``sign(pred) == sign(truth)``. C scores 0 on all true-negative
  records because it can never emit a negative value.
- **KS distance** between predicted and true marginal distribution.
- **Interior-band violations**: for a DGP with a designed gap between
  positive and negative regimes, the fraction of predictions that
  land in the "impossible" gap. Tripartite should have zero;
  approaches B and D should have nonzero.

Usage:

    uv run python experiments/regime_aware_holdout.py \
        --output experiments/regime_aware_holdout_results.json \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from microimpute.models.qrf import QRF
from microimpute.models.zero_inflated import ZeroInflatedImputer


# -------------------------------------------------------------------
# Synthetic data-generating process
# -------------------------------------------------------------------


@dataclass
class DGPConfig:
    """Parameters of the synthetic three-regime DGP.

    The target ``y`` is generated as a three-component mixture:
    - ``y = 0`` with probability ``p_zero(x)``
    - ``y ~ +exp(mu_pos(x) + sigma * z)`` with probability ``p_pos(x)``
    - ``y ~ -exp(mu_neg(x) + sigma * z)`` with probability ``p_neg(x)``

    Predictors ``x1, x2 ~ Uniform(0, 1)``. Mixing probabilities depend
    on ``x1 + x2`` so there is real conditional structure the gate
    classifier can learn. ``mu_pos`` and ``mu_neg`` use different
    coefficients so the positive and negative regimes are distinct
    populations.

    With ``gap_floor > 0`` positives are floored at exp(gap_floor) and
    negatives at -exp(gap_floor), creating a clean "interior band"
    between the regimes. Approaches that mix regimes will sometimes
    draw into the interior; the tripartite approach cannot.
    """

    n: int = 10_000
    gap_floor: float = 2.5  # exp(2.5) ≈ 12.2 ; interior band = (-12.2, 12.2)
    sigma: float = 0.6
    seed: int = 42


def generate_data(config: DGPConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    n = config.n
    x1 = rng.uniform(0, 1, size=n)
    x2 = rng.uniform(0, 1, size=n)
    # Mixing probabilities: three-way softmax over linear scores.
    logit_zero = 1.0 - 2.0 * (x1 + x2)  # higher when x small
    logit_pos = -1.0 + 3.0 * x1  # higher for large x1
    logit_neg = -1.0 + 3.0 * x2  # higher for large x2
    logits = np.stack([logit_neg, logit_zero, logit_pos], axis=1)
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    # Sample regime per record.
    u = rng.random(n)
    cum = np.cumsum(probs, axis=1)
    regime_idx = (cum >= u[:, None]).argmax(axis=1)
    # 0 => negative, 1 => zero, 2 => positive.
    y = np.zeros(n, dtype=float)
    z = rng.standard_normal(n)
    # Positive regime: distinct mu function. Hard-floor at
    # exp(gap_floor) so there is a genuine empty band between the
    # positive and negative regimes in the training data.
    pos_mask = regime_idx == 2
    mu_pos = 1.0 + 0.5 * x1[pos_mask] + 1.5 * x2[pos_mask]
    raw_pos = np.exp(mu_pos + config.sigma * z[pos_mask])
    y[pos_mask] = np.exp(config.gap_floor) + raw_pos
    # Negative regime: distinct mu function; hard ceiling at
    # -exp(gap_floor).
    neg_mask = regime_idx == 0
    mu_neg = 1.0 + 1.5 * x1[neg_mask] + 0.5 * x2[neg_mask]
    raw_neg = np.exp(mu_neg + config.sigma * z[neg_mask])
    y[neg_mask] = -(np.exp(config.gap_floor) + raw_neg)
    # Zero regime: exact 0.
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def split_train_test(
    df: pd.DataFrame, test_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed + 1)
    idx = rng.permutation(len(df))
    cut = int(len(df) * (1.0 - test_fraction))
    return df.iloc[idx[:cut]].reset_index(drop=True), df.iloc[
        idx[cut:]
    ].reset_index(drop=True)


# -------------------------------------------------------------------
# Four approaches
# -------------------------------------------------------------------


def fit_tripartite(train: pd.DataFrame):
    imputer = ZeroInflatedImputer(
        base_imputer_class=QRF,
        base_imputer_kwargs={},
    )
    result = imputer.fit(
        train, predictors=["x1", "x2"], imputed_variables=["y"]
    )
    return result, imputer.get_regime("y")


def _build_binary_gate_split(
    train: pd.DataFrame, include_negatives: bool
) -> tuple[Any, Any]:
    """Approach B or C: binary nonzero/zero gate + one QRF.

    include_negatives=True  → approach B (mixes pos + neg in QRF).
    include_negatives=False → approach C (drops neg, QRF on positives).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    x_values = train[["x1", "x2"]].to_numpy()
    y_values = train["y"].to_numpy()

    zero_atol = 1e-6
    if include_negatives:
        labels = (np.abs(y_values) > zero_atol).astype(int)
    else:
        labels = (y_values > zero_atol).astype(int)

    clf = HistGradientBoostingClassifier(random_state=42)
    clf.fit(x_values, labels)

    if include_negatives:
        nonzero_mask = np.abs(y_values) > zero_atol
    else:
        nonzero_mask = y_values > zero_atol

    qrf = QRF(log_level="ERROR")
    qrf_result = qrf.fit(
        train.loc[nonzero_mask].reset_index(drop=True),
        predictors=["x1", "x2"],
        imputed_variables=["y"],
    )
    return clf, qrf_result


def fit_binary_nonzero(train: pd.DataFrame):
    return _build_binary_gate_split(train, include_negatives=True)


def fit_positive_only(train: pd.DataFrame):
    return _build_binary_gate_split(train, include_negatives=False)


def fit_no_gate(train: pd.DataFrame):
    qrf = QRF(log_level="ERROR")
    return qrf.fit(
        train, predictors=["x1", "x2"], imputed_variables=["y"]
    )


# -------------------------------------------------------------------
# Prediction wrappers
# -------------------------------------------------------------------


def predict_tripartite(result, test: pd.DataFrame) -> np.ndarray:
    preds = result.predict(test[["x1", "x2"]])
    return preds["y"].to_numpy(dtype=float)


def predict_binary_gate(fitted_tuple, test: pd.DataFrame) -> np.ndarray:
    clf, qrf_result = fitted_tuple
    x_values = test[["x1", "x2"]].to_numpy()
    rng = np.random.default_rng(1234)
    proba = clf.predict_proba(x_values)
    # Positive class = 1 means nonzero (B) or positive (C).
    pos_idx = int(np.where(clf.classes_ == 1)[0][0])
    positive_prob = proba[:, pos_idx]
    u = rng.random(len(test))
    is_nonzero = u < positive_prob
    out = np.zeros(len(test), dtype=float)
    if is_nonzero.any():
        sub = qrf_result.predict(test.loc[is_nonzero, ["x1", "x2"]])
        if isinstance(sub, dict):
            sub = next(iter(sub.values()))
        out[is_nonzero] = sub["y"].to_numpy(dtype=float)
    return out


def predict_no_gate(qrf_result, test: pd.DataFrame) -> np.ndarray:
    preds = qrf_result.predict(test[["x1", "x2"]])
    if isinstance(preds, dict):
        preds = next(iter(preds.values()))
    return preds["y"].to_numpy(dtype=float)


# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------


def pinball_loss(pred: np.ndarray, truth: np.ndarray, q: float = 0.5) -> float:
    residual = truth - pred
    loss = np.where(residual >= 0, q * residual, (q - 1) * residual)
    return float(loss.mean())


def zero_rate_mae(pred: np.ndarray, truth: np.ndarray, atol: float = 1e-6) -> float:
    pred_zero = (np.abs(pred) <= atol).mean()
    true_zero = (np.abs(truth) <= atol).mean()
    return float(abs(pred_zero - true_zero))


def sign_match_rate(pred: np.ndarray, truth: np.ndarray, atol: float = 1e-6) -> float:
    def _sign(values: np.ndarray) -> np.ndarray:
        s = np.zeros_like(values, dtype=int)
        s[values > atol] = 1
        s[values < -atol] = -1
        return s

    return float((_sign(pred) == _sign(truth)).mean())


def ks_distance(pred: np.ndarray, truth: np.ndarray) -> float:
    from scipy import stats

    ks = stats.ks_2samp(pred, truth)
    return float(ks.statistic)


def interior_band_violation_rate(
    pred: np.ndarray, gap_floor: float, atol: float = 1e-6
) -> float:
    """Fraction of predictions in the interior band ``(-exp(gap_floor), exp(gap_floor))``
    that are not exact zero (those are legitimate)."""
    band = np.exp(gap_floor)
    interior = (np.abs(pred) < band) & (np.abs(pred) > atol)
    return float(interior.mean())


# -------------------------------------------------------------------
# Runner
# -------------------------------------------------------------------


def run_experiment(config: DGPConfig) -> Dict[str, Any]:
    data = generate_data(config)
    train, test = split_train_test(data, test_fraction=0.2, seed=config.seed)

    # Summary of the DGP itself on the holdout.
    test_y = test["y"].to_numpy(dtype=float)
    dgp_summary = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "test_frac_zero": float((np.abs(test_y) <= 1e-6).mean()),
        "test_frac_pos": float((test_y > 1e-6).mean()),
        "test_frac_neg": float((test_y < -1e-6).mean()),
    }

    # Approach A: tripartite.
    a_result, a_regime = fit_tripartite(train)
    a_pred = predict_tripartite(a_result, test)

    # Approach B: binary nonzero + single QRF (mixes pos+neg).
    b_fitted = fit_binary_nonzero(train)
    b_pred = predict_binary_gate(b_fitted, test)

    # Approach C: positive-only + QRF (the microplex-us bug).
    c_fitted = fit_positive_only(train)
    c_pred = predict_binary_gate(c_fitted, test)

    # Approach D: no gate.
    d_result = fit_no_gate(train)
    d_pred = predict_no_gate(d_result, test)

    def _score(name: str, pred: np.ndarray, extra: Dict[str, Any] = None) -> Dict[str, Any]:
        metrics = {
            "approach": name,
            "pinball_loss_q50": pinball_loss(pred, test_y, 0.5),
            "zero_rate_mae": zero_rate_mae(pred, test_y),
            "sign_match_rate": sign_match_rate(pred, test_y),
            "ks_distance": ks_distance(pred, test_y),
            "interior_band_violation_rate": interior_band_violation_rate(
                pred, config.gap_floor
            ),
            "pred_frac_zero": float((np.abs(pred) <= 1e-6).mean()),
            "pred_frac_pos": float((pred > 1e-6).mean()),
            "pred_frac_neg": float((pred < -1e-6).mean()),
        }
        if extra:
            metrics.update(extra)
        return metrics

    rows = [
        _score("A_tripartite", a_pred, {"detected_regime": a_regime}),
        _score("B_binary_nonzero_mixed_qrf", b_pred),
        _score("C_positive_only_qrf_bug", c_pred),
        _score("D_no_gate_bare_qrf", d_pred),
    ]

    return {
        "dgp_config": {
            "n": config.n,
            "gap_floor": config.gap_floor,
            "sigma": config.sigma,
            "seed": config.seed,
        },
        "dgp_summary_on_holdout": dgp_summary,
        "approaches": rows,
    }


def _aggregate(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate per-approach metrics across multiple seeds."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for run in runs:
        for row in run["approaches"]:
            grouped.setdefault(row["approach"], []).append(row)
    metric_keys = [
        "pinball_loss_q50",
        "zero_rate_mae",
        "sign_match_rate",
        "ks_distance",
        "interior_band_violation_rate",
        "pred_frac_zero",
        "pred_frac_pos",
        "pred_frac_neg",
    ]
    aggregated = []
    for name, rows in grouped.items():
        summary: Dict[str, Any] = {"approach": name, "n_seeds": len(rows)}
        for key in metric_keys:
            values = np.array([r[key] for r in rows], dtype=float)
            summary[f"{key}_mean"] = float(values.mean())
            summary[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        # Carry forward the detected regime if all seeds agree.
        regimes = {r.get("detected_regime") for r in rows}
        if regimes and None not in regimes and len(regimes) == 1:
            summary["detected_regime"] = next(iter(regimes))
        aggregated.append(summary)
    return aggregated


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--gap-floor", type=float, default=2.5)
    parser.add_argument("--sigma", type=float, default=0.6)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="Seeds to average over for multi-seed uncertainty.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent
        / "regime_aware_holdout_results.json",
    )
    args = parser.parse_args(argv)

    runs = []
    for seed in args.seeds:
        config = DGPConfig(
            n=args.n, gap_floor=args.gap_floor, sigma=args.sigma, seed=seed
        )
        runs.append(run_experiment(config))

    aggregated = _aggregate(runs)

    output = {
        "dgp_config_template": {
            "n": args.n,
            "gap_floor": args.gap_floor,
            "sigma": args.sigma,
            "seeds": args.seeds,
        },
        "dgp_summary_on_holdout_per_seed": [
            {**r["dgp_summary_on_holdout"], "seed": r["dgp_config"]["seed"]}
            for r in runs
        ],
        "aggregated_by_approach": aggregated,
        "per_run_raw": runs,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))

    print(
        f"\nMulti-seed holdout experiment ({len(args.seeds)} seeds, "
        f"n={args.n} each, gap_floor={args.gap_floor})"
    )
    print(
        f"\n{'approach':<32}{'pinball (mean±std)':>24}"
        f"{'zero_mae':>16}{'sign_hit':>16}{'ks':>14}{'interior':>14}"
    )
    for row in aggregated:
        print(
            f"{row['approach']:<32}"
            f"{row['pinball_loss_q50_mean']:>10.2f} ± {row['pinball_loss_q50_std']:<7.2f}   "
            f"{row['zero_rate_mae_mean']:>6.4f} ± {row['zero_rate_mae_std']:<6.4f}  "
            f"{row['sign_match_rate_mean']:>6.3f} ± {row['sign_match_rate_std']:<5.3f}  "
            f"{row['ks_distance_mean']:>5.3f} ± {row['ks_distance_std']:<5.3f}  "
            f"{row['interior_band_violation_rate_mean']:>5.3f} ± {row['interior_band_violation_rate_std']:<5.3f}"
        )
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
