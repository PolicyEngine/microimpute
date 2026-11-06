"""SPM poverty and MOOP distribution metrics with replicate-weight SEs.

Replacing a unit's reported MOOP ``m`` with an imputed value ``m'``
changes its SPM resources to ``SPM_RESOURCES + m - m'``; a person is in
SPM poverty when the unit's resources fall below ``SPM_POVTHRESHOLD``.
``check_spm_identity`` in ``cps_asec`` confirms both relations hold
exactly in the public-use file.

Standard errors use the 160 ASEC replicate weights with Census's
formula SE^2 = (4 / 160) * sum_r (theta_r - theta_0)^2. They treat the
fitted imputation models as fixed: they capture sampling variation in
the evaluation sample, not refitting variation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cps_asec import COVERAGE_LABELS, COVERAGE_ORDER, N_REPLICATES

REPLICATE_FACTOR = 4.0 / N_REPLICATES
RATIO_BANDS: list[tuple[float, float, str]] = [
    (-np.inf, 1.0, "Under 1.0"),
    (1.0, 1.25, "1.0 to 1.25"),
    (1.25, 1.5, "1.25 to 1.5"),
    (1.5, 2.0, "1.5 to 2.0"),
    (2.0, 3.0, "2.0 to 3.0"),
    (3.0, np.inf, "3.0 or more"),
]
QUANTILES = {"median": 0.5, "p75": 0.75, "p90": 0.9, "p95": 0.95, "p99": 0.99}


def weight_matrix(persons: pd.DataFrame) -> np.ndarray:
    """Columns: full-sample person weight, then the 160 replicates."""
    columns = ["weight"] + [f"repwgt_{i}" for i in range(1, N_REPLICATES + 1)]
    return persons[columns].to_numpy(dtype=float)


def replicate_se(estimates: np.ndarray) -> np.ndarray:
    """SE from an array whose last axis is (full sample, 160 replicates)."""
    deviations = estimates[..., 1:] - estimates[..., :1]
    return np.sqrt(REPLICATE_FACTOR * (deviations**2).sum(axis=-1))


def person_groups(persons: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    """Boolean membership of each person in each reporting group.

    Groups: everyone; age under 18, 18 to 64, 65 or older; disability
    (PRDISFLG, asked of civilians 15 and older); current coverage
    category; and the unit's resources before MOOP relative to its
    threshold.
    """
    age = persons["A_AGE"]
    ratio = persons["SPM_ID"].map(units["pre_moop_ratio"])
    groups = {
        ("all", "All people"): np.ones(len(persons), dtype=bool),
        ("age", "Under 18"): age < 18,
        ("age", "18 to 64"): (age >= 18) & (age < 65),
        ("age", "65 and older"): age >= 65,
        ("disability", "With a disability"): persons["PRDISFLG"] == 1,
        ("disability", "No disability"): persons["PRDISFLG"] == 2,
        ("disability", "Not asked (mostly under 15)"): persons["PRDISFLG"] == -1,
    }
    for category in COVERAGE_ORDER:
        groups[("coverage", COVERAGE_LABELS[category])] = (
            persons["coverage"] == category
        )
    for low, high, label in RATIO_BANDS:
        groups[("pre_moop_ratio", label)] = (ratio >= low) & (ratio < high)
    frame = pd.DataFrame(
        {f"{dim}::{label}": np.asarray(mask) for (dim, label), mask in groups.items()}
    )
    return frame


def poverty_flags(
    persons: pd.DataFrame, units: pd.DataFrame, unit_moop: np.ndarray
) -> np.ndarray:
    """Person-level SPM poverty under imputed unit MOOP.

    Args:
        persons: Person frame with SPM_ID, SPM_RESOURCES, SPM_MEDXPNS and
            SPM_POVTHRESHOLD.
        units: Unit frame indexed by SPM_ID, in the order of ``unit_moop``'s
            last axis.
        unit_moop: Imputed MOOP, shape ``(n_draws, n_units)`` or
            ``(n_units,)``.

    Returns:
        Boolean poverty flags of shape ``(n_draws, n_persons)``.
    """
    unit_moop = np.atleast_2d(unit_moop)
    position = pd.Index(units.index).get_indexer(persons["SPM_ID"])
    if (position < 0).any():
        raise ValueError("Persons reference SPM units missing from units.")
    pre_moop = (persons["SPM_RESOURCES"] + persons["SPM_MEDXPNS"]).to_numpy(dtype=float)
    threshold = persons["SPM_POVTHRESHOLD"].to_numpy(dtype=float)
    resources = pre_moop[None, :] - unit_moop[:, position]
    return resources < threshold[None, :]


def group_rates(
    flags: np.ndarray, groups: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Weighted share flagged within each group, for every weight column.

    Args:
        flags: Person values, shape ``(n_persons,)`` (may be fractional,
            e.g. the share of draws in which a person is poor).
        groups: Membership, shape ``(n_persons, n_groups)``.
        weights: Weights, shape ``(n_persons, n_weights)``.

    Returns:
        Array of shape ``(n_groups, n_weights)``.
    """
    member = groups.astype(float)
    numerator = (member * flags[:, None]).T @ weights
    denominator = member.T @ weights
    # An empty group has no rate; return NaN without a warning.
    with np.errstate(invalid="ignore", divide="ignore"):
        return numerator / denominator


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Smallest value whose weighted cumulative share reaches ``q``."""
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = np.searchsorted(cumulative, q * cumulative[-1], side="left")
    return float(values[order][min(index, len(values) - 1)])


def _mean_or_nan(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean, or NaN for an empty selection."""
    if len(values) == 0:
        return float("nan")
    return float(np.average(values, weights=weights))


def ratio_band_index(ratio: np.ndarray) -> np.ndarray:
    """Index into ``RATIO_BANDS`` for each resources-to-threshold ratio."""
    edges = np.array([high for _, high, _ in RATIO_BANDS[:-1]])
    return np.searchsorted(edges, ratio, side="right")


def moop_distribution(
    moop: np.ndarray,
    weights: np.ndarray,
    near_line: np.ndarray,
    band: np.ndarray,
) -> dict[str, float]:
    """Weighted SPM-unit MOOP statistics for one set of values.

    ``band_means`` gives mean MOOP within each band of resources before
    MOOP relative to the threshold (``RATIO_BANDS`` order).
    """
    stats = {
        "band_means": [
            _mean_or_nan(moop[band == b], weights[band == b])
            for b in range(len(RATIO_BANDS))
        ],
        "mean": float(np.average(moop, weights=weights)),
        "share_zero": float(np.average(moop == 0, weights=weights)),
        "aggregate_billions": float((moop * weights).sum() / 1e9),
        "near_line_mean": _mean_or_nan(moop[near_line], weights[near_line]),
    }
    for name, q in QUANTILES.items():
        stats[name] = weighted_quantile(moop, weights, q)
    return stats


def average_distribution(
    draws: np.ndarray,
    weights: np.ndarray,
    near_line: np.ndarray,
    band: np.ndarray,
) -> dict[str, float]:
    """Average the distribution statistics over draws."""
    per_draw = [moop_distribution(d, weights, near_line, band) for d in draws]
    out: dict = {
        k: float(np.mean([s[k] for s in per_draw]))
        for k in per_draw[0]
        if k != "band_means"
    }
    out["band_means"] = np.mean([s["band_means"] for s in per_draw], axis=0).tolist()
    if len(per_draw) > 1:
        out["mean_draw_sd"] = float(np.std([s["mean"] for s in per_draw], ddof=1))
    return out


def evaluate(
    persons: pd.DataFrame,
    units: pd.DataFrame,
    imputations: dict[str, np.ndarray],
) -> dict:
    """Compute every reported metric for reported and imputed MOOP.

    Args:
        persons: Person frame from ``load_persons``.
        units: Unit frame from ``build_units``.
        imputations: Method name to array ``(n_draws, n_units)``.

    Returns:
        A JSON-serializable dict of results.
    """
    weights = weight_matrix(persons)
    groups = person_groups(persons, units)
    group_matrix = groups.to_numpy()
    reported = poverty_flags(persons, units, units["moop"].to_numpy())[0]
    published_check = (reported == (persons["SPM_POOR"] == 1).to_numpy()).all()
    if not published_check:
        raise ValueError("Recomputed poverty does not match SPM_POOR.")
    reported_f = reported.astype(float)
    reported_rates = group_rates(reported_f, group_matrix, weights)

    unit_weight = units["unit_weight"].to_numpy()
    near_line = (units["pre_moop_ratio"] >= 1.0) & (units["pre_moop_ratio"] < 2.0)
    near_line = near_line.to_numpy()
    band = ratio_band_index(units["pre_moop_ratio"].to_numpy())
    no_allocation_person = (
        persons["SPM_ID"].map(units["any_allocated_moop"]).to_numpy() == 0
    )
    subset_groups = np.column_stack(
        [np.ones(len(persons), dtype=bool), no_allocation_person]
    )
    reported_subset = group_rates(reported_f, subset_groups, weights)

    results: dict = {
        "groups": [c.split("::") for c in groups.columns],
        "group_person_counts": groups.sum().astype(int).tolist(),
        "methods": {},
    }

    def pack(rates: np.ndarray, diff: np.ndarray) -> dict[str, list[float]]:
        return {
            "rate": rates[:, 0].tolist(),
            "rate_se": replicate_se(rates).tolist(),
            "diff": diff[:, 0].tolist(),
            "diff_se": replicate_se(diff).tolist(),
        }

    results["methods"]["reported"] = {
        **pack(reported_rates, np.zeros_like(reported_rates)),
        "distribution": moop_distribution(
            units["moop"].to_numpy(), unit_weight, near_line, band
        ),
        "n_draws": 1,
    }
    w0 = weights[:, 0]
    for method, draws in imputations.items():
        flags = poverty_flags(persons, units, draws)
        mean_flag = flags.mean(axis=0)
        rates = group_rates(mean_flag, group_matrix, weights)
        entry = pack(rates, rates - reported_rates)
        # Gross flows relative to reported MOOP, averaged over draws.
        newly_poor = (flags & ~reported[None, :]).mean(axis=0)
        newly_nonpoor = (~flags & reported[None, :]).mean(axis=0)
        entry["newly_poor_share"] = float(np.average(newly_poor, weights=w0))
        entry["newly_nonpoor_share"] = float(np.average(newly_nonpoor, weights=w0))
        per_draw_overall = [
            float(np.average(f.astype(float), weights=w0)) for f in flags
        ]
        entry["overall_draw_sd"] = (
            float(np.std(per_draw_overall, ddof=1)) if len(flags) > 1 else 0.0
        )
        entry["distribution"] = average_distribution(
            draws, unit_weight, near_line, band
        )
        subset_rates = group_rates(mean_flag, subset_groups, weights)
        subset_diff = subset_rates - reported_subset
        entry["no_allocation_subset"] = {
            "rate": float(subset_rates[1, 0]),
            "diff": float(subset_diff[1, 0]),
            "diff_se": float(replicate_se(subset_diff)[1]),
        }
        entry["n_draws"] = len(draws)
        results["methods"][method] = entry
    results["methods"]["reported"]["no_allocation_subset"] = {
        "rate": float(reported_subset[1, 0]),
        "diff": 0.0,
        "diff_se": 0.0,
    }
    return results
