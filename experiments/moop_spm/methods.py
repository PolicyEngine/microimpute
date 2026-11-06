"""MOOP imputation methods compared on held-out CPS ASEC SPM units.

Every method takes the same donor and recipient predictor matrices and
returns an array of shape ``(n_draws, n_recipients)`` of imputed SPM-unit
MOOP. Deterministic methods repeat one row.

Methods:

- ``logit_cell_mean``: a logistic regression for any MOOP; recipients
  with predicted probability of at least 0.5 are "predicted spenders"
  and get the mean MOOP of donor spenders in their cell.
- ``logit_reg_mean``: the same participation rule; predicted spenders
  get the OLS prediction (in dollars) from a regression on donor
  spenders.
- ``logit_draw_hotdeck``: participation drawn from the logistic
  probability; spenders get the MOOP of a random donor spender in their
  cell.
- ``pmm``: predictive mean matching within coverage groups: an OLS of
  log(1 + MOOP) on the predictors, then each recipient takes the
  reported MOOP of one of its five nearest donors by predicted value.
- ``qrf``: microimpute's ``QRF`` fit on all donors, one random quantile
  draw per recipient.
- ``zi_qrf``: microimpute's ``ZeroInflatedImputer`` around ``QRF``.
- ``microcosm_qrf``: Microcosm's ``RegimeGatedQRF``, the model that
  imputes PolicyEngine's microdata, fit with the donors' unit weights.
  On Microcosm's own data frames, a fit uses the frame's weights unless
  told not to.
- ``microcosm_qrf_unweighted``: the same model fit without weights, like
  every other method here.

Decomposition variants cross the participation rule (threshold or
draw) with the amount rule (cell mean or donor draw), and
``logit_expected_value`` assigns every recipient the probability times
the cell mean, and ``logit_group_mean`` uses the 0.5 rule with means by
coverage group only (five cells).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from microimpute.models.qrf import QRF
from microimpute.models.zero_inflated import ZeroInflatedImputer

MIN_CELL_DONORS = 30
PMM_NEIGHBORS = 5
PARTICIPATION_CUTOFF = 0.5

# Imputation cells, finest first. A recipient uses the finest level
# whose cell holds at least MIN_CELL_DONORS donor spenders.
CELL_LEVELS: list[tuple[str, ...]] = [
    ("group", "age_band", "income_band", "size_band"),
    ("group", "age_band", "income_band"),
    ("group", "age_band"),
    ("group",),
    (),
]
# Coarse cells for the sensitivity check: coverage group only.
COARSE_CELL_LEVELS: list[tuple[str, ...]] = [("group",), ()]


def cell_keys(units: pd.DataFrame) -> pd.DataFrame:
    """Return the cell variables for each SPM unit.

    - ``group``: coverage group of the unit (see ``unit_coverage_group``).
    - ``age_band``: head's age under 45, 45 to 64, 65 or older.
    - ``income_band``: cash income relative to the SPM threshold, under 1,
      1 to 2, 2 to 4, 4 or more.
    - ``size_band``: unit size 1, 2, or 3 or more.
    """
    age_band = pd.cut(
        units["head_age"], [-1, 44, 64, 200], labels=["u45", "45_64", "65p"]
    )
    income_band = pd.cut(
        units["cash_income"] / units["threshold"],
        [-np.inf, 1.0, 2.0, 4.0, np.inf],
        right=False,
        labels=["lt1", "1_2", "2_4", "4p"],
    )
    size_band = pd.cut(units["n_persons"], [0, 1, 2, np.inf], labels=["1", "2", "3p"])
    return pd.DataFrame(
        {
            "group": units["group"].astype(str),
            "age_band": age_band.astype(str),
            "income_band": income_band.astype(str),
            "size_band": size_band.astype(str),
        },
        index=units.index,
    )


def _key(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    if not columns:
        return np.full(len(frame), "all", dtype=object)
    return frame[list(columns)].astype(str).agg("|".join, axis=1).to_numpy()


@dataclass
class CellAssignment:
    """Donor spender amounts available to each recipient."""

    pools: list[np.ndarray]
    levels: np.ndarray

    def means(self) -> np.ndarray:
        return np.array([pool.mean() for pool in self.pools])

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        return np.array([pool[rng.integers(len(pool))] for pool in self.pools])


def assign_cells(
    donor_cells: pd.DataFrame,
    donor_moop: np.ndarray,
    recipient_cells: pd.DataFrame,
    min_donors: int = MIN_CELL_DONORS,
    cell_levels: Sequence[tuple[str, ...]] = tuple(CELL_LEVELS),
) -> CellAssignment:
    """Match each recipient to the finest cell with enough donor spenders.

    Args:
        donor_cells: Cell variables of donors.
        donor_moop: Reported MOOP of donors.
        recipient_cells: Cell variables of recipients.
        min_donors: Minimum number of donor spenders a cell needs.
        cell_levels: Cell definitions to try, finest first.

    Returns:
        For each recipient, the array of positive donor MOOP values in its
        cell and the index into ``cell_levels`` used (0 = finest).
    """
    spender = donor_moop > 0
    pools: list[np.ndarray] = [None] * len(recipient_cells)  # type: ignore
    levels = np.full(len(recipient_cells), -1)
    unresolved = np.ones(len(recipient_cells), dtype=bool)
    for level, columns in enumerate(cell_levels):
        donor_key = _key(donor_cells.loc[spender], columns)
        amounts = donor_moop[spender]
        grouped: dict[str, np.ndarray] = {
            key: amounts[donor_key == key] for key in np.unique(donor_key)
        }
        recipient_key = _key(recipient_cells, columns)
        for i in np.flatnonzero(unresolved):
            pool = grouped.get(recipient_key[i])
            if pool is not None and len(pool) >= min_donors:
                pools[i] = pool
                levels[i] = level
                unresolved[i] = False
        if not unresolved.any():
            break
    if unresolved.any():
        raise ValueError("Some recipients found no donor cell.")
    return CellAssignment(pools=pools, levels=levels)


def fit_participation(X: pd.DataFrame, moop: np.ndarray) -> Pipeline:
    """Fit an (effectively unpenalized) logistic regression for MOOP > 0."""
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1e6, max_iter=10_000),
    )
    model.fit(X.to_numpy(), (moop > 0).astype(int))
    return model


def participation_probability(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X.to_numpy())[:, 1]


@dataclass
class FoldData:
    """Donor and recipient data for one cross-fitting fold.

    ``weight_donor`` holds the donors' SPM unit weights; only the weighted
    Microcosm fit reads them.
    """

    X_donor: pd.DataFrame
    moop_donor: np.ndarray
    weight_donor: np.ndarray
    cells_donor: pd.DataFrame
    group_donor: np.ndarray
    X_recipient: pd.DataFrame
    cells_recipient: pd.DataFrame
    group_recipient: np.ndarray


def _repeat(values: np.ndarray, n_draws: int) -> np.ndarray:
    return np.tile(values, (n_draws, 1))


def logit_family(
    fold: FoldData, n_draws: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Run every logistic-participation method on one fold.

    Sharing the logistic fit and the cell assignment across the variants
    keeps their differences down to the participation and amount rules.
    """
    model = fit_participation(fold.X_donor, fold.moop_donor)
    prob = participation_probability(model, fold.X_recipient)
    cells = assign_cells(fold.cells_donor, fold.moop_donor, fold.cells_recipient)
    cell_mean = cells.means()
    group_mean = assign_cells(
        fold.cells_donor,
        fold.moop_donor,
        fold.cells_recipient,
        cell_levels=COARSE_CELL_LEVELS,
    ).means()

    spenders = fold.moop_donor > 0
    regression = LinearRegression().fit(
        fold.X_donor.to_numpy()[spenders], fold.moop_donor[spenders]
    )
    reg_mean = np.clip(regression.predict(fold.X_recipient.to_numpy()), 0.0, None)

    predicted_spender = prob >= PARTICIPATION_CUTOFF
    out = {
        "logit_cell_mean": _repeat(
            np.where(predicted_spender, cell_mean, 0.0), n_draws
        ),
        "logit_reg_mean": _repeat(np.where(predicted_spender, reg_mean, 0.0), n_draws),
        "logit_expected_value": _repeat(prob * cell_mean, n_draws),
        "logit_group_mean": _repeat(
            np.where(predicted_spender, group_mean, 0.0), n_draws
        ),
    }
    draw_hotdeck = np.empty((n_draws, len(prob)))
    threshold_hotdeck = np.empty((n_draws, len(prob)))
    draw_cell_mean = np.empty((n_draws, len(prob)))
    for r in range(n_draws):
        participates = rng.random(len(prob)) < prob
        donor_amount = cells.draw(rng)
        draw_hotdeck[r] = np.where(participates, donor_amount, 0.0)
        threshold_hotdeck[r] = np.where(predicted_spender, donor_amount, 0.0)
        draw_cell_mean[r] = np.where(participates, cell_mean, 0.0)
    out["logit_draw_hotdeck"] = draw_hotdeck
    out["logit_threshold_hotdeck"] = threshold_hotdeck
    out["logit_draw_cell_mean"] = draw_cell_mean
    return out


def nearest_donors(
    donor_pred: np.ndarray, recipient_pred: np.ndarray, k: int
) -> np.ndarray:
    """Indices of each recipient's ``k`` nearest donors by predicted value.

    Args:
        donor_pred: Predicted values of donors.
        recipient_pred: Predicted values of recipients.
        k: Number of neighbors; capped at the number of donors.

    Returns:
        Integer array of shape ``(n_recipients, min(k, n_donors))`` with
        positions into ``donor_pred``.
    """
    k = min(k, len(donor_pred))
    order = np.argsort(donor_pred, kind="stable")
    sorted_pred = donor_pred[order]
    position = np.searchsorted(sorted_pred, recipient_pred)
    # The k nearest donors sit in the window of 2k sorted donors around the
    # insertion point.
    offsets = np.arange(-k, k)
    candidates = np.clip(position[:, None] + offsets[None, :], 0, len(sorted_pred) - 1)
    distance = np.abs(sorted_pred[candidates] - recipient_pred[:, None])
    # Clipping at either end repeats an index; count each donor once.
    repeated = np.concatenate(
        [
            np.zeros((len(candidates), 1), dtype=bool),
            np.diff(candidates, axis=1) == 0,
        ],
        axis=1,
    )
    distance = np.where(repeated, np.inf, distance)
    closest = np.argsort(distance, axis=1, kind="stable")[:, :k]
    return order[np.take_along_axis(candidates, closest, axis=1)]


def pmm(
    fold: FoldData,
    n_draws: int,
    rng: np.random.Generator,
    neighbors: int = PMM_NEIGHBORS,
) -> np.ndarray:
    """Predictive mean matching within coverage groups.

    Within each coverage group, an OLS regression of log(1 + MOOP) on the
    predictors (all donors, zeros included) gives a predicted value for
    donors and recipients. Each recipient takes the reported MOOP of one
    donor drawn at random from its ``neighbors`` nearest donors by
    predicted value, so imputed values are always reported values.
    """
    draws = np.empty((n_draws, len(fold.X_recipient)))
    for group in np.unique(fold.group_recipient):
        donor_mask = fold.group_donor == group
        recipient_mask = fold.group_recipient == group
        X_d = fold.X_donor.to_numpy()[donor_mask]
        y_d = fold.moop_donor[donor_mask]
        model = LinearRegression().fit(X_d, np.log1p(y_d))
        nearest = nearest_donors(
            model.predict(X_d),
            model.predict(fold.X_recipient.to_numpy()[recipient_mask]),
            neighbors,
        )
        rows = np.arange(len(nearest))
        for r in range(n_draws):
            pick = nearest[rows, rng.integers(nearest.shape[1], size=len(rows))]
            draws[r, recipient_mask] = y_d[pick]
    return draws


def _as_frame(prediction) -> pd.DataFrame:
    if isinstance(prediction, dict):
        return next(iter(prediction.values()))
    return prediction


def qrf(
    fold: FoldData,
    n_draws: int,
    seed: int,
    n_jobs: int = 4,
) -> np.ndarray:
    """microimpute QRF on all donors; each draw samples a random quantile."""
    train = fold.X_donor.copy()
    train["moop"] = fold.moop_donor
    predictors = list(fold.X_donor.columns)
    fitted = QRF(seed=seed, log_level="ERROR").fit(
        train,
        predictors=predictors,
        imputed_variables=["moop"],
        not_numeric_categorical=predictors,
        n_jobs=n_jobs,
    )
    draws = np.empty((n_draws, len(fold.X_recipient)))
    for r in range(n_draws):
        draws[r] = _as_frame(fitted.predict(fold.X_recipient))["moop"].to_numpy()
    return draws


def zi_qrf(fold: FoldData, n_draws: int, seed: int) -> np.ndarray:
    """microimpute ZeroInflatedImputer (gate + QRF on positive donors)."""
    train = fold.X_donor.copy()
    train["moop"] = fold.moop_donor
    predictors = list(fold.X_donor.columns)
    fitted = ZeroInflatedImputer(
        base_imputer_class=QRF, seed=seed, log_level="ERROR"
    ).fit(
        train,
        predictors=predictors,
        imputed_variables=["moop"],
        not_numeric_categorical=predictors,
    )
    draws = np.empty((n_draws, len(fold.X_recipient)))
    for r in range(n_draws):
        draws[r] = _as_frame(fitted.predict(fold.X_recipient))["moop"].to_numpy()
    return draws


@contextmanager
def microcosm_threads(n_jobs: int) -> Iterator[None]:
    """Cap Microcosm's threads at ``n_jobs``.

    Microcosm reads its forest-fitting and drawing widths from environment
    variables that its ``qrf`` module names; when they are unset, it fits
    and draws on every core. Its gate, a scikit-learn gradient-boosted
    classifier, runs on OpenMP threads, which ``threadpoolctl`` caps. The
    previous settings come back on exit.
    """
    from microcosm.fit import qrf as microcosm_qrf_module
    from threadpoolctl import threadpool_limits

    names = (
        microcosm_qrf_module._N_JOBS_ENV,
        microcosm_qrf_module._PREDICT_WORKERS_ENV,
    )
    saved = {name: os.environ.get(name) for name in names}
    os.environ.update({name: str(n_jobs) for name in names})
    try:
        with threadpool_limits(limits=n_jobs, user_api="openmp"):
            yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def microcosm_qrf(
    fold: FoldData,
    n_draws: int,
    seed: int,
    weighted: bool,
    n_jobs: int = 4,
) -> np.ndarray:
    """Microcosm's ``RegimeGatedQRF`` on all donors.

    A gradient-boosted classifier gates zero against positive MOOP, and a
    quantile regression forest grown on donor spenders gives each gated
    spender the amount at a random quantile of its predicted distribution.
    With ``weighted``, the gate takes the donors' unit weights as sample
    weights and the forest grows on spenders resampled in proportion to
    them; without it, both ignore the weights. Each draw calls ``predict``
    again on the one fitted model, which advances the model's own random
    stream.
    """
    from microcosm.fit import NO_WEIGHTS, RegimeGatedQRF

    train = fold.X_donor.copy()
    train["moop"] = fold.moop_donor
    predictors = list(fold.X_donor.columns)
    weights = fold.weight_donor if weighted else NO_WEIGHTS
    draws = np.empty((n_draws, len(fold.X_recipient)))
    with microcosm_threads(n_jobs):
        fitted = RegimeGatedQRF(seed=seed).fit(
            train, predictors, ["moop"], weights=weights
        )
        for r in range(n_draws):
            draws[r] = fitted.predict(fold.X_recipient)["moop"].to_numpy()
    return draws


MAIN_METHODS = [
    "logit_cell_mean",
    "logit_reg_mean",
    "logit_draw_hotdeck",
    "pmm",
    "qrf",
    "zi_qrf",
    "microcosm_qrf",
    "microcosm_qrf_unweighted",
]
DECOMPOSITION_METHODS = [
    "logit_threshold_hotdeck",
    "logit_draw_cell_mean",
    "logit_expected_value",
    "logit_group_mean",
]
DETERMINISTIC = {
    "logit_cell_mean",
    "logit_reg_mean",
    "logit_expected_value",
    "logit_group_mean",
}
METHOD_LABELS = {
    "reported": "Reported (CPS ASEC)",
    "logit_cell_mean": "(a) Logit, p >= 0.5, cell mean",
    "logit_reg_mean": "(a) Logit, p >= 0.5, regression mean",
    "logit_draw_hotdeck": "(b) Logit draw, donor amount in cell",
    "pmm": "PMM within coverage groups",
    "qrf": "(c) QRF (microimpute)",
    "zi_qrf": "(c) Zero-inflated QRF (microimpute)",
    "microcosm_qrf": "(c) Regime-gated QRF (Microcosm), unit weights",
    "microcosm_qrf_unweighted": "(c) Regime-gated QRF (Microcosm), unweighted",
    "logit_threshold_hotdeck": "Logit, p >= 0.5, donor amount in cell",
    "logit_draw_cell_mean": "Logit draw, cell mean",
    "logit_expected_value": "p x cell mean for every unit",
    "logit_group_mean": "Logit, p >= 0.5, coverage-group mean",
}
STOCHASTIC = {
    "logit_draw_hotdeck",
    "pmm",
    "qrf",
    "zi_qrf",
    "microcosm_qrf",
    "microcosm_qrf_unweighted",
    "logit_threshold_hotdeck",
    "logit_draw_cell_mean",
}

ImputeFn = Callable[[FoldData, int, np.random.Generator], np.ndarray]
