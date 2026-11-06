"""Property tests for the MOOP-imputation SPM poverty experiment.

The experiment lives in ``experiments/moop_spm``. These tests use small
synthetic data, so they run without the CPS ASEC download. They check
the invariants the published comparison relies on:

- Fold assignment partitions households and keeps each household whole.
- Poverty is recomputed exactly from resources, MOOP and thresholds, and
  it never falls when any unit's MOOP rises.
- People whose unit is poor before MOOP stay poor under any imputation.
- Weighted group rates, replicate SEs and weighted quantiles match direct
  reference computations.
- Cell pools hold only same-cell donor spenders, and the nearest-donor
  search agrees with a brute-force search.
- The logistic-participation variants follow their stated rules.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from experiments.moop_spm.evaluate import (
    RATIO_BANDS,
    evaluate,
    group_rates,
    poverty_flags,
    ratio_band_index,
    replicate_se,
    weighted_quantile,
)
from experiments.moop_spm.methods import (
    CELL_LEVELS,
    PARTICIPATION_CUTOFF,
    FoldData,
    _key,
    assign_cells,
    fit_participation,
    logit_family,
    nearest_donors,
    participation_probability,
)
from experiments.moop_spm.run import assign_folds

SETTINGS = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def synthetic_spm(seed: int, n_units: int = 60):
    """Persons and units frames with the columns ``evaluate`` reads."""
    rng = np.random.default_rng(seed)
    unit_ids = np.arange(1, n_units + 1) * 7
    threshold = rng.uniform(20_000, 40_000, n_units)
    moop = np.where(rng.random(n_units) < 0.1, 0.0, rng.gamma(1.0, 5_000, n_units))
    pre_moop = threshold * rng.uniform(0.5, 3.0, n_units)
    units = pd.DataFrame(
        {
            "moop": moop,
            "unit_weight": rng.uniform(100, 3_000, n_units),
            "pre_moop_ratio": pre_moop / threshold,
            "any_allocated_moop": rng.integers(0, 2, n_units),
        },
        index=pd.Index(unit_ids, name="SPM_ID"),
    )
    sizes = rng.integers(1, 4, n_units)
    person_unit = np.repeat(np.arange(n_units), sizes)
    n_persons = len(person_unit)
    persons = pd.DataFrame(
        {
            "SPM_ID": unit_ids[person_unit],
            "A_AGE": rng.integers(0, 90, n_persons),
            "PRDISFLG": rng.choice([-1, 1, 2], n_persons),
            "coverage": rng.choice(
                [
                    "medicare",
                    "employer",
                    "direct",
                    "medicaid",
                    "military",
                    "uninsured",
                ],
                n_persons,
            ),
            "SPM_MEDXPNS": moop[person_unit],
            "SPM_RESOURCES": (pre_moop - moop)[person_unit],
            "SPM_POVTHRESHOLD": threshold[person_unit],
            "weight": rng.uniform(100, 3_000, n_persons),
        }
    )
    persons["SPM_POOR"] = (
        persons["SPM_RESOURCES"] < persons["SPM_POVTHRESHOLD"]
    ).astype(int)
    replicates = persons["weight"].to_numpy()[:, None] * rng.uniform(
        0.5, 1.5, (n_persons, 160)
    )
    replicate_frame = pd.DataFrame(
        replicates, columns=[f"repwgt_{i}" for i in range(1, 161)]
    )
    persons = pd.concat([persons, replicate_frame], axis=1)
    return persons, units


@SETTINGS
@given(
    households=st.lists(st.integers(1, 40), min_size=5, max_size=120),
    n_folds=st.integers(2, 6),
    seed=st.integers(0, 2**31),
)
def test_folds_partition_households(households, n_folds, seed):
    series = pd.Series(households)
    folds = assign_folds(series, n_folds, seed)
    assert set(np.unique(folds)) <= set(range(n_folds))
    # Every household lands in exactly one fold.
    assert (series.groupby(folds).nunique().sum()) == series.nunique()
    per_household = pd.Series(folds).groupby(series.to_numpy()).nunique()
    assert (per_household == 1).all()
    # Household counts per fold differ by at most one.
    household_counts = series.groupby(folds).nunique()
    assert household_counts.max() - household_counts.min() <= 1
    assert (assign_folds(series, n_folds, seed) == folds).all()


@SETTINGS
@given(seed=st.integers(0, 2**31), bump_seed=st.integers(0, 2**31))
def test_poverty_recomputation_and_monotonicity(seed, bump_seed):
    persons, units = synthetic_spm(seed)
    reported = poverty_flags(persons, units, units["moop"].to_numpy())[0]
    assert (reported == (persons["SPM_POOR"] == 1)).all()

    bump = np.random.default_rng(bump_seed).uniform(0, 20_000, len(units))
    higher = poverty_flags(persons, units, units["moop"].to_numpy() + bump)[0]
    # More MOOP can only push people into poverty.
    assert (higher >= reported).all()

    zero = poverty_flags(persons, units, np.zeros(len(units)))[0]
    assert (zero <= reported).all()
    pre_moop_poor = persons["SPM_ID"].map(units["pre_moop_ratio"]).to_numpy() < 1.0
    # Units poor before MOOP stay poor under any nonnegative MOOP.
    assert zero[pre_moop_poor].all()
    assert higher[pre_moop_poor].all()


@SETTINGS
@given(seed=st.integers(0, 2**31))
def test_group_rates_match_np_average(seed):
    rng = np.random.default_rng(seed)
    n = 50
    flags = rng.random(n)
    weights = rng.uniform(0.1, 10, (n, 3))
    groups = rng.random((n, 4)) < 0.5
    groups[0] = True  # keep every group nonempty
    rates = group_rates(flags, groups, weights)
    for g in range(groups.shape[1]):
        for w in range(weights.shape[1]):
            expected = np.average(flags[groups[:, g]], weights=weights[groups[:, g], w])
            assert rates[g, w] == pytest.approx(expected)
    assert ((rates >= 0) & (rates <= 1)).all()


@SETTINGS
@given(
    base=st.floats(-5, 5),
    deviations=st.lists(st.floats(-1, 1), min_size=160, max_size=160),
    scale=st.floats(-3, 3),
)
def test_replicate_se_formula(base, deviations, scale):
    estimates = np.array([base] + [base + d for d in deviations])
    expected = np.sqrt(4 / 160 * np.sum(np.square(deviations)))
    assert replicate_se(estimates) == pytest.approx(expected)
    assert replicate_se(scale * estimates) == pytest.approx(abs(scale) * expected)
    assert replicate_se(np.full(161, base)) == 0


@SETTINGS
@given(
    values=st.lists(st.floats(-1e6, 1e6), min_size=1, max_size=60),
    q=st.floats(0.01, 0.99),
)
def test_weighted_quantile_equal_weights(values, q):
    values = np.array(values)
    result = weighted_quantile(values, np.ones(len(values)), q)
    assert result in values
    assert result == pytest.approx(np.quantile(values, q, method="inverted_cdf"))


def _cells(rng: np.random.Generator, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group": rng.choice(["medicare", "employer", "public"], n),
            "age_band": rng.choice(["u45", "45_64", "65p"], n),
            "income_band": rng.choice(["lt1", "1_2", "2_4", "4p"], n),
            "size_band": rng.choice(["1", "2", "3p"], n),
        }
    )


@SETTINGS
@given(seed=st.integers(0, 2**31), min_donors=st.integers(1, 40))
def test_cell_pools_are_same_cell_donor_spenders(seed, min_donors):
    rng = np.random.default_rng(seed)
    donors = _cells(rng, 400)
    moop = np.where(rng.random(400) < 0.2, 0.0, rng.gamma(1.0, 3_000, 400))
    recipients = _cells(rng, 50)
    assignment = assign_cells(donors, moop, recipients, min_donors=min_donors)
    spender = moop > 0
    for i, (pool, level) in enumerate(zip(assignment.pools, assignment.levels)):
        assert len(pool) >= min_donors
        assert (pool > 0).all()
        columns = CELL_LEVELS[level]
        donor_key = _key(donors.loc[spender], columns)
        recipient_key = _key(recipients.iloc[[i]], columns)[0]
        expected = np.sort(moop[spender][donor_key == recipient_key])
        assert np.array_equal(np.sort(pool), expected)
        # No finer level had enough donor spenders.
        for finer in CELL_LEVELS[:level]:
            finer_key = _key(recipients.iloc[[i]], finer)[0]
            assert (_key(donors.loc[spender], finer) == finer_key).sum() < min_donors
    assert np.allclose(assignment.means(), [p.mean() for p in assignment.pools])


@SETTINGS
@given(
    donor_pred=st.lists(st.floats(-50, 50, allow_nan=False), min_size=1, max_size=40),
    recipient_pred=st.lists(
        st.floats(-60, 60, allow_nan=False), min_size=1, max_size=20
    ),
    k=st.integers(1, 8),
)
def test_nearest_donors_matches_brute_force(donor_pred, recipient_pred, k):
    donor_pred = np.array(donor_pred)
    recipient_pred = np.array(recipient_pred)
    nearest = nearest_donors(donor_pred, recipient_pred, k)
    k_eff = min(k, len(donor_pred))
    assert nearest.shape == (len(recipient_pred), k_eff)
    for i, value in enumerate(recipient_pred):
        assert len(set(nearest[i])) == k_eff
        got = np.sort(np.abs(donor_pred[nearest[i]] - value))
        brute = np.sort(np.abs(donor_pred - value))[:k_eff]
        assert np.allclose(got, brute)


def _fold(seed: int) -> FoldData:
    rng = np.random.default_rng(seed)
    n_donor, n_recipient = 600, 80

    def features(n):
        return pd.DataFrame(
            {
                "x1": rng.normal(size=n),
                "x2": rng.normal(size=n),
                "x3": rng.integers(0, 2, n).astype(float),
            }
        )

    X_donor = features(n_donor)
    logit = 2.0 + 1.5 * X_donor["x1"].to_numpy()
    spends = rng.random(n_donor) < 1 / (1 + np.exp(-logit))
    moop = np.where(spends, rng.gamma(1.0, 4_000, n_donor), 0.0)
    cells_donor = _cells(rng, n_donor)
    cells_recipient = _cells(rng, n_recipient)
    return FoldData(
        X_donor=X_donor,
        moop_donor=moop,
        cells_donor=cells_donor,
        group_donor=cells_donor["group"].to_numpy(),
        X_recipient=features(n_recipient),
        cells_recipient=cells_recipient,
        group_recipient=cells_recipient["group"].to_numpy(),
    )


@settings(max_examples=8, deadline=None)
@given(seed=st.integers(0, 2**31))
def test_logit_family_rules(seed):
    fold = _fold(seed)
    out = logit_family(fold, 4, np.random.default_rng(seed))
    prob = participation_probability(
        fit_participation(fold.X_donor, fold.moop_donor), fold.X_recipient
    )
    cells = assign_cells(fold.cells_donor, fold.moop_donor, fold.cells_recipient)
    means = cells.means()
    below = prob < PARTICIPATION_CUTOFF
    for method, draws in out.items():
        assert draws.shape == (4, len(prob))
        assert (draws >= 0).all(), method
    assert np.allclose(out["logit_cell_mean"][0], np.where(below, 0, means))
    assert np.allclose(out["logit_expected_value"][0], prob * means)
    assert (out["logit_reg_mean"][:, below] == 0).all()
    assert (out["logit_group_mean"][:, below] == 0).all()
    assert (out["logit_threshold_hotdeck"][:, below] == 0).all()
    for i, pool in enumerate(cells.pools):
        for method in ("logit_draw_hotdeck", "logit_threshold_hotdeck"):
            positive = out[method][:, i][out[method][:, i] > 0]
            assert np.isin(positive, pool).all()
        drawn = out["logit_draw_cell_mean"][:, i]
        assert np.all((drawn == 0) | np.isclose(drawn, means[i]))


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(0, 2**31))
def test_evaluate_invariants(seed):
    persons, units = synthetic_spm(seed, n_units=80)
    bump = np.random.default_rng(seed + 1).uniform(0, 10_000, len(units))
    imputations = {
        "no_moop": np.zeros((1, len(units))),
        "higher": (units["moop"].to_numpy() + bump)[None, :],
        "same": np.tile(units["moop"].to_numpy(), (3, 1)),
    }
    results = evaluate(persons, units, imputations)
    groups = results["groups"]
    methods = results["methods"]
    nonempty = np.array(results["group_person_counts"]) > 0
    assert np.allclose(np.array(methods["same"]["diff"])[nonempty], 0)
    assert methods["same"]["newly_poor_share"] == 0
    assert methods["same"]["newly_nonpoor_share"] == 0
    assert (np.array(methods["no_moop"]["diff"])[nonempty] <= 1e-12).all()
    assert (np.array(methods["higher"]["diff"])[nonempty] >= -1e-12).all()
    band = groups.index(["pre_moop_ratio", RATIO_BANDS[0][2]])
    if nonempty[band]:
        for method in methods.values():
            assert method["rate"][band] == pytest.approx(1.0)


@SETTINGS
@given(
    ratios=st.lists(
        st.one_of(
            st.floats(-5, 20, allow_nan=False),
            st.sampled_from([1.0, 1.25, 1.5, 2.0, 3.0]),
        ),
        min_size=1,
        max_size=50,
    )
)
def test_ratio_band_index_matches_band_masks(ratios):
    ratios = np.array(ratios)
    index = ratio_band_index(ratios)
    for b, (low, high, _) in enumerate(RATIO_BANDS):
        mask = (ratios >= low) & (ratios < high)
        assert np.array_equal(index == b, mask)
