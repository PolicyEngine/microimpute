# MOOP imputation and SPM poverty

How much does the choice of method for imputing medical out-of-pocket
spending (MOOP) change Supplemental Poverty Measure (SPM) poverty? We
measure it on the public-use file of the 2026 Current Population Survey
Annual Social and Economic Supplement (CPS ASEC), which records SPM-unit
MOOP for calendar year 2025.

The American Community Survey (ACS) does not ask about medical spending,
so researchers who build an SPM on the ACS impute MOOP. For 2014,
Census researchers imputed it with predictive mean matching by insurance
status ([Fox, Pacas and Glassman, 2017](https://www.census.gov/content/dam/Census/library/working-papers/2017/demo/Fox-ACSSPM2014.pdf)).
We impute MOOP within the CPS ASEC and check each method against the
file's own MOOP.

We summarize the findings in `NOTE.md` and give the full tables in
`TABLE.md`. `render.py` writes both from `results/results.json`.

## Data

- We use Census's CPS ASEC 2026 public-use CSV release
  (`asecpub26csv.zip`; `cps_asec.py` pins its SHA-256) and the 160 person
  replicate weights it ships.
- We impute `SPM_MEDXPNS` for each SPM unit. Census labels the variable
  "MOOP and Medicare Part B subsidy" in its data dictionary. On every
  record, the variable equals the unit sum of person `MOOP` plus the unit
  sum of `PEMCPREM` (edited Medicare premium).
- `check_spm_identity` confirms that, on every record, `SPM_RESOURCES`
  equals cash income plus SNAP, housing, school lunch, energy and WIC
  subsidies, less taxes, FICA, capped work and child care expenses,
  `SPM_MEDXPNS` and child support paid. So an imputed MOOP `m'` changes
  a unit's resources to `SPM_RESOURCES + SPM_MEDXPNS - m'`. It also
  confirms that `SPM_POOR` equals `SPM_RESOURCES < SPM_POVTHRESHOLD`, so
  we recompute poverty from the new resources and the unchanged
  threshold.
- With reported MOOP and the ASEC person weight, we reproduce Census's
  published 2025 SPM rates (P60-290 Table 5) and their margins of error,
  for all people and each age group, to within 0.01 percentage point.
  Subtracting MOOP raises our rates by 0.02 to 0.03 points less than it
  raises Census's. `run.py` writes the published figures beside ours
  into `results.json`.
- Census imputes MOOP for 43% of the file's people (`I_MOOP` of 1, 2 or
  3) and the Medicare premium for 13% (`I_MCPREM` of 2).

## Predictors

For each SPM unit, we build 73 predictor columns
(`cps_asec.design_matrix`) from information an ACS-style recipient file
carries:

- head's age and sex; counts of members aged 0-17, 18-44, 45-64, 65-74,
  75+; adult women;
- members with a disability (`PRDISFLG`);
- members with each type of current coverage (`NOW_*`: employer, direct
  purchase, Medicare, Medicaid or other means-tested, TRICARE/VA/CHAMPVA,
  uninsured);
- unit cash income (signed log) and cash income relative to the SPM
  threshold;
- SPM family type, tenure and state (indicators).

## Methods

We fit every method on donor units and impute held-out recipient units
(`methods.py`). Only the weighted Microcosm fit uses survey weights.

| Name | Participation | Amount |
|---|---|---|
| (a) `logit_cell_mean` | logistic P(MOOP > 0) >= 0.5 | mean of donor spenders in the unit's cell |
| (a) `logit_reg_mean` | same | max(0, OLS prediction among donor spenders) |
| (b) `logit_draw_hotdeck` | Bernoulli draw from the logistic P | random donor spender in the cell |
| `pmm` | implicit | predictive mean matching on log(1 + MOOP) within five coverage groups, random one of 5 nearest donors |
| (c) `qrf` | implicit | microimpute `QRF`, one random quantile per unit |
| (c) `zi_qrf` | microimpute gate | microimpute `ZeroInflatedImputer` around `QRF` |
| (c) `microcosm_qrf` | Microcosm gate, a gradient-boosted classifier | Microcosm `RegimeGatedQRF` forest on donor spenders, one random quantile per unit; gate and forest use the SPM unit weights |
| (c) `microcosm_qrf_unweighted` | same, without weights | same, without weights |

Microcosm (`PolicyEngine/microcosm`) builds PolicyEngine's microdata,
and its `microcosm-fit` package imputes with `RegimeGatedQRF`. We fit it
with its defaults (100 trees, `max_samples_leaf=None` and no
cross-fitting of the gate), once with each donor unit's SPM weight
(`SPM_WEIGHT / 100`) and once without weights. With the weights,
Microcosm passes them to the gate as sample weights and resamples
spenders in proportion to them before it grows the forest. Both fits
share a seed, so their draws use the same random numbers. Beyond the
weights, Microcosm's model differs from microimpute's `QRF` in three
ways:

- Its gate draws whether a unit spends. microimpute's `QRF` has no
  gate, and `ZeroInflatedImputer` adds one of its own.
- It keeps every training sample in each leaf of its MOOP forest, where
  microimpute keeps quantile-forest's default of at most one sample per
  leaf. Microcosm caps a forest's leaves at 4,096 samples only when the
  donors that forest trains on hold 2 to 32 distinct values. Its MOOP
  forest trains on the donor spenders, whose amounts take 7,830 to 7,880
  distinct values in each fold.
- It reads each unit's draw at that unit's own quantile, interpolating
  a 203-point grid whose ends lie 0.000001 from 0 and 1. microimpute
  rounds each unit's quantile q to the nearest hundredth, k/100, and
  reads the forest's (k + 1)/102 quantile, so its draws come from 101
  points that run from 1/102 to 101/102 (q = 0.01 reads 2/102).

We form cells from the unit's coverage group (Medicare, employer, direct
purchase, public, uninsured), the head's age band, cash income relative
to the threshold (under 1, 1-2, 2-4, 4+) and unit size (1, 2, 3+). Each
recipient uses the finest cell with at least 30 donor spenders. Where its
full cell has fewer, we drop unit size, then income, then age and then
coverage group.

For PMM, we match within five coverage groups, where the Census
researchers used eight groups by insurance status.

To decompose method (a), we cross the participation rule (0.5 cutoff or
a draw) with the amount rule (cell mean or a donor draw). In two more
variants, we give every unit `p x cell mean`, or give each unit at or
above the 0.5 cutoff its coverage group's mean (no age, income or size
cells).

## Design

- We assign whole households to 5 folds (`run.assign_folds`) and impute
  each SPM unit from models we fit on the other 4.
- We take 10 draws for each stochastic method and average poverty rates
  over them.
- We weight rates with the ASEC person weight (`MARSUPWT / 100`) and
  compute standard errors from the 160 replicate weights,
  `SE^2 = (4 / 160) * sum_r (theta_r - theta_0)^2`, treating the fitted
  imputation models as fixed.
- We break the results down by age, disability, current coverage, and
  the unit's resources before MOOP relative to its threshold.
- We repeat the comparison among people in SPM units where Census
  imputes no member's MOOP or Medicare premium (`I_MOOP` of 0 or -1, and
  `I_MCPREM` of 0 or -1).

## Running

Install microimpute with its `dev` extra, and Microcosm's `graph`,
`frame` and `fit` packages at the commit we used. Microcosm needs Python
3.13 or later and installs from git:

```bash
uv pip install -e ".[dev]"
uv pip install \
  "microcosm-graph @ git+https://github.com/PolicyEngine/microcosm@7080ba8a3ff4ba498575ec1a624a9064a6f619c8#subdirectory=packages/microcosm-graph" \
  "microcosm-frame @ git+https://github.com/PolicyEngine/microcosm@7080ba8a3ff4ba498575ec1a624a9064a6f619c8#subdirectory=packages/microcosm-frame" \
  "microcosm-fit @ git+https://github.com/PolicyEngine/microcosm@7080ba8a3ff4ba498575ec1a624a9064a6f619c8#subdirectory=packages/microcosm-fit"
```

Then run from the repository root:

```bash
python -m experiments.moop_spm.run --data-dir ~/data/cps_asec_2026
python -m experiments.moop_spm.render
```

The first command downloads the 139 MB Census zip, checks its SHA-256,
caches each fold's imputations in `results/cache/`, which git ignores,
and writes `results/results.json`. `--n-jobs` (default 4) caps the
threads of microimpute's `QRF` and of each Microcosm fit; a test checks
that Microcosm's draws do not depend on it.
`run.py` refits every fold when any of these change:

- the settings or the experiment's source files;
- any source file of microimpute or of Microcosm's `fit`, `frame` and
  `graph` packages;
- the version of Python, numpy, pandas, scipy, scikit-learn,
  quantile-forest, microimpute or Microcosm, or the commit Microcosm came
  from;
- the model inputs or their row keys.

`run.py` also records each method's run time, and the commit each
Microcosm package came from, in `results.json`. The tests in
`tests/test_moop_spm_experiment.py` check the experiment's invariants on
synthetic data; the Microcosm tests skip on a machine without Microcosm.
