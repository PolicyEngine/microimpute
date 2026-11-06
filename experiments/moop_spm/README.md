# MOOP imputation and SPM poverty

How much does the choice of method for imputing medical out-of-pocket
spending (MOOP) change Supplemental Poverty Measure (SPM) poverty? We
measure it on the 2026 CPS ASEC public-use file, which records SPM-unit
MOOP for calendar year 2025.

The American Community Survey (ACS) does not ask about medical spending,
so researchers who build an SPM on the ACS impute MOOP. Census
researchers used predictive mean matching by insurance status
([Fox, Pacas and Glassman, 2017](https://www.census.gov/content/dam/Census/library/working-papers/2017/demo/Fox-ACSSPM2014.pdf)).
We impute MOOP within the CPS ASEC and check each method against the
file's own MOOP.

`NOTE.md` summarizes the findings and `TABLE.md` has the full tables.
`render.py` writes both from `results/results.json`.

## Data

- Source: Census CPS ASEC 2026 public-use CSV release
  (`asecpub26csv.zip`; `cps_asec.py` pins its SHA-256), with the 160
  person replicate weights it ships.
- Imputation unit: the SPM unit. Target: `SPM_MEDXPNS`, which the
  dictionary labels "MOOP and Medicare Part B subsidy". In the file it
  equals the unit sum of person `MOOP` plus the unit sum of `PEMCPREM`
  (edited Medicare premium) on every record.
- `check_spm_identity` confirms, on every record, that
  `SPM_RESOURCES` equals cash income plus SNAP, housing, school lunch,
  energy and WIC subsidies, less taxes, FICA, capped work and child care
  expenses, MOOP and child support paid; and that `SPM_POOR` equals
  `SPM_RESOURCES < SPM_POVTHRESHOLD`. So an imputed MOOP `m'` changes a
  unit's resources to `SPM_RESOURCES + SPM_MEDXPNS - m'`.
- Baseline: with reported MOOP and the ASEC person weight, we reproduce
  Census's published 2025 SPM rates (P60-290 Table 5) and their margins
  of error, for all people and each age group, to within 0.01 percentage
  point. Our MOOP effects (the rise in the rate from subtracting MOOP)
  fall 0.02 to 0.03 points short of Census's.
  `results.json` records the published figures beside the computed ones.
- Census imputes MOOP for 43% of the file's people (`I_MOOP` of 1, 2 or
  3) and the Medicare premium for 13% (`I_MCPREM` of 2). So we repeat the
  comparison on units where Census imputes neither for any member.

## Predictors

We limit every method to predictors an ACS-style recipient file carries,
built at the SPM-unit level (`cps_asec.design_matrix`, 73 columns):

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

We fit every method on donor units, without survey weights, and impute
held-out recipient units (`methods.py`).

| Name | Participation | Amount |
|---|---|---|
| (a) `logit_cell_mean` | logistic P(MOOP > 0) >= 0.5 | mean of donor spenders in the unit's cell |
| (a) `logit_reg_mean` | same | OLS prediction among donor spenders (floored at 0) |
| (b) `logit_draw_hotdeck` | Bernoulli draw from the logistic P | random donor spender in the cell |
| `pmm` | implicit | predictive mean matching on log(1 + MOOP) within five coverage groups, random one of 5 nearest donors |
| (c) `qrf` | implicit | microimpute `QRF`, one random quantile per unit |
| (c) `zi_qrf` | microimpute gate | microimpute `ZeroInflatedImputer` around `QRF` |

We form cells from the unit's coverage group (Medicare, employer, direct
purchase, public, uninsured), the head's age band, cash income relative
to the threshold (under 1, 1-2, 2-4, 4+) and unit size (1, 2, 3+). A cell
needs 30 donor spenders; otherwise the recipient falls back to coarser
cells.

For PMM, we follow the approach Census researchers described for their
ACS-SPM research in 2017 (predictive mean matching within eight groups
by insurance status) but match within five coverage groups.

To decompose method (a), we cross the participation rule (0.5 cutoff or
a draw) with the amount rule (cell mean or a donor draw). In two more
variants, we give every unit `p x cell mean`, or give each unit at or
above the 0.5 cutoff its coverage group's mean (no age, income or size
cells).

## Design

- We split whole households into 5 folds (`run.assign_folds`), and each
  SPM unit gets an imputation from models we fit on the other 4 folds.
- We take 10 draws for each stochastic method and average poverty rates
  over them.
- We weight rates with the ASEC person weight (`MARSUPWT / 100`) and
  compute standard errors from the 160 replicate weights,
  `SE^2 = (4 / 160) * sum_r (theta_r - theta_0)^2`, treating the fitted
  imputation models as fixed.
- Breakdowns: age, disability, current coverage, and the unit's resources
  before MOOP relative to its threshold.
- Robustness: the same comparison among people in SPM units where
  Census imputes no member's MOOP or Medicare premium (`I_MOOP` of 0 or
  -1, and `I_MCPREM` of 0 or -1).

## Running

From the repository root, with microimpute installed:

```bash
python -m experiments.moop_spm.run --data-dir ~/data/cps_asec_2026
python -m experiments.moop_spm.render
```

The first command downloads the 139 MB Census zip, checks its SHA-256,
and caches per-fold imputations in `results/cache/` (git-ignored).
`run.py` refits every fold when any of these change:

- the settings or the experiment's source files;
- any microimpute source file;
- the version of Python, numpy, pandas, scipy, scikit-learn,
  quantile-forest or microimpute;
- the model inputs or their row keys.

`results.json` records the run time of each method. The tests in
`tests/test_moop_spm_experiment.py` check the experiment's invariants on
synthetic data.
