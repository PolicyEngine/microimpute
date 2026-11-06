# MOOP imputation and SPM poverty

How much does the method used to impute medical out-of-pocket spending
(MOOP) change Supplemental Poverty Measure (SPM) poverty? This experiment
answers that on a file that carries MOOP: the 2026 CPS ASEC public-use
file, which records SPM-unit MOOP for calendar year 2025.

The American Community Survey (ACS) does not ask about medical spending,
so researchers who build an SPM on the ACS impute MOOP. Census
researchers used predicted mean matching by insurance status
([Fox, Pacas and Glassman, 2017](https://www.census.gov/content/dam/Census/library/working-papers/2017/demo/Fox-ACSSPM2014.pdf)).
This test imputes MOOP within the CPS ASEC and checks each method against
the file's own MOOP.

`NOTE.md` summarizes the findings and `TABLE.md` has the full tables.
`render.py` writes both from `results/results.json`.

## Data

- Source: Census CPS ASEC 2026 public-use CSV release
  (`asecpub26csv.zip`, SHA-256 pinned in `cps_asec.py`), with the 160
  person replicate weights it ships.
- Imputation unit: the SPM unit. Target: `SPM_MEDXPNS`, which the
  dictionary labels "MOOP and Medicare Part B subsidy". In the file it
  equals the unit sum of person `MOOP` plus the unit sum of `PEMCPREM`
  (edited Medicare premium), exactly.
- `check_spm_identity` confirms, on every record, that
  `SPM_RESOURCES` equals cash income plus SNAP, housing, school lunch,
  energy and WIC subsidies, less taxes, FICA, capped work and child care
  expenses, MOOP and child support paid; and that `SPM_POOR` equals
  `SPM_RESOURCES < SPM_POVTHRESHOLD`. So an imputed MOOP `m'` changes a
  unit's resources to `SPM_RESOURCES + SPM_MEDXPNS - m'`.
- Baseline: with reported MOOP and the ASEC person weight, the file
  reproduces Census's published 2025 SPM rates (P60-290 Table 5) and
  their margins of error, for all people and each age group, to within
  0.01 percentage point. Its MOOP effects (the rise in the rate from
  subtracting MOOP) fall 0.02 to 0.03 points short of Census's.
  `results.json` records the published figures beside the computed ones.
- Census imputes MOOP for 43% of the file's people (`I_MOOP` of 1, 2 or
  3), so the robustness check below repeats the comparison on units with
  no imputed MOOP.

## Predictors

Only variables an ACS-style recipient file carries, built at the SPM-unit
level (`cps_asec.design_matrix`, 73 columns):

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

All methods fit on donor units and impute held-out recipient units
(`methods.py`). We fit every method without survey weights.

| Name | Participation | Amount |
|---|---|---|
| (a) `logit_cell_mean` | logistic P(MOOP > 0) >= 0.5 | mean of donor spenders in the unit's cell |
| (a) `logit_reg_mean` | same | OLS prediction among donor spenders (floored at 0) |
| (b) `logit_draw_hotdeck` | Bernoulli draw from the logistic P | random donor spender in the cell |
| `pmm` | implicit | predictive mean matching on log(1 + MOOP) within five coverage groups, random one of 5 nearest donors |
| (c) `qrf` | implicit | microimpute `QRF`, one random quantile per unit |
| (c) `zi_qrf` | microimpute gate | microimpute `ZeroInflatedImputer` around `QRF` |

Cells cross the unit's coverage group (Medicare, employer, direct
purchase, public, uninsured), the head's age band, cash income relative
to the threshold (under 1, 1-2, 2-4, 4+) and unit size (1, 2, 3+). A cell
needs 30 donor spenders; otherwise the recipient falls back to coarser
cells.

PMM follows the approach Census researchers described for their ACS-SPM
research in 2017 (predicted mean matching by insurance group) without
replicating it.

Decomposition variants cross the participation rule (0.5 cutoff or a
draw) with the amount rule (cell mean or a donor draw). Two more assign
`p x cell mean` to every unit, or give each unit at or above the 0.5
cutoff its coverage group's mean (no age, income or size cells).

## Design

- 5 folds of whole households (`run.assign_folds`): every SPM unit gets
  an imputation from models fit on the other 4 folds.
- Stochastic methods take 10 draws; poverty rates average over draws.
- Rates use the ASEC person weight (`MARSUPWT / 100`). Standard errors
  use the 160 replicate weights,
  `SE^2 = (4 / 160) * sum_r (theta_r - theta_0)^2`, and treat the fitted
  imputation models as fixed.
- Breakdowns: age, disability, current coverage, and the unit's resources
  before MOOP relative to its threshold.
- Robustness: the same comparison among people in SPM units where
  Census imputed no member's MOOP (`I_MOOP` of 0 or -1).

## Running

From the repository root, with microimpute installed:

```bash
python -m experiments.moop_spm.run --data-dir ~/data/cps_asec_2026
python -m experiments.moop_spm.render
```

The first command downloads the 139 MB Census zip, checks its SHA-256,
and caches per-fold imputations in `results/cache/` (git-ignored).
`results.json` records the run time of each method. The tests in
`tests/test_moop_spm_experiment.py` check the experiment's invariants on
synthetic data.
