# How the MOOP imputation method moves SPM poverty

The American Community Survey (ACS) does not ask about medical out-of-pocket spending (MOOP), so researchers who build a Supplemental Poverty Measure (SPM) on the ACS impute it. For 2014, Census researchers imputed it with predictive mean matching by insurance status, and subtracting their imputed MOOP raised the ACS SPM rate by 4.57 percentage points. The Current Population Survey's Annual Social and Economic Supplement (CPS ASEC) asks about MOOP, and for the same year, subtracting reported MOOP raised its SPM rate by 3.48 points. Yet the ACS MOOP total of $561 billion trailed the CPS ASEC's $567 billion by 1% ([Fox, Pacas and Glassman, 2017](https://www.census.gov/content/dam/Census/library/working-papers/2017/demo/Fox-ACSSPM2014.pdf), slides 10, 11 and 26). The surveys also differ in income and other resources, so we test the imputation alone. We impute MOOP within the CPS ASEC and check each method against the file's own MOOP.

## Data and design

We use the 134,729 people, 55,401 SPM units and 53,344 households in the 2026 CPS ASEC public-use file, which covers calendar-year 2025. We impute each unit's SPM medical expenses (`SPM_MEDXPNS`), its members' MOOP plus their Medicare premiums. Census subtracts them from SPM resources dollar for dollar, so we swap each unit's reported value for an imputed one and compare the new resources with an unchanged threshold. With reported MOOP and the ASEC person weight, we compute an SPM rate of 13.11% with a 90% margin of error of 0.31 points. Census published 13.11% and 0.31 in [P60-290](https://www.census.gov/library/publications/2026/demo/p60-290.html). Subtracting MOOP raises the rate by 2.25 points (SE 0.08), and Census reports 2.28.

We limit every method to predictors an ACS-style file carries: members' ages, sex, disability and current health coverage, the unit's cash income on its own and relative to its SPM threshold, family type, tenure and state. We assign whole households to 5 folds and impute each fold from models we fit on the rest, without survey weights. We average 10 draws for each stochastic method and compute standard errors from the 160 replicate weights, treating each fitted model as fixed.

- In method (a), we fit a logistic model of whether a unit has any MOOP. Each unit with a predicted probability of 0.5 or more gets the mean MOOP of donor spenders in its cell, and the rest get zero (logit plus cell means). In the OLS variant, predicted spenders get an OLS prediction among donor spenders instead. We form cells from coverage group, head's age, cash income relative to the threshold and unit size.
- In method (b), logit draws with donors, we draw participation at random from the same model's probabilities, and each spender takes a random donor spender's MOOP from its cell.
- In method (c), we fit quantile regression forests from PolicyEngine's microimpute package, alone (QRF) and inside its zero-inflated wrapper (ZI-QRF). We give each unit the value at a random quantile of its predicted distribution. In PolicyEngine's microimpute [paper](../../paper/main.pdf), María Juaristi found that quantile regression forests had the lowest average quantile loss of five methods for imputing Survey of Consumer Finances (SCF) net worth onto the CPS.
- We also apply predictive mean matching (PMM) within five coverage groups, where the Census researchers used eight. In each group, we regress log(1 + MOOP) on the predictors across all donors, including those with zero spending. Each recipient then takes the MOOP of a random one of the five donors whose predictions lie nearest its own.

## Results

|  | Reported | (a) Logit + cell mean | (a) Logit + OLS mean | (b) Logit draw + donor | PMM | (c) QRF | (c) ZI-QRF |
|---|---:|---:|---:|---:|---:|---:|---:|
| SPM rate, all people (%) | 13.11 | 13.42 | 13.46 | 13.50 | 13.43 | 13.04 | 13.07 |
| Change, all people (pp, SE) |  | +0.30 (0.08) | +0.34 (0.08) | +0.39 (0.07) | +0.32 (0.07) | -0.07 (0.07) | -0.05 (0.07) |
| Change, 65 and older (pp) |  | +0.98 | +1.15 | +0.75 | +0.60 | -0.10 | -0.13 |
| Change, with a disability (pp) |  | +1.20 | +1.40 | +0.59 | +0.41 | -0.21 | -0.22 |
| Change, Medicare (pp) |  | +1.20 | +1.23 | +0.85 | +0.59 | -0.08 | -0.11 |
| Change, resources before MOOP 1.0 to 1.25 x threshold (pp) |  | +14.69 | +14.37 | +4.88 | +4.19 | +0.55 | +0.61 |
| Mean MOOP, resources before MOOP 1.0 to 1.25 x threshold | $2,385 | $3,189 | $3,108 | $2,949 | $2,799 | $2,352 | $2,372 |
| Units with zero MOOP (%) | 6.4 | 0.5 | 4.3 | 6.5 | 6.2 | 6.2 | 6.1 |
| Median MOOP | $3,700 | $6,132 | $5,891 | $3,678 | $3,624 | $3,623 | $3,610 |
| 99th percentile MOOP | $33,640 | $15,585 | $15,769 | $33,147 | $32,666 | $31,356 | $31,424 |
| Total MOOP ($ billions) | 859 | 880 | 876 | 855 | 846 | 828 | 828 |
| People newly poor / newly not poor (%) |  | 1.37 / 1.07 | 1.40 / 1.06 | 1.74 / 1.36 | 1.63 / 1.31 | 1.33 / 1.40 | 1.37 / 1.41 |

We compute rates and changes over people and MOOP statistics over SPM units, and show standard errors in parentheses.

We estimate that logit plus cell means raises the SPM rate by 0.30 percentage points (SE 0.08), or 13% of MOOP's own 2.25-point effect. The OLS variant adds 0.34 points, logit draws with donors 0.39 and PMM 0.32. QRF changes the rate by -0.07 points (SE 0.07) and ZI-QRF by -0.05 (SE 0.07), though their draws fall 3.6% and 3.7% short of the reported MOOP total. For people 65 and older, the rate changes by +0.98 points under logit plus cell means, +0.60 under PMM and -0.10 under QRF. For people with a disability, it changes by +1.20, +0.41 and -0.21.

Logit draws with donors track the reported distribution, with 6.5% of units at zero against 6.4% and a 99th percentile of $33,147 against $33,640, yet raise the rate by 0.39 points. A unit falls below the line when its MOOP exceeds the margin by which its other resources clear the threshold, and reported MOOP rises with resources. Units whose resources before MOOP run 1.0 to 1.25 times their threshold spend $2,385 on average, and units at 2 to 3 times spend $6,086. In that first band above the line, cell means assign $3,189, donor draws $2,949, PMM $2,799 and QRF $2,352. The band's 24.9% poverty rate rises by 14.69, 4.88, 4.19 and 0.55 points under those four methods.

Under the 0.5 cutoff, 0.5% of units get no MOOP, against 6.4% in the file. Drawing participation instead lowers the national change from +0.30 to +0.10 points with cell means, and from +0.59 to +0.39 with donor draws (Table 3 in `TABLE.md`). When we draw participation and assign cell means, the rate still rises 11.50 points for people at 1.0 to 1.25 times the threshold and falls 4.80 points at 1.25 to 1.5. Cell means also cut the 99th percentile from $33,640 to $15,585. Under the same cutoff, coverage-group means alone, without the age, income and size cells, raise the national rate by 2.07 points, 92% of MOOP's own effect.

Every method moves people across the line in both directions. QRF moves 1.33% of people into poverty and 1.40% out of it, and logit plus cell means moves 1.37% in and 1.07% out. Census researchers turned to the ACS for state and substate SPM estimates, where these flows need not offset.

Census imputes MOOP for 43% of the people in the file and the Medicare premium for 13%, so part of our benchmark comes from its imputation. In the 44% of SPM units where Census imputes neither for any member, methods (a), (b) and PMM raise the rate by 0.26 to 0.52 points. QRF and ZI-QRF change it by +0.01 and +0.07 points (SE 0.09 and 0.10). Because our donors and recipients share one survey's questions, income measures and population, we do not test the differences between surveys that an ACS imputation also faces.

For each method's change by age, disability, coverage type and resources relative to the threshold, see Table 1 in `TABLE.md`; for the code, see `README.md`.
