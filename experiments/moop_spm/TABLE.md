# MOOP imputation and SPM poverty: tables

Source: CPS ASEC 2026 public-use file (calendar-year 2025 income), 134,729 people in 55,401 SPM units and 53,344 households. We impute each unit's MOOP from models fit on the other 4 of 5 folds of whole households, and stochastic methods average 10 draws. Rates use the ASEC person weight. Standard errors in parentheses use the 160 replicate weights and treat the fitted imputation models as fixed.

## Table 1. SPM poverty rate with reported MOOP (%), and the change under each imputation (percentage points, SE)

| Group | People (n) | Reported | (a) Logit + cell mean | (a) Logit + OLS mean | (b) Logit draw + donor | PMM | (c) QRF | (c) ZI-QRF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All people | 134,729 | 13.1 | +0.30 (0.08) | +0.34 (0.08) | +0.39 (0.07) | +0.32 (0.07) | -0.07 (0.07) | -0.05 (0.07) |
| **Age** | | | | | | | | |
| Under 18 | 31,444 | 13.4 | +0.17 (0.13) | +0.26 (0.14) | +0.42 (0.11) | +0.41 (0.11) | +0.00 (0.11) | +0.06 (0.11) |
| 18 to 64 | 77,083 | 12.3 | +0.13 (0.09) | +0.11 (0.09) | +0.26 (0.07) | +0.19 (0.07) | -0.08 (0.07) | -0.06 (0.07) |
| 65 and older | 26,202 | 15.4 | +0.98 (0.17) | +1.15 (0.16) | +0.75 (0.13) | +0.60 (0.15) | -0.10 (0.14) | -0.13 (0.13) |
| **Disability** | | | | | | | | |
| With a disability | 14,477 | 22.7 | +1.20 (0.22) | +1.40 (0.21) | +0.59 (0.18) | +0.41 (0.20) | -0.21 (0.18) | -0.22 (0.18) |
| No disability | 94,282 | 11.7 | +0.21 (0.09) | +0.20 (0.09) | +0.35 (0.07) | +0.28 (0.07) | -0.07 (0.07) | -0.04 (0.07) |
| Not asked (mostly under 15) | 25,970 | 13.2 | +0.15 (0.14) | +0.28 (0.15) | +0.42 (0.11) | +0.41 (0.12) | +0.02 (0.12) | +0.05 (0.12) |
| **Current health insurance** | | | | | | | | |
| Medicare | 27,328 | 16.6 | +1.20 (0.17) | +1.23 (0.16) | +0.85 (0.14) | +0.59 (0.15) | -0.08 (0.15) | -0.11 (0.13) |
| Employer | 66,520 | 3.9 | -0.21 (0.08) | -0.12 (0.08) | +0.14 (0.07) | +0.23 (0.07) | -0.02 (0.07) | +0.02 (0.08) |
| Direct purchase | 8,341 | 18.3 | +0.06 (0.39) | +2.28 (0.48) | +0.20 (0.32) | +0.71 (0.36) | -0.22 (0.36) | -0.04 (0.36) |
| Medicaid or other means-tested | 18,134 | 32.1 | +1.18 (0.23) | +0.32 (0.22) | +0.99 (0.18) | +0.33 (0.20) | +0.01 (0.17) | -0.03 (0.18) |
| TRICARE, VA or CHAMPVA | 3,174 | 7.2 | -0.17 (0.21) | -0.59 (0.36) | -0.29 (0.29) | -0.15 (0.29) | -0.01 (0.26) | -0.09 (0.30) |
| Uninsured | 11,232 | 25.8 | +0.07 (0.27) | -0.22 (0.28) | +0.06 (0.23) | -0.05 (0.22) | -0.33 (0.23) | -0.28 (0.24) |
| **Resources before MOOP / threshold** | | | | | | | | |
| Under 1.0 | 13,803 | 100.0 | +0.00 (0.00) | +0.00 (0.00) | +0.00 (0.00) | +0.00 (0.00) | +0.00 (0.00) | +0.00 (0.00) |
| 1.0 to 1.25 | 8,441 | 24.9 | +14.69 (1.01) | +14.37 (1.05) | +4.88 (0.74) | +4.19 (0.75) | +0.55 (0.79) | +0.61 (0.81) |
| 1.25 to 1.5 | 9,233 | 5.8 | -4.75 (0.50) | -3.92 (0.49) | +0.50 (0.49) | +0.23 (0.51) | -0.76 (0.48) | -0.57 (0.50) |
| 1.5 to 2.0 | 17,495 | 1.6 | -1.56 (0.15) | -1.54 (0.15) | +0.19 (0.17) | +0.17 (0.18) | -0.29 (0.17) | -0.26 (0.16) |
| 2.0 to 3.0 | 30,010 | 0.3 | -0.32 (0.07) | -0.32 (0.07) | +0.05 (0.07) | +0.03 (0.08) | -0.10 (0.07) | -0.06 (0.07) |
| 3.0 or more | 55,747 | 0.1 | -0.05 (0.02) | -0.05 (0.02) | +0.03 (0.02) | +0.02 (0.02) | +0.02 (0.02) | +0.00 (0.02) |

Every method imputes MOOP of zero or more, so people whose unit's resources before MOOP fall below its threshold stay poor under every method, and the 'Under 1.0' row cannot change.

## Table 2. SPM-unit MOOP distribution (weighted by SPM unit) and gross flows

| Statistic | Reported | (a) Logit + cell mean | (a) Logit + OLS mean | (b) Logit draw + donor | PMM | (c) QRF | (c) ZI-QRF |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mean | $6,011 | $6,156 | $6,133 | $5,984 | $5,922 | $5,797 | $5,791 |
| Median | $3,700 | $6,132 | $5,891 | $3,678 | $3,624 | $3,623 | $3,610 |
| Share with zero MOOP | 6.4% | 0.5% | 4.3% | 6.5% | 6.2% | 6.2% | 6.1% |
| 75th percentile | $8,160 | $8,625 | $8,983 | $8,154 | $8,091 | $7,949 | $7,891 |
| 90th percentile | $14,300 | $11,436 | $11,507 | $14,276 | $14,158 | $13,820 | $13,765 |
| 99th percentile | $33,640 | $15,585 | $15,769 | $33,147 | $32,666 | $31,356 | $31,424 |
| Total ($ billions) | 859 | 880 | 876 | 855 | 846 | 828 | 828 |
| Mean, resources before MOOP 1.0 to 1.25 x threshold | $2,385 | $3,189 | $3,108 | $2,949 | $2,799 | $2,352 | $2,372 |
| Mean, resources before MOOP 1.25 to 1.5 x threshold | $3,381 | $3,679 | $3,667 | $3,446 | $3,452 | $3,282 | $3,272 |
| Mean, resources before MOOP 1.5 to 2.0 x threshold | $4,510 | $4,788 | $4,612 | $4,546 | $4,527 | $4,335 | $4,389 |
| Mean, resources before MOOP 2.0 to 3.0 x threshold | $6,086 | $6,116 | $5,872 | $5,953 | $5,783 | $5,809 | $5,843 |
| People newly poor (%) | 0.0 | 1.37 | 1.40 | 1.74 | 1.63 | 1.33 | 1.37 |
| People newly not poor (%) | 0.0 | 1.07 | 1.06 | 1.36 | 1.31 | 1.40 | 1.41 |
| Units where Census imputes no MOOP or Medicare premium: change in rate (pp) | 0.00 | +0.36 (0.11) | +0.26 (0.11) | +0.52 (0.09) | +0.48 (0.10) | +0.01 (0.09) | +0.07 (0.10) |

## Table 3. How the participation and amount rules of the logit approach move the rate (all people)

| Participation rule | Amount rule | Rate (%) | Change (pp) | SE |
|---|---|---:|---:|---:|
| p >= 0.5 | Cell mean | 13.42 | +0.30 | 0.08 |
| p >= 0.5 | Random donor in cell | 13.70 | +0.59 | 0.07 |
| Bernoulli(p) | Cell mean | 13.21 | +0.10 | 0.08 |
| Bernoulli(p) | Random donor in cell | 13.50 | +0.39 | 0.07 |
| None (all units) | p x cell mean | 13.19 | +0.07 | 0.08 |
| p >= 0.5 | Coverage-group mean (5 cells) | 15.18 | +2.07 | 0.10 |
