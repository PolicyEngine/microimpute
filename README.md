# Microimpute

> PolicyEngine has archived microimpute. [Microcosm](https://github.com/PolicyEngine/microcosm) builds PolicyEngine's microdata and fits its imputation models with the `microcosm-fit` package. [PyPI](https://pypi.org/project/microimpute/) keeps every release, so `pip install microimpute` still works. The [open issues](https://github.com/PolicyEngine/microimpute/issues) include correctness bugs that no release fixes, such as [the default QRF's miscalibrated quantiles](https://github.com/PolicyEngine/microimpute/issues/205).

Microimpute is a Python package for imputing variables from one survey dataset onto another. It wraps five imputation methods behind a common interface so you can benchmark them on your data and pick the one that works best, rather than defaulting to a single approach.

## Methods

- **Statistical Matching**: distance-based matching to find similar donor observations
- **Ordinary Least Squares (OLS)**: linear regression imputation
- **Quantile Regression**: models conditional quantiles instead of the conditional mean
- **Quantile Regression Forests (QRF)**: non-parametric, tree-based quantile estimation
- **Mixture Density Networks (MDN)**: neural network with a Gaussian mixture output

## Autoimpute

The `autoimpute` function tunes hyperparameters, runs cross-validation across all five methods, and selects the best performer based on quantile loss (for numerical targets) or log loss (for categorical targets). It handles numerical, categorical, and boolean variables.

## API

All models follow a `fit()` / `predict()` interface. The package supports sample weights to account for survey design, and validates inputs automatically. Adding a custom imputation method is straightforward since new models just need to implement the same interface.

## Documentation and paper

- [Documentation](https://policyengine.github.io/microimpute/) with examples and interactive notebooks
- [Paper](https://github.com/PolicyEngine/microimpute/blob/main/paper/main.pdf) presenting microimpute and demonstrating it for SCF-to-CPS net worth imputation

## Dashboard

Install Node.js, clone the [microimpute repository](https://github.com/PolicyEngine/microimpute) and run `make dashboard` from its root to start the interactive dashboard in `microimputation-dashboard`. Open http://localhost:3000 and upload a CSV file, load one from a URL or try the sample data to explore imputation results.

## Installation

```bash
pip install microimpute
```

For image export (PNG/JPG):

```bash
pip install microimpute[images]
```

## License

PolicyEngine releases the code in this repository under the [MIT License](https://github.com/PolicyEngine/microimpute/blob/main/LICENSE), and its original text and figures under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) with attribution to PolicyEngine. Third-party data and materials keep their own terms. The optional `matching` extra installs rpy2, which its authors license under GPL-2.0-or-later. The paper notebook `paper/imputing-from-scf-to-cps.ipynb` imports policyengine-us and policyengine-core, which carry AGPL-3.0.
