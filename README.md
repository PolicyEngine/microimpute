# Microimpute

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

An interactive dashboard for exploring imputation results is available at https://microimpute-dashboard.vercel.app/. It supports file upload, URL loading, direct GitHub artifact integration, and sample data.

## Installation

```bash
pip install microimpute
```

For image export (PNG/JPG):

```bash
pip install microimpute[images]
```

## Contributing

Pull requests are welcome. If you find a bug or have a feature idea, open an issue or submit a PR.
