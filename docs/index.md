# Microimpute

> PolicyEngine has archived microimpute. [Microcosm](https://github.com/PolicyEngine/microcosm) builds PolicyEngine's microdata and fits its imputation models with the `microcosm-fit` package. [PyPI](https://pypi.org/project/microimpute/) keeps the package's releases, so `pip install microimpute` still works. The [open issues](https://github.com/PolicyEngine/microimpute/issues) include correctness bugs that no release fixes, such as [the default QRF's miscalibrated quantiles](https://github.com/PolicyEngine/microimpute/issues/205).

Microimpute is a Python package for imputing variables from one survey dataset onto another. It wraps five imputation methods behind a common interface so you can benchmark them on your data and pick the one that works best: impute one or multiple variables with any of the available methods, and compare their performance using quantile loss and log loss.

The package supports:
- Hot Deck Matching
- Ordinary Least Squares (OLS) Linear Regression
- Quantile Regression Forests (QRF)
- Quantile Regression
- Mixture Density Networks (MDN)

## Microimputation dashboard

To visualize imputation and benchmarking results, install Node.js, run `make dashboard` from the root of the [microimpute repository](https://github.com/PolicyEngine/microimpute), and open http://localhost:3000.

To use the dashboard, CSV files must contain the following columns in this exact order:
- `type`: Type of metric (e.g., "benchmark_loss", "distribution_distance", "predictor_correlation")
- `method`: Imputation method name (e.g., "QRF", "OLS", "QuantReg", "Matching", "MDN")
- `variable`: Variable being imputed or analyzed
- `quantile`: Quantile level (numeric value, "mean", or "N/A")
- `metric_name`: Name of the metric (e.g., "quantile_loss", "log_loss")
- `metric_value`: Numeric value of the metric
- `split`: Data split indicator (e.g., "train", "test", "full")
- `additional_info`: JSON-formatted string with additional metadata

The `format_csv()` function from `microimpute.utils` formats imputation and benchmarking results into the correct structure for the dashboard. It accepts outputs from various analysis functions (autoimpute results, comparison metrics, distribution comparisons) and returns a properly formatted DataFrame.
