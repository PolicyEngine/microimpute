# Microimpute

Microimpute is a Python package for imputing variables from one survey dataset onto another. It wraps five imputation methods behind a common interface so you can benchmark them on your data and pick the one that works best: impute one or multiple variables with any of the available methods, and compare their performance using quantile loss and log loss.

The package currently supports:
- Hot Deck Matching
- Ordinary Least Squares (OLS) Linear Regression
- Quantile Regression Forests (QRF)
- Quantile Regression
- Mixture Density Networks (MDN)

This is a work in progress and may evolve over time with new methods and features.

## Microimputation dashboard

Users can visualize imputation and benchmarking results at https://microimpute-dashboard.vercel.app/.

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
