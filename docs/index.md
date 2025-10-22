# Microimpute

Microimpute is a powerful framework that enables variable imputation through a variety of statistical methods. By providing a consistent interface across different imputation techniques, it allows researchers and data scientists to easily compare and benchmark different approaches using quantile loss and log loss calculations to determine the method provding most accurate results. Thus, Microimpute provides two main uses: imputing one or multiple variables with one of the methods available, and comparing and benchmarking different methods to inform a method's choice.

The framework currently supports the following imputation methods:
- Statistical Matching
- Ordinary Least Squares Linear Regression
- Quantile Random Forests
- Quantile Regression

This is a work in progress that may evolve over time, including new statistical imputation methods and features. 

## Microimputation dashboard

Users can visualize imputation and benchmarking results at https://microimpute-dashboard.vercel.app/.

To use the dashboard for visualization, CSV files must contain the following columns in this exact order:
- `type`: Type of metric (e.g., "benchmark_loss", "distribution_distance", "predictor_correlation")
- `method`: Imputation method name (e.g., "QRF", "OLS", "QuantReg", "Matching")
- `variable`: Variable being imputed or analyzed
- `quantile`: Quantile level (numeric value, "mean", or "N/A")
- `metric_name`: Name of the metric (e.g., "quantile_loss", "log_loss")
- `metric_value`: Numeric value of the metric
- `split`: Data split indicator (e.g., "train", "test", "full")
- `additional_info`: JSON-formatted string with additional metadata

Users can use the `format_csv()` function from `microimpute.utils` to automatically format imputation and benchmarking results into the correct structure for dashboard visualization. This function accepts outputs from various analysis functions (autoimpute results, comparison metrics, distribution comparisons, etc.) and returns a properly formatted DataFrame.