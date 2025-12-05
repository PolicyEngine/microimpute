# Microimpute

Microimpute enables variable imputation through a variety of statistical methods. By providing a consistent interface across different imputation techniques, it allows researchers and data scientists to easily compare and benchmark different approaches using quantile loss and log loss calculations to determine the method providing most accurate results.

## Features

### Multiple imputation methods
- **Statistical Matching**: Distance-based matching for finding similar observations
- **Ordinary Least Squares (OLS)**: Linear regression-based imputation
- **Quantile Regression**: Distribution-aware regression imputation
- **Quantile Random Forests (QRF)**: Non-parametric forest-based approach
- **Mixture Density Networks (MDN)**: Neural network with Gaussian mixture approximation head

### Automated method selection
- **AutoImpute**: Automatically compares and selects the best imputation method for your data
- **Cross-validation**: Built-in evaluation using quantile loss (numerical) and log loss (categorical)
- **Variable type support**: Handles numerical, categorical, and boolean variables

### Developer-friendly design
- **Consistent API**: Standardized `fit()` and `predict()` interface across all models
- **Extensible architecture**: Easy to implement custom imputation methods
- **Weighted data handling**: Preserve data distributions with sample weights
- **Input validation**: Automatic parameter and data validation

### Interactive dashboard
- **Visual exploration**: Analyze imputation results through interactive charts at https://microimpute-dashboard.vercel.app/
- **GitHub integration**: Load artifacts directly from CI/CD workflows
- **Multiple data sources**: File upload, URL loading and sample data

## Installation

```bash
pip install microimpute
```

For image export functionality (PNG/JPG), install with:

```bash
pip install microimpute[images]
```

## Examples and documentation

For detailed examples and interactive notebooks, see the [documentation](https://policyengine.github.io/microimpute/).

## Contributing

Contributions are welcome to the project. Please feel free to submit a Pull Request with your improvements.
