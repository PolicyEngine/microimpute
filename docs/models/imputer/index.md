# The Imputer class

The `Imputer` class is the abstract base class that defines the common interface for all imputation models in microimpute. Every model inherits from it and implements the required abstract methods for fitting and prediction.

## Key features

All models share `fit()` and `predict()` methods, with model-specific capabilities. Matching returns donor draws and rejects quantile and probability requests. MDN rejects sample weights; OLS, QRF and Matching support weighted fitting through `weight_col`.

Use `fit(..., target_types={"status": "categorical"})` to declare numeric-coded categories explicitly. Numeric counts and 0/1 integers otherwise remain numeric. See the [migration guide](../../imputation-benchmarking/migration.md) for probability scoring, QRF modes and fitted-model compatibility.

The design enforces that `predict()` cannot be called before `fit()`. The base implementation also handles parameter and input data validation, so individual models don't need to duplicate those checks.

When using imputers in isolation (not through `autoimpute`), preprocessing is available via `preprocess_data`, which can normalize the data and split it into train/test sets. See [matching-imputation.ipynb](../matching/matching-imputation.ipynb) for an example.
