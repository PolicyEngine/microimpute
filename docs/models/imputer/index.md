# The Imputer class

The `Imputer` class is the abstract base class that defines the common interface for all imputation models in microimpute. Every model inherits from it and implements the required abstract methods for fitting and prediction.

## Key features

All models share standardized `fit()` and `predict()` methods, so they can be used interchangeably regardless of underlying implementation. All imputers also share functionality like weighted data handling through a `weight_col` parameter.

The design enforces that `predict()` cannot be called before `fit()`. The base implementation also handles parameter and input data validation, so individual models don't need to duplicate those checks.

When using imputers in isolation (not through `autoimpute`), preprocessing is available via `preprocess_data`, which can normalize the data and split it into train/test sets. See [matching-imputation.ipynb](../matching/matching-imputation.ipynb) for an example.
