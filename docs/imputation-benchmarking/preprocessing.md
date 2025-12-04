# Data preprocessing

Preprocessing transformations can improve model performance by normalizing scale differences or handling skewed distributions. These are supported by `preprocess_data` and transformation-specific functions.

## Transformation options

Microimpute supports three transformation types that can be applied to numeric columns before training. Each transformation automatically excludes categorical and boolean columns to prevent encoding issues.

**Normalization (z-score)** standardizes data to have mean 0 and standard deviation 1. This transformation is useful when predictors have different scales, ensuring that all features contribute equally to distance-based or gradient-based models.

**Log transformation** applies the natural logarithm to values. This is effective for right-skewed distributions common in financial data like income or wealth. The transformation requires all values to be strictly positive.

**Asinh transformation** applies the inverse hyperbolic sine function, which behaves like $\log(2x)$ for large positive values and $-\log(-2x)$ for large negative values, while remaining approximately linear near zero. Unlike log transformation, asinh handles zero and negative values, making it suitable for variables like net worth that can take any real value.

### preprocess_data

The main entry point for data preparation, combining splitting and transformation.

```python
def preprocess_data(
    data: pd.DataFrame,
    full_data: Optional[bool] = False,
    train_size: Optional[float] = TRAIN_SIZE,
    test_size: Optional[float] = TEST_SIZE,
    random_state: Optional[int] = RANDOM_STATE,
    normalize: Optional[Union[bool, List[str]]] = False,
    log_transform: Optional[Union[bool, List[str]]] = False,
    asinh_transform: Optional[Union[bool, List[str]]] = False,
) -> Union[Tuple[pd.DataFrame, dict], Tuple[pd.DataFrame, pd.DataFrame, dict]]
```

| Parameter | Type | Default used | Description |
|-----------|------|---------|-------------|
| data | pd.DataFrame | - | DataFrame containing the data to preprocess |
| full_data | bool | False | If True, return complete dataset without splitting |
| train_size | float | 0.8 | Proportion of data for training split |
| test_size | float | 0.2 | Proportion of data for test split |
| random_state | int | 42 | Random seed for reproducibility |
| normalize | bool or List[str] | False | True for all numeric columns, or list of specific columns |
| log_transform | bool or List[str] | False | True for all numeric columns, or list of specific columns |
| asinh_transform | bool or List[str] | False | True for all numeric columns, or list of specific columns |

The return type depends on parameters. If `full_data=True` and transformations are applied, returns `(data, transform_params)`. If `full_data=False` with transformations, returns `(X_train, X_test, transform_params)`. Without transformations, the transform_params dict is omitted.

### normalize_data

```python
def normalize_data(
    data: pd.DataFrame,
    columns_to_normalize: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, dict]
```

| Parameter | Type | Default used | Description |
|-----------|------|---------|-------------|
| data | pd.DataFrame | - | DataFrame to normalize |
| columns_to_normalize | List[str] | None | Specific columns to normalize. If None, all numeric columns |

It returns uple of `(normalized_data, normalization_params)` where `normalization_params` maps column names to `{"mean": float, "std": float}`.

### log_transform_data

```python
def log_transform_data(
    data: pd.DataFrame,
    columns_to_transform: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, dict]
```

| Parameter | Type | Default used | Description |
|-----------|------|---------|-------------|
| data | pd.DataFrame | - | DataFrame to transform |
| columns_to_transform | List[str] | None | Specific columns to transform. If None, all numeric columns |

Returns a tuple of `(log_transformed_data, log_transform_params)`.

Note: Raises `ValueError` if any values are non-positive.

### asinh_transform_data

```python
def asinh_transform_data(
    data: pd.DataFrame,
    columns_to_transform: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, dict]
```

| Parameter | Type | Default used | Description |
|-----------|------|---------|-------------|
| data | pd.DataFrame | - | DataFrame to transform |
| columns_to_transform | List[str] | None | Specific columns to transform. If None, all numeric columns |

Returns a tuple of `(asinh_transformed_data, asinh_transform_params)`.

## Reversal functions

After imputation, predictions can be converted back to the original scale using the stored parameters.

```python
def unnormalize_predictions(imputations: dict, normalization_params: dict) -> dict
def unlog_transform_predictions(imputations: dict, log_transform_params: dict) -> dict
def un_asinh_transform_predictions(imputations: dict, asinh_transform_params: dict) -> dict
```

Each function takes the imputation dictionary (mapping quantiles to DataFrames) and the parameter dictionary returned by the corresponding transform function, returning imputations in the original scale.

## Usage with autoimpute

The `autoimpute()` function accepts a `preprocessing` parameter that specifies transformations per variable:

```python
from microimpute.comparisons.autoimpute import autoimpute

result = autoimpute(
    donor_data=donor,
    receiver_data=receiver,
    predictors=["age", "education"],
    imputed_variables=["income", "wealth"],
    preprocessing={
        "income": "log",       # Log transform (positive values only)
        "wealth": "asinh",     # Asinh transform (handles zeros/negatives)
        "age": "normalize"     # Z-score normalization
    }
)
```

The transformations are applied automatically before model training and reversed after prediction, so the returned imputations are in the original scale.

## Constraints

Each column can only have one transformation applied. Attempting to apply multiple transformations to the same column raises a `ValueError`. When specifying transformations as `True` (apply to all), only one transformation type can be used. For different transformations on different columns, use the list format to specify columns explicitly.
