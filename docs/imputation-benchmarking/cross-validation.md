# Cross-validation and model imputation comparison

This page documents the cross-validation utilities for evaluating imputation model performance. Cross-validation provides robust 
estimates of how well a model will generalize to unseen data by training and testing on multiple data splits. Functions like `get_imputations`, will then build upon it, to standardize evaluation for all models, ensuring possible through a consistent experimental setup.

Microimpute's cross-validation automatically selects the appropriate metric based on variable type. Numerical variables are evaluated using quantile loss, which measures prediction accuracy across the conditional distribution. Categorical variables are evaluated using log loss (cross-entropy), which penalizes confident but incorrect predictions, see the [Metrics page](./metrics.md) for more details.

## Cross-validation

Cross-validation provides robust estimates of how well a model will generalize to unseen data by training and testing on multiple data splits. Microimpute's cross-validation automatically selects the appropriate metric based on variable type: quantile loss for numerical variables and log loss for categorical variables.

### cross_validate_model

```python
def cross_validate_model(
    model_class: Type,
    data: pd.DataFrame,
    predictors: List[str],
    imputed_variables: List[str],
    weight_col: Optional[str] = None,
    quantiles: Optional[List[float]] = QUANTILES,
    n_splits: Optional[int] = 5,
    random_state: Optional[int] = RANDOM_STATE,
    model_hyperparams: Optional[dict] = None,
    tune_hyperparameters: Optional[bool] = False,
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict]]
```

| Parameter | Type | Default used | Description |
|-----------|------|---------|-------------|
| model_class | Type | - | Imputer class to evaluate (e.g., QRF, OLS, MDN) |
| data | pd.DataFrame | - | Complete dataset for cross-validation |
| predictors | List[str] | - | Column names of predictor variables |
| imputed_variables | List[str] | - | Column names of variables to impute |
| weight_col | str | None | Column name for sampling weights |
| quantiles | List[float] | [0.05 to 0.95 in steps of 0.05] | Quantiles to evaluate |
| n_splits | int | 5 | Number of cross-validation folds |
| random_state | int | 42 | Random seed for reproducibility |
| model_hyperparams | dict | None | Hyperparameters to pass to the model |
| tune_hyperparameters | bool | False | Enable hyperparameter tuning |

Returns a dictionary containing separate results for each metric type:

```python
{
    "quantile_loss": {
        "results": pd.DataFrame,      # rows: ["train", "test"], cols: quantiles (mean across folds)
        "results_std": pd.DataFrame,  # rows: ["train", "test"], cols: quantiles (std across folds)
        "mean_train": float,
        "mean_test": float,
        "std_train": float,
        "std_test": float,
        "variables": List[str]        # numerical variables evaluated
    },
    "log_loss": {
        "results": pd.DataFrame,      # rows: ["train", "test"], cols: quantiles
        "results_std": pd.DataFrame,  # rows: ["train", "test"], cols: quantiles (std across folds)
        "mean_train": float,
        "mean_test": float,
        "std_train": float,
        "std_test": float,
        "variables": List[str]        # categorical variables evaluated
    }
}
```

The `results_std` DataFrame and `std_train`/`std_test` values provide the standard deviation of the loss across cross-validation folds, which can be used to visualize uncertainty via error bars.

If `tune_hyperparameters=True`, returns a tuple of `(results_dict, best_hyperparameters)`.

## Example usage

```python
from microimpute.evaluations import cross_validate_model
from microimpute.models import QRF

# Run 5-fold cross-validation
results = cross_validate_model(
    model_class=QRF,
    data=diabetes_df,
    predictors=["age", "sex", "bmi", "bp"],
    imputed_variables=["s1", "s4"],
    n_splits=5
)

# Check performance for numerical variables
print(f"Mean test quantile loss: {results['quantile_loss']['mean_test']:.4f}")

# View detailed results by quantile
print(results["quantile_loss"]["results"])
```

## Interpreting results

The results DataFrame shows loss values for each quantile, with rows for train and test splits. Lower values indicate better performance. Comparing train and test loss helps identify overfitting: a large gap suggests the model may not generalize well.

For model selection, focus on the test loss (`mean_test`). When comparing multiple models, the `autoimpute()` function automates this comparison and selects the best-performing model using a rank-based approach that handles mixed variable types.

## Imputation generation for model comparison

The `get_imputations` function generates imputations using cross-validation for multiple model classes in a single call, organizing results in a consistent format for downstream comparison and evaluation.

### get_imputations

```python
def get_imputations(
    model_classes: List[Type],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    predictors: List[str],
    imputed_variables: List[str],
    quantiles: Optional[List[float]] = QUANTILES,
) -> Dict[str, Dict[float, pd.DataFrame]]
```

| Parameter | Type | Default used | Description |
|-----------|------|---------|-------------|
| model_classes | List[Type] | - | List of model classes to use (e.g., QRF, OLS, QuantReg, Matching) |
| X_train | pd.DataFrame | - | Training data containing predictors and variables to impute |
| X_test | pd.DataFrame | - | Test data on which to generate imputations |
| predictors | List[str] | - | Column names of predictor variables |
| imputed_variables | List[str] | - | Column names of variables to impute |
| quantiles | List[float] | [0.05 to 0.95 in steps of 0.05] | List of quantiles to predict |

Returns a nested dictionary mapping method names to dictionaries of quantile-indexed DataFrames:

```python
{
    "QRF": {
        0.1: pd.DataFrame,  # predictions at 10th percentile
        0.5: pd.DataFrame,  # predictions at 50th percentile
        0.9: pd.DataFrame,  # predictions at 90th percentile
    },
    "OLS": {
        0.1: pd.DataFrame,
        ...
    },
}
```

### Example usage

```python
from microimpute.comparisons import get_imputations
from microimpute.models import QRF, OLS, QuantReg, Matching

# Generate imputations from multiple models
method_imputations = get_imputations(
    model_classes=[QRF, OLS, QuantReg, Matching],
    X_train=train_data,
    X_test=test_data,
    predictors=["age", "sex", "bmi"],
    imputed_variables=["income", "wealth"],
)

# Access predictions for a specific model and quantile
qrf_median = method_imputations["QRF"][0.5]
```