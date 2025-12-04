# Visualizations

The following visualization utilities create informative plots for comparing methods and understanding model performance.

## Method comparison

The `MethodComparisonResults` class and `method_comparison_results()` factory function provide visualization for comparing multiple imputation methods.

### method_comparison_results

```python
def method_comparison_results(
    data: Union[pd.DataFrame, Dict[str, Dict[str, Dict]]],
    metric_name: Optional[str] = None,
    metric: str = "quantile_loss",
    data_format: str = "wide",
) -> MethodComparisonResults
```

| Parameter | Type | Default used | Description |
|-----------|------|---------|-------------|
| data | DataFrame or Dict | - | Comparison data from cross-validation results |
| metric_name | str | None | Deprecated, use `metric` instead |
| metric | str | "quantile_loss" | Metric to visualize: "quantile_loss", "log_loss", or "combined" |
| data_format | str | "wide" | Input format: "wide", "long", or "dual_metrics" |

Returns: `MethodComparisonResults` object for visualization.

### MethodComparisonResults class

```python
class MethodComparisonResults:
    def plot(
        self,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_mean: bool = True,
        figsize: Tuple[int, int] = (PLOT_CONFIG["width"], PLOT_CONFIG["height"]),
        plot_type: str = "bar",
    ) -> go.Figure

    def summary(self, format: str = "wide") -> pd.DataFrame

    def get_best_method(self, criterion: str = "mean") -> str
```

`MethodComparisonResults.plot()` parameters:

| Parameter | Type | Default used | Description |
|-----------|------|---------|-------------|
| title | str | None | Custom plot title |
| save_path | str | None | Path to save the plot |
| show_mean | bool | True | Show horizontal lines for mean loss |
| figsize | tuple | (width, height) | Figure dimensions in pixels |
| plot_type | str | "bar" | Plot type: "bar" for grouped bars, "stacked" for contribution analysis |

The `"stacked"` plot type shows rank-based contribution scores, useful for understanding how each variable contributes to overall model performance.

`MethodComparisonResults.summary()` parameters:

| Parameter | Type | Default used | Description |
|-----------|------|---------|-------------|
| format | str | "wide" | Output format: "wide" for methods as columns, "long" for stacked |

`MethodComparisonResults.get_best_method()` parameters:

| Parameter | Type | Default used | Description |
|-----------|------|---------|-------------|
| criterion | str | "mean" | Selection criterion: "mean" or "median" |

### Example usage

```python
from microimpute.visualizations import method_comparison_results

# Create comparison visualization from cross-validation results
comparison_viz = method_comparison_results(
    data=cv_results,
    metric="quantile_loss",
    data_format="wide",
)

# Generate plot
fig = comparison_viz.plot(
    title="Method comparison",
    show_mean=True,
    plot_type="bar"
)
fig.show()

# Get summary statistics
summary_df = comparison_viz.summary(format="wide")

# Identify best method
best = comparison_viz.get_best_method(criterion="mean")
```

## Individual model performance

The `PerformanceResults` class and `model_performance_results()` factory function visualize single model performance.

### model_performance_results

```python
def model_performance_results(
    results: Union[pd.DataFrame, Dict[str, Dict[str, any]]],
    model_name: Optional[str] = None,
    method_name: Optional[str] = None,
    metric: str = "quantile_loss",
    class_probabilities: Optional[Dict[str, pd.DataFrame]] = None,
    y_true: Optional[Dict[str, np.ndarray]] = None,
    y_pred: Optional[Dict[str, np.ndarray]] = None,
) -> PerformanceResults
```

| Parameter | Type | Default used | Description |
|-----------|------|---------|-------------|
| results | DataFrame or Dict | - | Performance results from cross-validation |
| model_name | str | None | Name of the model |
| method_name | str | None | Name of the imputation method |
| metric | str | "quantile_loss" | Metric to visualize: "quantile_loss", "log_loss", or "combined" |
| class_probabilities | Dict | None | Class probability DataFrames for categorical |
| y_true | Dict | None | True values for confusion matrix |
| y_pred | Dict | None | Predicted values for confusion matrix |

Returns: `PerformanceResults` object for visualization.

### PerformanceResults class

```python
class PerformanceResults:
    def plot(
        self,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (PLOT_CONFIG["width"], PLOT_CONFIG["height"]),
    ) -> go.Figure

    def summary(self) -> pd.DataFrame
```

`PerformanceResults.plot()` parameters:

| Parameter | Type | Default used | Description |
|-----------|------|---------|-------------|
| title | str | None | Custom plot title |
| save_path | str | None | Path to save the plot |
| figsize | tuple | (width, height) | Figure dimensions in pixels |

For quantile loss, the plot shows train and test loss across quantiles as grouped bars. For log loss, the plot includes the loss bars and optionally confusion matrix and class probability distribution subplots. For combined metrics, both are shown in subplots.

### Example usage

```python
from microimpute.visualizations import model_performance_results

# Visualize cross-validation results for a single model
perf_viz = model_performance_results(
    results=cv_results["quantile_loss"]["results"],
    model_name="QRF",
    method_name="Cross-validation",
    metric="quantile_loss"
)

fig = perf_viz.plot(title="QRF performance")
fig.show()

# Get summary statistics
summary = perf_viz.summary()
```

## Plot customization

All plots are created using Plotly and return `go.Figure` objects that can be further customized using the standard Plotly API. Plots use a consistent light gray background (`#F0F0F0`) and the Plotly qualitative color palette for consistency across the documentation.
