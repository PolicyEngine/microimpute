# Metrics and evaluation

This page documents the evaluation metrics and predictor analysis tools for assessing imputation quality.

## Loss metrics

Microimpute selects the evaluation metric based on whether the imputed variable is numerical or categorical.

### Quantile loss

Quantile loss assesses imputation quality for numerical variables. It captures performance across different parts of the distribution, unlike mean squared error which only measures average accuracy.

The quantile loss implements the standard pinball loss formulation:

$$L_q(y, f) = \max(q(y-f), (q-1)(y-f))$$

where $q$ is the quantile being evaluated, $y$ represents the true value, and $f$ is the imputed value. This asymmetric loss function penalizes under-prediction more heavily for higher quantiles and over-prediction more heavily for lower quantiles. The asymmetry aligns with the interpretation of quantiles: a 90th percentile prediction should rarely fall below the true value, while a 10th percentile prediction should rarely exceed it.

```python
def quantile_loss(q: float, y: np.ndarray, f: np.ndarray) -> np.ndarray
```

| Parameter | Type | Description |
|-----------|------|-------------|
| q | float | Quantile to evaluate (e.g., 0.5 for median) |
| y | np.ndarray | True values |
| f | np.ndarray | Predicted values |

Returns an array of element-wise quantile losses.

### Log loss

Log loss (cross-entropy) evaluates probabilistic predictions of categorical outcomes. It measures the performance of a classification model where the prediction output is a probability value between 0 and 1.

The log loss metric is calculated as:

$$\text{LogLoss} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{M} y_{ij} \log(p_{ij})$$

where $N$ is the number of samples, $M$ is the number of classes, $y_{ij}$ is 1 if sample $i$ belongs to class $j$ and 0 otherwise, and $p_{ij}$ is the predicted probability of sample $i$ belonging to class $j$.

A perfect classifier achieves a log loss of 0, while worse predictions yield increasingly higher values. The metric heavily penalizes confident misclassifications: predicting a class with high probability when incorrect results in a large loss value.

```python
def log_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = True,
    labels: Optional[np.ndarray] = None,
) -> float
```

| Parameter | Type | Default used | Description |
|-----------|------|---------|-------------|
| y_true | np.ndarray | - | True class labels |
| y_pred | np.ndarray | - | Predicted probabilities or class labels |
| normalize | bool | True | If True, return mean loss; if False, return sum |
| labels | np.ndarray | None | List of possible label values |

Returns the Log loss value (float).

When predictions are class labels rather than probabilities, the function converts them to high-confidence probabilities (0.99/0.01) with a warning. For more accurate evaluation, use probability predictions when available.

### compute_loss

A unified function that selects the loss metric based on the specified type.

```python
def compute_loss(
    test_y: np.ndarray,
    imputations: np.ndarray,
    metric: Literal["quantile_loss", "log_loss"],
    q: float = 0.5,
    labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]
```

| Parameter | Type | Default used | Description |
|-----------|------|---------|-------------|
| test_y | np.ndarray | - | True values |
| imputations | np.ndarray | - | Predicted/imputed values |
| metric | str | - | "quantile_loss" or "log_loss" |
| q | float | 0.5 | Quantile (for quantile_loss only) |
| labels | np.ndarray | None | Class labels (for log_loss only) |

Returns a tuple of `(element_wise_losses, mean_loss)`.

### compare_metrics

Compares metrics across multiple imputation methods, detecting variable types and applying the appropriate metric. For models that handle both numerical and categorical variables, results are produced separately for each metric type.

```python
def compare_metrics(
    test_y: pd.DataFrame,
    method_imputations: Dict[str, Dict[float, pd.DataFrame]],
    imputed_variables: List[str],
) -> pd.DataFrame
```

| Parameter | Type | Description |
|-----------|------|-------------|
| test_y | pd.DataFrame | DataFrame containing true values |
| method_imputations | Dict | Nested dict: method → quantile → DataFrame |
| imputed_variables | List[str] | Variables to evaluate |

Returns a DataFrame with columns `Method`, `Imputed Variable`, `Percentile`, `Loss`, and `Metric`.

## Distribution comparison

Evaluating how well imputed values preserve distributional characteristics tells you whether the imputation maintains the statistical properties of the original data.

### Wasserstein distance

For continuous numerical variables, the Wasserstein distance (Earth Mover's Distance) quantifies the difference between distributions:

$$W_p(P, Q) = \left(\inf_{\gamma \in \Pi(P, Q)} \int_{X \times Y} d(x, y)^p d\gamma(x, y)\right)^{1/p}$$

where $\Pi(P, Q)$ denotes the set of all joint distributions whose marginals are $P$ and $Q$ respectively. The Wasserstein distance measures the minimum "work" required to transform one distribution into another, where work is the amount of distribution mass moved times the distance moved. Lower values indicate better preservation of the original distribution's shape.

When sample weights are provided, the weighted Wasserstein distance accounts for varying observation importance, which is essential when comparing survey data with different sampling designs. We use scipy's `wasserstein_distance` implementation, which supports sample weights via the `u_weights` and `v_weights` parameters.

### Kullback-Leibler divergence

For discrete distributions (categorical and boolean variables), KL divergence quantifies how one probability distribution diverges from a reference:

$$D_{KL}(P||Q) = \sum_{x \in \mathcal{X}} P(x) \log\left(\frac{P(x)}{Q(x)}\right)$$

where $P$ is the reference distribution (original data), $Q$ is the approximation (imputed data), and $\mathcal{X}$ is the set of all possible categorical values. KL divergence measures how much information is lost when using the imputed distribution to approximate the true distribution. Lower values indicate better preservation of the original categorical distribution.

When sample weights are provided, the probability distributions are computed as weighted proportions rather than simple counts, so that weighted survey data can be compared correctly.

### kl_divergence

Computes the Kullback-Leibler divergence between two categorical distributions, with optional sample weights.

```python
def kl_divergence(
    donor_values: np.ndarray,
    receiver_values: np.ndarray,
    donor_weights: Optional[np.ndarray] = None,
    receiver_weights: Optional[np.ndarray] = None,
) -> float
```

| Parameter | Type | Default used | Description |
|-----------|------|---------|-------------|
| donor_values | np.ndarray | - | Categorical values from donor data (reference distribution) |
| receiver_values | np.ndarray | - | Categorical values from receiver data (approximation) |
| donor_weights | np.ndarray | None | Optional sample weights for donor values |
| receiver_weights | np.ndarray | None | Optional sample weights for receiver values |

Returns KL divergence value (float >= 0), where 0 indicates identical distributions.

### compare_distributions

Compares distributions between donor and receiver data, automatically selecting the appropriate metric based on variable type and supporting sample weights for survey data.

```python
def compare_distributions(
    donor_data: pd.DataFrame,
    receiver_data: pd.DataFrame,
    imputed_variables: List[str],
    donor_weights: Optional[Union[pd.Series, np.ndarray]] = None,
    receiver_weights: Optional[Union[pd.Series, np.ndarray]] = None,
) -> pd.DataFrame
```

| Parameter | Type | Default used | Description |
|-----------|------|---------|-------------|
| donor_data | pd.DataFrame | - | Original donor data |
| receiver_data | pd.DataFrame | - | Receiver data with imputations |
| imputed_variables | List[str] | - | Variables to compare |
| donor_weights | pd.Series or np.ndarray | None | Sample weights for donor data (must match donor_data length) |
| receiver_weights | pd.Series or np.ndarray | None | Sample weights for receiver data (must match receiver_data length) |

Returns a DataFrame with columns `Variable`, `Metric`, and `Distance`. The function automatically selects Wasserstein distance for numerical variables and KL divergence for categorical variables.

Note that data must not contain null or infinite values. If your data contains such values, filter them before calling this function.

## Predictor analysis

These tools analyze which predictors contribute most to imputation quality, helping with feature selection and model interpretation.

### Mutual information

Mutual information measures the reduction in uncertainty about one variable given knowledge of another. Unlike correlation coefficients, which capture only linear relationships, mutual information detects any statistical dependency.

For discrete random variables $X$ and $Y$:

$$I(X; Y) = \sum_{x \in X} \sum_{y \in Y} p(x, y) \log\left(\frac{p(x, y)}{p(x)p(y)}\right)$$

For continuous variables, the summations are replaced by integrals. The normalized mutual information (NMI) used in the implementation is:

$$\text{NMI}(X; Y) = \frac{I(X; Y)}{\sqrt{H(X) \cdot H(Y)}}$$

where $H(X)$ and $H(Y)$ are the entropies of $X$ and $Y$ respectively. Normalized values range from 0 (no relationship) to 1 (perfect dependency), allowing direct comparison of predictor importance across different variable types.

### compute_predictor_correlations

```python
def compute_predictor_correlations(
    data: pd.DataFrame,
    predictors: List[str],
    imputed_variables: Optional[List[str]] = None,
    method: str = "all",
) -> Dict[str, pd.DataFrame]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| data | pd.DataFrame | - | Dataset containing predictors and target variables |
| predictors | List[str] | - | Column names of predictor variables |
| imputed_variables | List[str] | None | Column names of target variables |
| method | str | "all" | Which correlation method to use: "all", "mi" (mutual information), "pearson", or "spearman" |

Returns a dictionary containing DataFrames with correlation scores (e.g. `predictor_target_mi` for mutual information).

### Leave-one-out analysis

Leave-one-out predictor analysis evaluates model performance when each predictor is excluded. Predictors whose removal causes large increases in loss are the most important, while those with minimal impact might be candidates for removal.

### leave_one_out_analysis

```python
def leave_one_out_analysis(
    data: pd.DataFrame,
    predictors: List[str],
    imputed_variables: List[str],
    model_class: Type[Imputer],
    weight_col: Optional[Union[str, np.ndarray, pd.Series]] = None,
    quantiles: List[float] = QUANTILES,
    train_size: float = TRAIN_SIZE,
    n_jobs: int = 1,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| data | pd.DataFrame | - | Complete dataset |
| predictors | List[str] | - | Column names of predictor variables |
| imputed_variables | List[str] | - | Column names of variables to impute |
| model_class | Type[Imputer] | - | Imputer class to evaluate |
| weight_col | str, np.ndarray, or pd.Series | None | Sample weights column name or array |
| quantiles | List[float] | [0.05 to 0.95 in steps of 0.05] | Quantiles to evaluate |
| train_size | float | 0.8 | Proportion of data for training |
| n_jobs | int | 1 | Number of parallel jobs |
| random_state | int | 42 | Random seed for reproducibility |

Returns a DataFrame containing loss increase and relative impact for each predictor.

### Progressive predictor inclusion

Progressive inclusion adds predictors one at a time, ordered by their mutual information with the target. This greedy forward selection reveals the optimal inclusion order and the marginal contribution of each predictor. Diminishing returns in loss reduction indicate when additional predictors add little.

### progressive_predictor_inclusion

```python
def progressive_predictor_inclusion(
    data: pd.DataFrame,
    predictors: List[str],
    imputed_variables: List[str],
    model_class: Type[Imputer],
    weight_col: Optional[Union[str, np.ndarray, pd.Series]] = None,
    quantiles: Optional[List[float]] = QUANTILES,
    train_size: Optional[float] = TRAIN_SIZE,
    max_predictors: Optional[int] = None,
    random_state: Optional[int] = RANDOM_STATE,
) -> Dict[str, Any]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| data | pd.DataFrame | - | Complete dataset |
| predictors | List[str] | - | Column names of predictor variables |
| imputed_variables | List[str] | - | Column names of variables to impute |
| model_class | Type[Imputer] | - | Imputer class to evaluate |
| weight_col | str, np.ndarray, or pd.Series | None | Sample weights column name or array |
| quantiles | List[float] | [0.05 to 0.95 in steps of 0.05] | Quantiles to evaluate |
| train_size | float | 0.8 | Proportion of data for training |
| max_predictors | int | None | Maximum number of predictors to include (None for all) |
| random_state | int | 42 | Random seed for reproducibility |

Returns a dictionary containing `inclusion_order` (list of predictors in optimal order) and `predictor_impacts` (list of dicts with predictor name and loss reduction).

## Example usage

```python
from microimpute.comparisons.metrics import compare_metrics, compare_distributions
from microimpute.evaluations import (
    compute_predictor_correlations,
    leave_one_out_analysis,
    progressive_predictor_inclusion,
)
from microimpute.models import QRF

# Compare methods
metrics_df = compare_metrics(
    test_y=test_data[imputed_variables],
    method_imputations={
        "QRF": qrf_imputations,
        "OLS": ols_imputations,
    },
    imputed_variables=imputed_variables,
)

# Evaluate distributional match with survey weights
dist_df_weighted = compare_distributions(
    donor_data=donor,
    receiver_data=receiver_with_imputations,
    imputed_variables=imputed_variables,
    donor_weights=donor["sample_weight"],
    receiver_weights=receiver["sample_weight"],
)

# Analyze predictor importance
mi_scores = compute_predictor_correlations(
    data, predictors, imputed_variables, method="mi"
)
loo_results = leave_one_out_analysis(
    data, predictors, imputed_variables, QRF, weight_col="wgt"
)
inclusion_results = progressive_predictor_inclusion(
    data, predictors, imputed_variables, QRF, weight_col="wgt"
)
```
