# Hot-Deck Matching

The `Matching` model imputes missing values using nearest neighbor distance hot deck matching. It finds donor records that are similar to each recipient based on predictor variables and transfers the donor's observed values.

## Variable type support

Matching handles numerical, categorical, boolean and mixed targets. It transfers observed donor values, preserving their support and within-record target combinations. The distribution among recipients depends on donor selection and can differ from the donor marginal distribution.

## How it works

The implementation builds on R's StatMatch package, accessed through the rpy2 interface.

During fitting, the model stores the complete donor dataset and the relevant variable names. During prediction, each record in the test dataset (the recipients) is compared against the stored donors using distance calculations on the predictor variables. The algorithm finds the closest donor for each recipient and transfers the target variable values.

Because the imputed values are drawn from actually observed records, the natural relationships in the original data are preserved.

## Key features

Matching is non-parametric: it makes no assumptions about the data distribution. This makes it useful when the data doesn't fit standard parametric models, or when the relationships between predictors and targets are hard to specify in closed form.

Donated values respect the observed target bounds and discrete categories. Nearest-neighbor selection determines how often donors contribute, so multimodality and skewness need assessment in the resulting recipient population.

`fitted.predict(receiver)` returns a DataFrame with one donor draw per recipient. Matching rejects explicit quantiles and `return_probs=True`, because a donor draw does not estimate conditional quantiles or class probabilities. AutoImpute excludes Matching from distributional ranking, but includes its draws when explicitly requested with `impute_all=True`. Distributional predictor analysis rejects Matching. See the [migration guide](../../imputation-benchmarking/migration.md).
