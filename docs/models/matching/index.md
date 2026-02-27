# Hot-Deck Matching

The `Matching` model imputes missing values using nearest neighbor distance hot deck matching. It finds donor records that are similar to each recipient based on predictor variables and transfers the donor's observed values.

## Variable type support

Matching handles any variable type: numerical, categorical, boolean, or mixed. Because it transfers actual observed values rather than generating predictions, it preserves the original data type and distribution of each variable.

## How it works

The implementation builds on R's StatMatch package, accessed through the rpy2 interface.

During fitting, the model stores the complete donor dataset and the relevant variable names. During prediction, each record in the test dataset (the recipients) is compared against the stored donors using distance calculations on the predictor variables. The algorithm finds the closest donor for each recipient and transfers the target variable values.

Because the imputed values are drawn from actually observed records, the natural relationships in the original data are preserved.

## Key features

Matching is non-parametric: it makes no assumptions about the data distribution. This makes it useful when the data doesn't fit standard parametric models, or when the relationships between predictors and targets are hard to specify in closed form.

The method preserves the empirical distribution of the imputed variables. Since values come directly from observed data points, features like multimodality, skewness, and natural bounds are maintained. A model-based approach might smooth these away.

One limitation is that Matching does not incorporate quantile information. It matches donor and receiver units identically regardless of the quantile being predicted, which means it cannot distinguish between different parts of the conditional distribution. It may also fail to capture non-linear predictor-target relationships despite producing a plausible marginal distribution.
