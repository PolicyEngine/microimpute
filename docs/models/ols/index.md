# Ordinary least squares

The `OLS` model employs linear regression techniques to predict missing values by leveraging the relationships between predictor and target variables. This classic statistical approach provides a computationally efficient method for imputation while offering theoretical guarantees under certain assumptions.

## Variable type support

OLS automatically adapts to your target variable types. For numerical variables, it uses standard linear regression. For categorical variables (including strings, booleans, or numerically-encoded categorical variables), it automatically switches to logistic regression classification. This automatic detection means you don't need to specify variable types—simply pass your predictors and targets, and the model handles the rest internally.

## How it works

The OLS imputer works by fitting a linear regression model using the statsmodels implementation of Ordinary Least Squares. During the training phase, it identifies the coefficients that minimize the sum of squared residuals between the predicted and actual values in the training data. This creates a model that captures the linear relationship between the predictors and target variables.

For prediction at different quantiles, the model makes an important assumption that the residuals (the differences between predicted and actual values) follow a normal distribution. This assumption allows the model to generate predictions at various quantiles by starting with the mean prediction and adding a quantile-specific offset derived from the normal distribution. Specifically, it computes the standard error of the predictions and applies the inverse normal cumulative distribution function to generate predictions at the requested quantiles.

## Key features

The OLS imputer offers a simple yet powerful parametric approach with fast training and prediction times compared to more complex models. It relies on the assumption of linear relationships between variables, making it particularly suitable for datasets where such relationships hold or as a baseline comparison for more complex approaches.
