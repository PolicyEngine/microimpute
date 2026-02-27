# Ordinary Least Squares

The `OLS` model uses linear regression to predict missing values from the relationships between predictor and target variables. It is computationally fast and provides a useful baseline for comparison with more complex methods.

## Variable type support

OLS adapts to target variable types automatically. For numerical variables, it uses standard linear regression. For categorical variables (including strings, booleans, or numerically-encoded categorical variables), it switches to logistic regression. You don't need to specify variable types manually.

## How it works

The OLS imputer fits a linear regression model using the statsmodels implementation. During training, it finds the coefficients that minimize the sum of squared residuals between predicted and actual values.

For prediction at different quantiles, the model assumes normally distributed residuals. It starts with the mean prediction and adds a quantile-specific offset computed from the normal distribution's inverse CDF and the standard error of the predictions.

## Key features

OLS is fast to train and predict. It works well when the relationship between predictors and targets is approximately linear. Because it assumes constant variance and normally distributed errors, it tends to compress imputed values toward the mean, producing a narrower distribution than the true one. This makes it a good baseline but a poor choice when the data has heavy tails or heteroscedastic errors.
