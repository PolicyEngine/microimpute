# Quantile Regression

The `QuantReg` model directly models specific quantiles of the target variable distribution. Unlike methods that model the conditional mean and derive quantiles from distributional assumptions, quantile regression addresses each conditional quantile separately.

## Variable type support

QuantReg is designed for numerical variables and does not support categorical imputation. If your targets include categorical variables, use OLS or QRF instead, which handle both numerical and categorical targets through internal classification methods.

## How it works

The implementation uses statsmodels' `QuantReg`. During training, a separate regression model is fitted for each requested quantile level.

The objective function minimizes asymmetrically weighted absolute residuals rather than squared residuals (as in OLS). For higher quantiles, under-predictions are penalized more heavily; for lower quantiles, over-predictions are penalized more heavily. This asymmetry causes each model to converge toward the true conditional quantile.

When predicting, the system applies the quantile-specific model for each requested level. Predictions at different quantiles come from distinct models, each optimized for that part of the distribution.

## Key features

Quantile regression models conditional quantiles without assuming a particular error distribution, making it robust to outliers. It naturally captures heteroscedasticity, adapting to changing variance patterns across the feature space (unlike OLS, which assumes constant variance).

By fitting multiple quantile levels, the method gives a picture of the full conditional distribution. This can reveal asymmetries that other methods would miss. However, QuantReg is limited to linear relationships between predictors and the target.
