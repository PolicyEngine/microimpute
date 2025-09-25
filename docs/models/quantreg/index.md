# Quantile Regression

The `QuantReg` model takes a direct approach to modeling specific quantiles of the target variable distribution. Unlike methods that model the mean and then derive quantiles from distributional assumptions, quantile regression addresses each conditional quantile explicitly, providing greater flexibility and robustness in heterogeneous data settings.

## Variable type support

QuantReg is designed specifically for numerical variables and does not support categorical variable imputation. If your imputation targets include categorical variables (string, or numerically-encoded categorical variables), consider using OLS or QRF models instead, which automatically handle both numerical and categorical targets through internal classification methods.

## How it works

Quantile regression in MicroImpute leverages the statsmodels' QuantReg implementation to create precise models of conditional quantiles. During the training phase, the approach fits separate regression models for each requested quantile level, creating a focused model for each part of the conditional distribution you wish to estimate.

The mathematical foundation of the method lies in its objective function, which minimizes asymmetrically weighted absolute residuals rather than squared residuals as in ordinary least squares. This asymmetric weighting system penalizes under-predictions more heavily when estimating higher quantiles and over-predictions more heavily when estimating lower quantiles. This clever formulation allows the model to converge toward solutions that represent true conditional quantiles.

When making predictions, the system applies the appropriate quantile-specific model for each requested quantile level. This direct approach means predictions at different quantiles come from distinct models optimized for those specific portions of the distribution, rather than from a single model with assumptions about the error distribution.

## Key features

Quantile regression offers several compelling advantages for imputation tasks. It allows direct modeling of conditional quantiles without making restrictive assumptions about the underlying distribution of the data. This distribution-free approach makes the method robust to outliers and applicable in a wide range of scenarios where normal distribution assumptions might be violated.

The method excels at capturing heteroscedasticity—situations where the variability of the target depends on the predictor values. While methods like OLS assume constant variance throughout the feature space, quantile regression naturally adapts to changing variance patterns, providing more accurate predictions in regions with different error characteristics.

By fitting multiple quantile levels, the approach provides a comprehensive picture of the conditional distribution of the target variable. This detailed view enables more nuanced imputation where understanding the full range of possible values is important. For instance, it can reveal asymmetries in the conditional distribution that other methods might miss, offering valuable insights into the uncertainty structure of the imputed values.
