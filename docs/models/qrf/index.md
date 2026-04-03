# Quantile Regression Forests

The `QRF` model uses an ensemble of decision trees to predict different quantiles of the target variable distribution. This allows it to model non-linear relationships while estimating uncertainty across the conditional distribution.

## Variable type support

QRF handles both numerical and categorical variables. For numerical targets, it uses quantile regression forests. For categorical targets (strings, booleans, numerically-encoded categorical variables), it automatically uses a Random Forest Classifier. The model detects variable types internally and requires no manual specification.

## How it works

Quantile Regression Forests build on standard random forests using the `quantile_forest` package. The method constructs an ensemble of decision trees, each trained on a bootstrapped sample of the data (bagging). At each split, only a random subset of features is considered, which introduces diversity among trees and reduces overfitting.

Unlike standard random forests that aggregate predictions into averages, QRF retains the full predictive distribution from each tree and estimates quantiles directly from this empirical distribution.

## Key features

QRF is non-parametric and makes minimal assumptions about the data structure. It adapts its uncertainty estimates to different regions of the input space, producing wider prediction intervals where the data is more variable and tighter intervals where it is less so.

The method is computationally heavier than linear models, but often more accurate on datasets with non-linear relationships or heteroscedasticity (where variance depends on predictor values). Hyperparameter tuning via Optuna is available for optimizing the number of trees, minimum samples per leaf, split thresholds, and feature sampling.
