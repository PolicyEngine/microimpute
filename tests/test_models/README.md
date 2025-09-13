# Imputer model tests

This directory contains tests for the `Imputer` abstract base class and its implementations.

## Overview

The tests in this directory verify that all imputation models in this package:

1. Correctly inherit from the `Imputer` abstract base class
2. Implement the required interface methods (`fit` and `predict`)
3. Have interchangeable functionality through the common interface
4. Can be evaluated using common testing approaches like cross-validation
5. Provide consistent outputs in expected formats

## Test files

- **test_imputers.py**: Verifies the common interface across all models:
  - Tests model initialization with no required arguments
  - Verifies that all models follow the common fit/predict interface
  - Confirms models store predictor and imputed variable names correctly
  - Ensures models can be used interchangeably
  - Tests both explicit and default quantile prediction

- **test_ols.py**: Tests for the Ordinary Least Squares (OLS) imputer model:
  - Cross-validation evaluation on the Iris dataset
  - Basic functionality and prediction format verification
  - Confirms OLS produces symmetric quantile predictions due to normal distribution assumptions

- **test_quantreg.py**: Tests for the Quantile Regression imputer model:
  - Cross-validation evaluation on the Iris dataset
  - Tests the model's ability to be fit to specific quantiles
  - Verifies proper prediction format and structure

- **test_qrf.py**: Tests for the Quantile Random Forest imputer model:
  - Cross-validation evaluation on the Iris dataset
  - Tests model fitting with optional RandomForest hyperparameters
  - Verifies prediction structure across multiple quantiles

- **test_matching.py**: Tests for the Statistical Matching imputer model:
  - Cross-validation evaluation on the Iris dataset
  - Verifies that the model stores donor data correctly
  - Tests that predictions maintain the expected structure
