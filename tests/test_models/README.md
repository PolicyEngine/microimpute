# Imputer Model Tests

This directory contains tests for the `Imputer` abstract base class and its implementations.

## Overview

The tests in this directory verify that all imputation models in this package:

1. Correctly inherit from the `Imputer` abstract base class
2. Implement the required interface methods (`fit` and `predict`)
3. Have interchangeable functionality through the common interface
4. Can be evaluated using common testing approaches like cross-validation
5. Provide consistent outputs in expected formats

## Test Files

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

## Using the Imputer Interface

### Base Interface

All imputation models inherit from `Imputer` and implement:

```python
def fit(self, X_train, predictors, imputed_variables, **kwargs) -> "Imputer":
    """Fit the model to training data."""
    pass

def predict(self, test_X, quantiles=None) -> Dict[float, Union[np.ndarray, pd.DataFrame]]:
    """Predict imputed values at specified quantiles."""
    pass
```

### Example: Using Models Interchangeably

```python
# Function that works with any Imputer model
def impute_values(imputer: Imputer, train_data, test_data, predictors, target):
    # Fit the model
    imputer.fit(train_data, predictors, [target])
    
    # Make predictions at median
    predictions = imputer.predict(test_data, [0.5])
    
    return predictions[0.5]

# Use with different model types
ols_preds = impute_values(OLS(), train_data, test_data, predictors, target)
qrf_preds = impute_values(QRF(), train_data, test_data, predictors, target)
```

## Available Model Implementations

### OLS (Ordinary Least Squares)

- Simple linear regression model
- Assumes normally distributed residuals
- Predicts quantiles by adding scaled normal quantiles to the mean prediction

```python
model = OLS()
model.fit(train_data, predictors, target_vars)
predictions = model.predict(test_data, [0.25, 0.5, 0.75])
```

### QuantReg (Quantile Regression)

- Directly models conditional quantiles
- Can capture asymmetric distributions
- Fits separate models for each quantile

```python
model = QuantReg()
model.fit(train_data, predictors, target_vars, quantiles=[0.25, 0.5, 0.75])
predictions = model.predict(test_data)  # Uses pre-fitted quantiles
```

### QRF (Quantile Random Forest)

- Uses random forests to model quantiles
- Can capture complex nonlinear relationships
- Supports RF hyperparameters through kwargs

```python
model = QRF()
model.fit(train_data, predictors, target_vars, n_estimators=100)
predictions = model.predict(test_data, [0.25, 0.5, 0.75])
```

### Matching (Statistical Matching)

- Uses distance hot deck matching to find donors
- Non-parametric approach based on R's StatMatch package
- Returns matched donor values for all quantiles

```python
model = Matching()
model.fit(train_data, predictors, target_vars)
predictions = model.predict(test_data, [0.5])
```
