# Creating a new imputer model

This document shows how to create a new imputation model by extending the `Imputer` and `ImputerResults` abstract base classes.

## Architecture

Microimpute uses a two-class architecture:

1. **Imputer**: Handles model initialization and fitting
2. **ImputerResults**: Represents a fitted model and handles prediction

This separation is similar to statsmodels' approach. Look at the existing model implementations for reference.

```python
from typing import Dict, List, Optional, Any

import pandas as pd
from pydantic import validate_call

from microimpute.models.imputer import Imputer, ImputerResults
from microimpute.config import VALIDATE_CONFIG
```

## Implementing a model results class

First, we need to implement the `ImputerResults` subclass that will represent our fitted model and handle predictions. Let's create a model-specific imputer results class:

```python
class NewModelResults(ImputerResults):
    """
    Fitted Model imputer ready for prediction.
    """

    def __init__(
        self,
        predictors: List[str],
        imputed_variables: List[str],
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Initialize the NewModelResults parameter.

        Args:
            predictors: List of predictor variable names
            imputed_variables: List of imputed variable names
            **kwargs: Additional keyword arguments for model parameters
        """
        super().__init__(predictors, imputed_variables)
        # Add any additional model specific parameters here

    # You may choose to validate your model parameters with pydantic
    @validate_call(config=VALIDATE_CONFIG)
    def _predict(
        self, X_test: pd.DataFrame, quantiles: Optional[List[float]] = None
    ) -> Dict[float, pd.DataFrame]:
        """
        Predict imputed values at specified quantiles.

        Args:
            X_test: DataFrame containing the test data
            quantiles: List of quantiles to predict. If None, predicts at median

        Returns:
            Dictionary mapping quantiles to DataFrames with predicted values

        Raises:
            RuntimeError: If prediction fails
        """
        try:
            # Implement model specific prediction functionality...

            return

        except Exception as e:
            self.logger.error(f"Error during Model prediction: {str(e)}")
            raise RuntimeError(
                f"Failed to predict with Model: {str(e)}"
            ) from e
```

## Implementing the main model class

Next, let's implement the main `Imputer` subclass that will handle model initialization and fitting:

```python
class NewModel(Imputer):
    """
    Imputation model to be fitted.
    """

    def __init__(self) -> None:
        """Initialize the model parameters."""
        super().__init__()

    @validate_call(config=VALIDATE_CONFIG)
    def _fit(
        self,
        X_train: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        **kwargs: Any,
    ) -> NewModelResults:
        """
        Fit the Model on training data.

        Args:
            X_train: DataFrame containing training data
            predictors: List of predictor variable names
            imputed_variables: List of variable names to impute
            **kwargs: Additional arguments passed specific to Model

        Returns:
            NewModelResults instance with the fitted model

        Raises:
            RuntimeError: If model fitting fails
        """
        try:
            # Implement model specific training functionality...

            # Return the results object with fitted models
            return NewModelResults(
                predictors=predictors,
                imputed_variables=imputed_variables,
                **kwargs,  # Pass any additional model parameters here
            )

        except Exception as e:
            self.logger.error(f"Error fitting Model: {str(e)}")
            raise RuntimeError(f"Failed to fit Model: {str(e)}") from e
```

## Testing the new model

You can test the functionality of your newly implemented `NewModel` imputer model with a simple example using the Diabetes dataset:

```python
from sklearn.datasets import load_diabetes
from microimpute.utils.data import preprocess_data

# Load the Diabetes dataset
diabetes = load_diabetes()
diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

# Define predictors and variables to impute
predictors = ["age", "sex", "bmi", "bp"]
imputed_variables = ["s1", "s4"]

# Filter the data
data = diabetes_df[predictors + imputed_variables]

# Split into train and test
X_train, X_test, dummy_info = preprocess_data(data)

# Initialize our new model
new_imputer = NewModel()

# Fit the model
fitted_model = new_imputer.fit(
    X_train,
    predictors,
    imputed_variables,
)

# Make predictions at different quantiles
test_quantiles = [0.1, 0.5, 0.9]
predictions = fitted_model.predict(X_test, test_quantiles)

# Print sample predictions
for q in test_quantiles:
    print(f"\nPredictions at {q} quantile:")
    print(predictions[q].head())
```

## Integrating with the benchmarking framework

The new `NewModel` model is then ready to be integrated into the Microimpute benchmarking framework. Here's how you would compare it with other models:

```python
from microimpute.models import OLS, QRF
from microimpute.comparisons import get_imputations
from microimpute.comparisons.metrics import compare_metrics
from microimpute.visualizations import method_comparison_results

# Define models to compare
model_classes = [NewModel, OLS, QRF]

# Get test data for evaluation
Y_test = X_test[imputed_variables]

# Get imputations from all models
method_imputations = get_imputations(
    model_classes, X_train, X_test, predictors, imputed_variables
)

# Compare metrics across methods
loss_comparison_df = compare_metrics(Y_test, method_imputations, imputed_variables)

# Plot the comparison
comparison_viz = method_comparison_results(
    data=loss_comparison_df,
    metric="quantile_loss",
    data_format="long",
)
fig = comparison_viz.plot(show_mean=True)
fig.show()
```

## Best practices

### Architecture

Create an `Imputer` subclass for fitting and an `ImputerResults` subclass for prediction. Implement `_fit()` in the former and `_predict()` in the latter. Look at existing models to see how they handle iterative imputation across multiple target variables, which is needed for cross-method comparison via quantile loss.

### Error handling

Wrap fitting and prediction in try/except blocks. Use `ValueError` for bad inputs, `RuntimeError` for operational failures, and include informative messages. Use `self.logger` for logging significant events (fitting start/end, parameter values, warnings).

### Parameters and validation

Add type hints to all methods. Use the `@validate_call(config=VALIDATE_CONFIG)` decorator for parameter validation. Document model-specific parameters in docstrings with their purpose, expected values, and defaults.

### Testing

Write tests for both interface compliance (does your model follow the expected API?) and model-specific correctness (does it produce sensible results?).
