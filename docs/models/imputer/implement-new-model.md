# Creating a New Imputer Model

This document demonstrates how to create a new imputation model by extending the `Imputer` and `ImputerResults` abstract base classes in MicroImpute.

## 1. Understanding the MicroImpute Architecture

MicroImpute uses a two-class architecture for imputation models:

1. **Imputer**: The base model class that handles model initialization and fitting
2. **ImputerResults**: Represents a fitted model and handles prediction

This separation provides a clean distinction between the model definition and the fitted model instance, similar to statsmodels' approach.

```python
from typing import Dict, List, Optional, Any

import pandas as pd
from pydantic import validate_call

from microimpute.models.imputer import Imputer, ImputerResults
from microimpute.config import VALIDATE_CONFIG
```

## 2. Implementing a Model Results Class

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

## 3. Implementing the Main Model Class

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

## 4. Testing the New Model

You can test the functionality of your newly implemented `NewModel` imputer model with a simple example using the Iris dataset:

```python
from sklearn.datasets import load_iris
from microimpute.comparisons.data import preprocess_data

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Define predictors and variables to impute
predictors = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)"]
imputed_variables = ["petal width (cm)"]

# Filter the data
data = iris_df[predictors + imputed_variables]

# Split into train and test
X_train, X_test = preprocess_data(data)

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

## 5. Integrating with the Benchmarking Framework

The new `NewModel` model is then ready to be integrated into the MicroImpute benchmarking framework. Here's how you would compare it with other models:

```python
from microimpute.models import OLS, QRF
from microimpute.comparisons import (
    get_imputations,
    compare_quantile_loss,
)
from microimpute.comparisons.plot import plot_loss_comparison

# Define models to compare
model_classes = [NewModel, OLS, QRF]

# Get test data for evaluation
Y_test = X_test[imputed_variables]

# Get imputations from all models
method_imputations = get_imputations(
    model_classes, X_train, X_test, predictors, imputed_variables
)

# Compare quantile loss
loss_comparison_df = compare_quantile_loss(Y_test, method_imputations, imputed_variables)

# Plot the comparison
plot_loss_comparison(loss_comparison_df)
```

## 6. Best Practices for Implementing New Models

When implementing a new imputation model for MicroImpute, adhering to certain best practices will ensure your model integrates seamlessly with the framework and provides a consistent experience for users.

### Architecture

The two-class architecture forms the foundation of a well-designed imputation model. You should create an `Imputer` subclass that handles model definition and fitting operations, establishing the core functionality of your approach. This class should be complemented by an `ImputerResults` subclass that represents the fitted model state and handles all prediction-related tasks. This separation of concerns creates a clean distinction between the fitting and prediction phases of your model's lifecycle.

Within these classes, you must implement the required abstract methods to fulfill the contract with the base classes. Your `Imputer` subclass should provide a thorough implementation of the `_fit()` method that handles the training process for your specific algorithm. Similarly, your `ImputerResults` subclass needs to implement the `_predict()` method that applies the fitted model to new data and generates predictions at requested quantiles.

### Error Handling

Robust error handling is crucial for creating reliable imputation models. Your implementation should wrap model fitting and prediction operations in appropriate try/except blocks to capture and handle potential errors gracefully. When exceptions occur, provide informative error messages that help users understand what went wrong and how to address the issue. Use appropriate error types such as ValueError for input validation failures and RuntimeError for operational failures during model execution.

Effective logging complements good error handling by providing visibility into the model's operation. Use the self.logger instance consistently throughout your code to record important information about the model's state and progress. Log significant events like the start and completion of fitting operations, parameter values, and any potential issues or warnings that arise during execution.

### Parameters and Validation

Type safety and parameter validation enhance the usability and reliability of your model. Add comprehensive type hints to all methods and parameters to enable better IDE support and make your code more self-documenting. Apply the `validate_call` decorator with the standard VALIDATE_CONFIG configuration to method signatures to enforce parameter validation consistently.

Your implementation should thoughtfully support model-specific parameters that may be needed to control the behavior of your algorithm. Design your `_fit()` and `_predict()` methods to accept and properly utilize these parameters, ensuring they affect the model's operation as intended. Document all parameters clearly in your docstrings, explaining their purpose, expected values, and default behavior to guide users in effectively configuring your model.

### Documentation

Comprehensive documentation makes your model accessible to others. Include detailed class-level docstrings that explain your model's theoretical approach, strengths, limitations, and appropriate use cases. Document all methods with properly structured docstring sections covering arguments, return values, and potential exceptions. Where appropriate, provide usage examples that demonstrate how to initialize, train, and use your model for prediction tasks.

The documentation should be complemented by thorough unit tests that verify your implementation works correctly. Create tests that check both basic interface compliance (ensuring your model adheres to the expected API) and model-specific functionality (validating that your algorithm produces correct results). Comprehensive testing helps catch issues early and provides confidence that your implementation will work reliably in production environments.