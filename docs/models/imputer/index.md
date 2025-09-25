# The Imputer class

The `Imputer` class serves as an abstract base class that defines the common interface for all imputation models within the Microimpute framework. It establishes a foundational structure with essential methods for data validation, model fitting, and prediction. Every specialized imputation model in the system inherits from this class and implements the required abstract methods to provide its unique functionality.

## Key features

The Imputer architecture provides numerous benefits to the overall system design. It defines a consistent API with standardized `fit()` and `predict()` methods, ensuring that all models can be used interchangeably regardless of their underlying implementation details. This uniformity makes it straightforward to swap imputation techniques within your workflow. Thus, all imputers will share basic functionality like the handling of weighted data, using a "weights" column for sampling training data, to preserve data distributions better.

The design carefully enforces proper usage by ensuring no model can call `predict()` without first fitting the model to the data. This logical constraint helps prevent common errors and makes the API more intuitive to use. Additionally, the base implementation handles validation of parameters and input data, reducing code duplication across different model implementations and ensuring that all models perform appropriate validation checks.

When using the different imputers in isolation, and not as part of wider pipeline functions like `autoimpute` preprocessing is supported by `preprocess_data` which can help normalize the data and split it into train and test splits. For an example of how to integrate them see [matching-imputation.ipynb](../matching/matching-imputation.ipynb).
