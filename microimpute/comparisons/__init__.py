"""Imputation method comparison and evaluation utilities

This module provides comprehensive tools for comparing and evaluating different
imputation methods. It includes automated model selection, quantile loss metrics,
and validation utilities for ensuring data integrity.

Key components:
    - autoimpute: automated imputation method selection and application
    - get_imputations: generate imputations using multiple model classes
    - quantile_loss: calculate quantile-based loss metrics
    - compare_quantile_loss: compare performance across imputation methods
    - Validation utilities for data and parameter validation
"""

# Import automated imputation utilities
from microimpute.comparisons.autoimpute import AutoImputeResult, autoimpute

# Import imputation utilities
from microimpute.comparisons.imputations import get_imputations

# Import loss functions
from microimpute.comparisons.quantile_loss import (
    compare_quantile_loss,
    compute_quantile_loss,
    quantile_loss,
)

# Import validation utilities
from microimpute.comparisons.validation import (
    validate_columns_exist,
    validate_dataframe_compatibility,
    validate_imputation_inputs,
    validate_quantiles,
)
