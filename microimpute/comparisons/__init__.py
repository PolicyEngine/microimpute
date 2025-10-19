"""Imputation method comparison and evaluation utilities

This module provides comprehensive tools for comparing and evaluating different
imputation methods. It includes automated model selection, quantile loss metrics,
and validation utilities for ensuring data integrity.

Key components:
    - autoimpute: automated imputation method selection and application
    - get_imputations: generate imputations using multiple model classes
    - metrics: calculate quantile loss and log loss metrics based on variable type
    - compare_metrics: compare performance across imputation methods using appropriate metrics
    - Validation utilities for data and parameter validation
"""

# Import automated imputation utilities
from microimpute.comparisons.autoimpute import AutoImputeResult, autoimpute

# Import imputation utilities
from microimpute.comparisons.imputations import get_imputations

# Import loss/metric functions
from microimpute.comparisons.metrics import (
    compare_distributions,
    compare_metrics,
    compute_loss,
    get_metric_for_variable_type,
    kl_divergence,
    log_loss,
    quantile_loss,
    wasserstein_distance,
)

# Import validation utilities
from microimpute.comparisons.validation import (
    validate_columns_exist,
    validate_dataframe_compatibility,
    validate_imputation_inputs,
    validate_quantiles,
)
