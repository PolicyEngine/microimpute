"""Microimpute: advanced statistical imputation for microdata

Microimpute is a comprehensive Python package for benchmarking and applying
various statistical imputation methods to microdata. It provides tools for
automated model selection, cross-validation, and visualization of imputation results.

Key features:
    - Multiple imputation models (OLS, QRF, QuantReg, Matching)
    - Automated model selection with cross-validation
    - Quantile-based imputation and evaluation
    - Comprehensive visualization tools
    - Support for weighted imputation
    - Hyperparameter tuning capabilities

Main components:
    - autoimpute: automated imputation with model selection
    - Models: OLS, QRF, QuantReg, Matching (optional)
    - Evaluation: cross-validation and quantile loss metrics
    - Visualization: performance and comparison plots
"""

__version__ = "1.1.2"

# Import automated imputation
from microimpute.comparisons.autoimpute import AutoImputeResult, autoimpute
from microimpute.comparisons.imputations import get_imputations

# Import comparison and metric utilities
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

# Main configuration
from microimpute.config import (
    PLOT_CONFIG,
    QUANTILES,
    RANDOM_STATE,
    VALIDATE_CONFIG,
)

# Import evaluation modules
from microimpute.evaluations.cross_validation import cross_validate_model
from microimpute.evaluations.predictor_analysis import (
    compute_predictor_correlations,
    leave_one_out_analysis,
    progressive_predictor_inclusion,
)

# Import main models and utilities
from microimpute.models import OLS, QRF, Imputer, ImputerResults, QuantReg

# Import data handling functions
from microimpute.utils.data import preprocess_data, unnormalize_predictions

try:
    from microimpute.models.matching import Matching
except ImportError:
    pass

# Import visualization modules
from microimpute.visualizations import (
    MethodComparisonResults,
    PerformanceResults,
    method_comparison_results,
    model_performance_results,
)
