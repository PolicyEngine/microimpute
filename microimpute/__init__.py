"""MicroImpute Package

A package for benchmarking different imputation methods using microdata.
"""

__version__ = "1.1.2"

from microimpute.comparisons.imputations import get_imputations

# Import comparison utilities
from microimpute.comparisons.quantile_loss import (
    compare_quantile_loss,
    compute_quantile_loss,
    quantile_loss,
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

# Import main models and utilities
from microimpute.models import OLS, QRF, Imputer, ImputerResults, QuantReg

# Import data handling functions
from microimpute.utils.data import prepare_scf_data, preprocess_data

try:
    from microimpute.models.matching import Matching
except ImportError:
    pass

# Import visualization modules
from microimpute.visualizations.plotting import (
    method_comparison_results,
    model_performance_results,
)
