"""Utility functions for microimpute operations

This module provides utility functions that support various microimpute processes,
including data preprocessing, normalization/unnormalization, and optional R-based
statistical matching functionality.

Key components:
    - preprocess_data: prepare and normalize data for imputation
    - unnormalize_predictions: convert normalized predictions back to original scale
    - nnd_hotdeck_using_rpy2: R-based nearest neighbor hot deck imputation (optional)
"""

from microimpute.utils.data import preprocess_data, unnormalize_predictions

# Optional import for R-based functions
try:
    from microimpute.utils.statmatch_hotdeck import nnd_hotdeck_using_rpy2
except ImportError:
    # rpy2 is not available, matching functionality will be limited
    nnd_hotdeck_using_rpy2 = None
