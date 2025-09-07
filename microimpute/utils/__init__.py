"""
This module contains utilities that support microimpute processes.
"""

from .data import preprocess_data
from .logging_utils import configure_logging

# Optional import for R-based functions
try:
    from .statmatch_hotdeck import nnd_hotdeck_using_rpy2
except ImportError:
    # rpy2 is not available, matching functionality will be limited
    nnd_hotdeck_using_rpy2 = None
