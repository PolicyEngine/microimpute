"""Statistical imputation models

This module provides a collection of statistical models for data imputation,
including both parametric and non-parametric approaches. Each model extends
the base Imputer class and provides quantile-based predictions.

Available models:
    - OLS: ordinary least squares regression with bootstrapped quantiles
    - QRF: quantile random forest for non-parametric quantile regression
    - QuantReg: linear quantile regression model
    - Matching: statistical matching/hot-deck imputation (optional, requires rpy2)

Base classes:
    - Imputer: abstract base class for all imputation models
    - ImputerResults: container for fitted model and prediction methods
"""

# Import base classes
from microimpute.models.imputer import Imputer, ImputerResults

try:
    from microimpute.models.matching import Matching
except ImportError:
    pass

# Import specific model implementations
from microimpute.models.ols import OLS
from microimpute.models.qrf import QRF
from microimpute.models.quantreg import QuantReg
