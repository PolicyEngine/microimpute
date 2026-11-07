"""Statistical imputation models

This module provides a collection of statistical models for data imputation,
including both parametric and non-parametric approaches. Each model extends
the base Imputer class and provides quantile-based predictions.

Available models:
    - OLS: ordinary least squares regression with bootstrapped quantiles
    - QRF: quantile regression forest for non-parametric quantile regression
    - QuantReg: linear quantile regression model
    - Matching: statistical matching/hot-deck imputation (optional, requires rpy2)
    - MDN: Mixture Density Network for probabilistic imputation
        (optional, requires pytorch-tabular)
    - ZeroInflatedImputer: wrapper composing a model for the probability of a
        zero with a model for the positive part, for variables such as asset
        holdings where a large share of the population is at zero

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

try:
    from microimpute.models.mdn import MDN
except ImportError:
    pass

# Import specific model implementations
from microimpute.models.ols import OLS
from microimpute.models.qrf import QRF
from microimpute.models.quantreg import QuantReg
from microimpute.models.zero_inflated import ZeroInflatedImputer
