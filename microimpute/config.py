"""
Configuration module for MicroImpute.

This module centralizes all constants and configuration parameters used across
the package.
"""

from typing import Any, Dict, List

import numpy as np
from pydantic import ConfigDict

# Define a configuration for pydantic validation that allows
# arbitrary types like pd.DataFrame
VALIDATE_CONFIG = ConfigDict(arbitrary_types_allowed=True)

# Data configuration
VALID_YEARS: List[int] = [
    1989,
    1992,
    1995,
    1998,
    2001,
    2004,
    2007,
    2010,
    2013,
    2016,
    2019,
    2022,
]

TRAIN_SIZE: float = 0.8
TEST_SIZE: float = 0.2

# Analysis configuration
QUANTILES: List[float] = [round(q, 2) for q in np.arange(0.05, 1.00, 0.05)]

# Random state for reproducibility
RANDOM_STATE: int = 42

# Model parameters (passed via **kwargs to fit() or as __init__ params)
DEFAULT_MODEL_PARAMS: Dict[str, Dict[str, Any]] = {
    "qrf": {
        # RandomForestQuantileRegressor parameters
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": 1.0,
    },
    "quantreg": {
        # statsmodels QuantReg uses default parameters
    },
    "ols": {
        # statsmodels OLS uses default parameters
        # LogisticRegression params for categorical targets:
        "penalty": "l2",
        "C": 1.0,
        "max_iter": 1000,
    },
    "matching": {
        # StatMatch NND hotdeck default parameters
    },
    "mdn": {
        # Backbone network parameters
        "layers": "128-64-32",
        "activation": "ReLU",
        "dropout": 0.0,
        "use_batch_norm": False,
        # MDN head parameters
        "num_gaussian": 5,
        "softmax_temperature": 1.0,
        "n_samples": 100,
        # Training parameters
        "learning_rate": 1e-3,
        "max_epochs": 100,
        "early_stopping_patience": 10,
        "batch_size": 256,
    },
}

# Plotting configuration
PLOT_CONFIG: Dict[str, Any] = {
    "width": 750,
    "height": 600,
    "colors": {},
}
