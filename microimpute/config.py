"""
Configuration module for MicroImpute.

Shared validation, analysis and plotting settings, plus compatibility constants
for existing callers. Model implementations define their own runtime defaults.
"""

from typing import Any, Dict, List

import numpy as np
from pydantic import ConfigDict

# Define a configuration for pydantic validation that allows
# arbitrary types like pd.DataFrame
VALIDATE_CONFIG = ConfigDict(arbitrary_types_allowed=True)

# Historical SCF years retained for imports in existing notebooks and callers.
# This compatibility list does not restrict datasets accepted by the imputers.
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

# Data configuration
TRAIN_SIZE: float = 0.8
TEST_SIZE: float = 0.2

# Analysis configuration
QUANTILES: List[float] = [round(q, 2) for q in np.arange(0.05, 1.00, 0.05)]

# Random state for reproducibility
RANDOM_STATE: int = 42

# Historical parameter mapping retained for import compatibility. This mapping
# does not configure the learners; their implementations own runtime defaults.
DEFAULT_MODEL_PARAMS: Dict[str, Dict[str, Any]] = {
    "qrf": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": 1.0,
    },
    "quantreg": {},
    "ols": {
        "l1_ratio": 0,
        "C": 1.0,
        "max_iter": 1000,
    },
    "matching": {},
    "mdn": {
        "layers": "128-64-32",
        "activation": "ReLU",
        "dropout": 0.0,
        "use_batch_norm": False,
        "num_gaussian": 5,
        "softmax_temperature": 1.0,
        "n_samples": 100,
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
    # Plotly Safe palette - colorblind-friendly
    "color_palette": [
        "#88CCEE",  # Cyan
        "#CC6677",  # Rose
        "#DDCC77",  # Sand
        "#117733",  # Green
        "#332288",  # Indigo
        "#AA4499",  # Purple
        "#44AA99",  # Teal
        "#999933",  # Olive
        "#882255",  # Wine
        "#661100",  # Brown
    ],
    # Background colors (same for both)
    "plot_bgcolor": "#FAFAFA",
    "paper_bgcolor": "#FAFAFA",
    # Grid styling (horizontal only)
    "gridcolor": "#E5E5E5",
    "gridwidth": 1,
    "showgrid_x": False,
    "showgrid_y": True,
    # Axis line styling
    "linecolor": "#CCCCCC",
    "showline": True,
}
