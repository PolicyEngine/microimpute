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

TRAIN_SIZE: float = 0.8
TEST_SIZE: float = 0.2

# Analysis configuration
QUANTILES: List[float] = [round(q, 2) for q in np.arange(0.05, 1.00, 0.05)]

# Random state for reproducibility
RANDOM_STATE: int = 42

# Model parameters (passed via **kwargs to fit() or as __init__ params)

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
