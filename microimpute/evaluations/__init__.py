"""Model evaluation and validation utilities

This module provides comprehensive tools for evaluating imputation model performance
using cross-validation techniques and predictor analysis. It includes utilities for
assessing predictor importance, correlations, and sensitivity analysis.

Key components:
    - cross_validate_model: perform k-fold cross-validation for imputation models with optional hyperparameter tuning
    - compute_predictor_correlations: analyze correlations between predictor variables
    - leave_one_out_analysis: evaluate importance of each predictor by removal
    - progressive_predictor_inclusion: find optimal predictor subset and ordering
"""

from microimpute.evaluations.cross_validation import cross_validate_model
from microimpute.evaluations.predictor_analysis import (
    compute_predictor_correlations,
    leave_one_out_analysis,
    progressive_predictor_inclusion,
)
