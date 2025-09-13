"""Model evaluation and validation utilities

This module provides comprehensive tools for evaluating imputation model performance
using cross-validation techniques. It calculates train and test quantile loss metrics
across multiple folds to provide robust performance estimates.

Key components:
    - cross_validate_model: perform k-fold cross-validation for imputation models with optional hyperparameter tuning
"""

from microimpute.evaluations.cross_validation import cross_validate_model
