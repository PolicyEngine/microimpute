"""Visualization utilities for imputation results

This module provides comprehensive visualization tools for analyzing and comparing
imputation model performance. It includes utilities for visualizing individual model
performance metrics and comparing results across multiple imputation methods.

Key components:
    - PerformanceResults: data class for storing model performance visualization results
    - model_performance_results: function to create performance visualizations for a single model
    - MethodComparisonResults: data class for storing method comparison visualization results
    - method_comparison_results: function to create comparison visualizations across methods
"""

# Import from comparison_plots module
from microimpute.visualizations.comparison_plots import (
    MethodComparisonResults,
    method_comparison_results,
)

# Import from performance_plots module
from microimpute.visualizations.performance_plots import (
    PerformanceResults,
    model_performance_results,
)
