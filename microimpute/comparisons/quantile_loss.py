"""Quantile loss calculation functions for imputation evaluation.

This module contains utilities for evaluating imputation quality using quantile loss metrics.
It implements the standard quantile loss function that penalizes under-prediction more heavily
for higher quantiles and over-prediction more heavily for lower quantiles.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from pydantic import validate_call

from microimpute.comparisons.validation import (
    validate_columns_exist,
    validate_quantiles,
)
from microimpute.config import QUANTILES, VALIDATE_CONFIG

log = logging.getLogger(__name__)


@validate_call(config=VALIDATE_CONFIG)
def quantile_loss(q: float, y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Calculate the quantile loss.

    Args:
        q: Quantile to be evaluated, e.g., 0.5 for median.
        y: True value.
        f: Fitted or predicted value.

    Returns:
        Array of quantile losses.
    """
    e = y - f
    return np.maximum(q * e, (q - 1) * e)


@validate_call(config=VALIDATE_CONFIG)
def compute_quantile_loss(
    test_y: np.ndarray, imputations: np.ndarray, q: float
) -> np.ndarray:
    """Compute quantile loss for given true values and imputations.

    Args:
        test_y: Array of true values.
        imputations: Array of predicted/imputed values.
        q: Quantile value.

    Returns:
        np.ndarray: Element-wise quantile loss values between true values and predictions.

    Raises:
        ValueError: If q is not between 0 and 1.
        ValueError: If test_y and imputations have different shapes.
    """
    try:
        # Validate quantile value
        validate_quantiles([q])

        # Validate input dimensions
        if len(test_y) != len(imputations):
            error_msg = (
                f"Length mismatch: test_y has {len(test_y)} elements, "
                f"imputations has {len(imputations)} elements"
            )
            log.error(error_msg)
            raise ValueError(error_msg)

        log.debug(
            f"Computing quantile loss for q={q} with {len(test_y)} samples"
        )
        losses = quantile_loss(q, test_y, imputations)
        mean_loss = np.mean(losses)
        log.debug(f"Quantile loss at q={q}: mean={mean_loss:.6f}")

        return losses

    except (TypeError, AttributeError) as e:
        log.error(f"Error computing quantile loss: {str(e)}")
        raise RuntimeError(f"Failed to compute quantile loss: {str(e)}") from e


def _compute_method_losses(
    method: str,
    imputation: Dict[float, pd.DataFrame],
    test_y: pd.DataFrame,
    imputed_variables: List[str],
    quantiles: List[float],
) -> List[Dict]:
    """Compute losses for a single method across all quantiles and variables.

    Args:
        method: Name of the imputation method.
        imputation: Dictionary mapping quantiles to imputation DataFrames.
        test_y: DataFrame containing true values.
        imputed_variables: List of variables to evaluate.
        quantiles: List of quantiles to evaluate.

    Returns:
        List of dictionaries containing loss results.

    Raises:
        ValueError: If required quantiles or variables are missing.
    """
    results = []

    for quantile in quantiles:
        log.debug(f"Computing loss for {method} at quantile {quantile}")

        # Validate that the quantile exists in the imputation results
        if quantile not in imputation:
            error_msg = f"Quantile {quantile} not found in imputations for method {method}"
            log.error(error_msg)
            raise ValueError(error_msg)

        variable_losses = []

        for variable in imputed_variables:
            # Validate variable exists
            if variable not in imputation[quantile].columns:
                error_msg = f"Variable {variable} not found in imputation results for method {method}"
                log.error(error_msg)
                raise ValueError(error_msg)

            # Get values
            test_values = test_y[variable].values
            pred_values = imputation[quantile][variable].values

            # Compute loss
            q_loss = compute_quantile_loss(test_values, pred_values, quantile)
            variable_losses.append(q_loss.mean())

            # Add variable-specific result
            results.append(
                {
                    "Method": method,
                    "Imputed Variable": variable,
                    "Percentile": quantile,
                    "Loss": q_loss.mean(),
                }
            )

            log.debug(
                f"Loss for {method}/{variable} at q={quantile}: {q_loss.mean():.6f}"
            )

        # Add average across variables for this quantile
        avg_var_loss = np.mean(variable_losses)
        results.append(
            {
                "Method": method,
                "Imputed Variable": "mean_loss",
                "Percentile": quantile,
                "Loss": avg_var_loss,
            }
        )

    # Add overall average across all quantiles
    all_quantile_losses = [
        r["Loss"]
        for r in results
        if r["Imputed Variable"] == "mean_loss"
        and r["Percentile"] != "mean_loss"
    ]
    if all_quantile_losses:
        avg_quant_loss = np.mean(all_quantile_losses)
        results.append(
            {
                "Method": method,
                "Imputed Variable": "mean_loss",
                "Percentile": "mean_loss",
                "Loss": avg_quant_loss,
            }
        )

    return results


@validate_call(config=VALIDATE_CONFIG)
def compare_quantile_loss(
    test_y: pd.DataFrame,
    method_imputations: Dict[str, Dict[float, pd.DataFrame]],
    imputed_variables: List[str],
) -> pd.DataFrame:
    """Compare quantile loss across different imputation methods.

    Args:
        test_y: DataFrame containing true values.
        method_imputations: Nested dictionary mapping method names
            to dictionaries mapping quantiles to imputation values.
        imputed_variables: List of variables to evaluate.

    Returns:
        pd.DataFrame: Results dataframe with columns 'Method',
            'Percentile', and 'Loss' containing the mean quantile
            loss for each method and percentile.

    Raises:
        ValueError: If input data formats are invalid.
        RuntimeError: If comparison operation fails.
    """
    try:
        log.info(
            f"Comparing quantile loss for {len(method_imputations)} methods: {list(method_imputations.keys())}"
        )
        log.info(f"Using {len(QUANTILES)} quantiles: {QUANTILES}")
        log.info(f"True values shape: {test_y.shape}")

        # Validate inputs
        validate_columns_exist(test_y, imputed_variables, "test_y")

        # Collect all results in a list first (more efficient than repeated DataFrame concatenation)
        all_results = []

        # Process each method
        for method, imputation in method_imputations.items():
            method_results = _compute_method_losses(
                method, imputation, test_y, imputed_variables, QUANTILES
            )
            all_results.extend(method_results)

        # Create DataFrame from all results at once
        results_df = pd.DataFrame(all_results)

        log.info(f"Comparison complete. Results shape: {results_df.shape}")

        return results_df

    except ValueError as e:
        # Re-raise validation errors
        raise e
    except (KeyError, TypeError, AttributeError) as e:
        log.error(f"Error in quantile loss comparison: {str(e)}")
        raise RuntimeError(f"Failed to compare quantile loss: {str(e)}") from e
