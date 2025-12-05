"""Metrics for imputation evaluation.

This module contains utilities for evaluating imputation quality using various metrics:
- Quantile loss for numerical variables
- Log loss for categorical variables
- Distributional similarity metrics (Wasserstein distance, KL Divergence)
The module automatically detects which metric to use based on variable type.
"""

import logging
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import validate_call
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance
from sklearn.metrics import log_loss as sklearn_log_loss

from microimpute.comparisons.validation import (
    validate_columns_exist,
    validate_dataframe_compatibility,
    validate_quantiles,
)
from microimpute.config import QUANTILES, VALIDATE_CONFIG
from microimpute.utils.type_handling import VariableTypeDetector

log = logging.getLogger(__name__)

MetricType = Literal["quantile_loss", "log_loss"]


def get_metric_for_variable_type(
    series: pd.Series, col_name: str = "variable"
) -> str:
    """Detect the metric to use depending on whether a variable is categorical or numerical.

    Uses the VariableTypeDetector from the imputer module for consistency.

    Args:
        series: Pandas series to analyze.
        col_name: Name of the column (for logging purposes).

    Returns:
        'log_loss' or ' quantile_loss'
    """
    detector = VariableTypeDetector()
    var_type, _ = detector.categorize_variable(series, col_name, log)

    # Map the detector's output to our binary classification
    if var_type in ["bool", "categorical", "numeric_categorical"]:
        return "log_loss"
    else:
        return "quantile_loss"


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


def log_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = True,
    labels: Optional[np.ndarray] = None,
) -> float:
    """Calculate log loss for categorical predictions.

    Args:
        y_true: True labels (can be class indices or one-hot encoded).
        y_pred: Predicted probabilities. Shape should be (n_samples,) for binary
                or (n_samples, n_classes) for multiclass.
                If class labels are provided instead of probabilities, they will be
                converted to high-confidence probabilities (0.99/0.01) with a warning.
        normalize: If True, return the mean loss. If False, return sum.
        labels: List of labels to include in the loss computation.

    Returns:
        Log loss value.

    Note:
        For more accurate metrics, models should provide predicted probabilities
        rather than class labels. Use model.predict_proba() instead of model.predict()
        when available.
    """
    try:
        # Handle case where predictions are class labels instead of probabilities
        if len(y_pred.shape) == 1 or (
            len(y_pred.shape) == 2 and y_pred.shape[1] == 1
        ):
            # Binary case or class predictions
            if labels is None:
                labels = np.unique(y_true)

            # Convert to probabilities if needed
            if np.all(np.isin(y_pred.flatten(), labels)):
                # These are class predictions, not probabilities
                log.info(
                    "Converting class labels to probabilities for log loss computation. "
                    "For more accurate metrics, please provide predicted probabilities "
                    "using model.predict_proba() or equivalent method instead of class predictions. "
                    "Class labels are being converted to high-confidence probabilities (0.99/0.01)."
                )

                # Create one-hot encoded probabilities with high confidence
                n_samples = len(y_true)
                n_classes = len(labels)

                if n_classes == 2:
                    # Binary case
                    y_pred_proba = np.zeros(n_samples)
                    y_pred_proba[y_pred.flatten() == labels[1]] = 0.99
                    y_pred_proba[y_pred.flatten() == labels[0]] = 0.01
                else:
                    # Multiclass case
                    y_pred_proba = np.full(
                        (n_samples, n_classes), 0.01 / (n_classes - 1)
                    )
                    for i, label in enumerate(labels):
                        mask = y_pred.flatten() == label
                        y_pred_proba[mask, i] = 0.99

                y_pred = y_pred_proba

                log.info(
                    f"Converted {n_samples} class predictions to probabilities "
                    f"for {n_classes}-class classification."
                )

        return sklearn_log_loss(
            y_true, y_pred, normalize=normalize, labels=labels
        )
    except Exception as e:
        log.error(f"Error computing log loss: {str(e)}")
        raise RuntimeError(f"Failed to compute log loss: {str(e)}") from e


def order_probabilities_alphabetically(
    probabilities: np.ndarray,
    model_classes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Order probability matrix columns to match alphabetically sorted class labels.

    The probabilities from sklearn's predict_proba() are ordered according to the model's
    .classes_ attribute, which may not be in alphabetical order. This function reorders
    them alphabetically, which is required for sklearn's log_loss function.

    Args:
        probabilities: Probability matrix from model.predict_proba(), shape (n_samples, n_classes)
                      where columns are ordered according to model.classes_
        model_classes: The model's .classes_ attribute indicating the current order of columns

    Returns:
        Tuple of (reordered_probabilities, alphabetically_sorted_labels)
    """
    # Get the alphabetical order of classes
    alphabetical_indices = np.argsort(model_classes)
    alphabetical_classes = model_classes[alphabetical_indices]

    # Reorder probability columns to match alphabetical order
    reordered_probabilities = probabilities[:, alphabetical_indices]

    return reordered_probabilities, alphabetical_classes


@validate_call(config=VALIDATE_CONFIG)
def compute_loss(
    test_y: np.ndarray,
    imputations: np.ndarray,
    metric: MetricType,
    q: float = 0.5,
    labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Compute loss for given true values and imputations using specified metric.

    Args:
        test_y: Array of true values.
        imputations: Array of predicted/imputed values.
        metric: Type of metric to use ('quantile_loss' or 'log_loss').
        q: Quantile value (only used for quantile_loss).
        labels: Possible label values (only used for log_loss).

    Returns:
        Tuple of (element-wise losses or single loss value, mean loss)

    Raises:
        ValueError: If inputs have different shapes or invalid metric type.
    """
    try:
        # Validate input dimensions
        if len(test_y) != len(imputations):
            error_msg = (
                f"Length mismatch: test_y has {len(test_y)} elements, "
                f"imputations has {len(imputations)} elements"
            )
            log.error(error_msg)
            raise ValueError(error_msg)

        if metric == "quantile_loss":
            # Validate quantile value
            validate_quantiles([q])

            log.debug(
                f"Computing quantile loss for q={q} with {len(test_y)} samples"
            )
            losses = quantile_loss(q, test_y, imputations)
            mean_loss = np.mean(losses)
            log.debug(f"Quantile loss at q={q}: mean={mean_loss:.6f}")
            return losses, mean_loss

        elif metric == "log_loss":
            log.debug(f"Computing log loss with {len(test_y)} samples")
            # Log loss returns a single value by default
            loss_value = log_loss(
                test_y, imputations, normalize=True, labels=labels
            )
            log.debug(f"Log loss: {loss_value:.6f}")
            # Return array of same value for consistency
            losses = np.full(len(test_y), loss_value)
            return losses, loss_value

        else:
            raise ValueError(f"Unknown metric type: {metric}")

    except (TypeError, AttributeError) as e:
        log.error(f"Error computing {metric}: {str(e)}")
        raise RuntimeError(f"Failed to compute {metric}: {str(e)}") from e


def _compute_method_losses(
    method: str,
    imputation: Dict[float, pd.DataFrame],
    test_y: pd.DataFrame,
    imputed_variables: List[str],
    quantiles: List[float],
    variable_metrics: Dict[str, MetricType],
) -> List[Dict]:
    """Compute losses for a single method across all quantiles and variables.

    Args:
        method: Name of the imputation method.
        imputation: Dictionary mapping quantiles to imputation DataFrames.
        test_y: DataFrame containing true values.
        imputed_variables: List of variables to evaluate.
        quantiles: List of quantiles to evaluate.
        variable_metrics: Dictionary mapping variable names to metric types.

    Returns:
        List of dictionaries containing loss results.

    Raises:
        ValueError: If required quantiles or variables are missing.
    """
    results = []

    # Separate variables by metric type
    quantile_vars = [
        v for v in imputed_variables if variable_metrics[v] == "quantile_loss"
    ]
    categorical_vars = [
        v for v in imputed_variables if variable_metrics[v] == "log_loss"
    ]

    for quantile in quantiles:
        log.debug(f"Computing loss for {method} at quantile {quantile}")

        # Validate that the quantile exists in the imputation results
        if quantile not in imputation:
            error_msg = f"Quantile {quantile} not found in imputations for method {method}"
            log.error(error_msg)
            raise ValueError(error_msg)

        # Process quantile loss variables
        quantile_losses = []
        for variable in quantile_vars:
            # Validate variable exists
            if variable not in imputation[quantile].columns:
                error_msg = f"Variable {variable} not found in imputation results for method {method}"
                log.error(error_msg)
                raise ValueError(error_msg)

            # Get values
            test_values = test_y[variable].values
            pred_values = imputation[quantile][variable].values

            # Compute loss
            _, mean_loss = compute_loss(
                test_values, pred_values, "quantile_loss", q=quantile
            )
            quantile_losses.append(mean_loss)

            # Add variable-specific result
            results.append(
                {
                    "Method": method,
                    "Imputed Variable": variable,
                    "Percentile": quantile,
                    "Loss": mean_loss,
                    "Metric": "quantile_loss",
                }
            )

            log.debug(
                f"Quantile loss for {method}/{variable} at q={quantile}: {mean_loss:.6f}"
            )

        # Process categorical variables (log loss doesn't use quantiles, but we compute at each for consistency)
        # Note: Models should ideally provide predicted probabilities for categorical variables
        # instead of class labels for more accurate log loss computation
        categorical_losses = []
        for variable in categorical_vars:
            # Validate variable exists
            if variable not in imputation[quantile].columns:
                error_msg = f"Variable {variable} not found in imputation results for method {method}"
                log.error(error_msg)
                raise ValueError(error_msg)

            # Get values
            test_values = test_y[variable].values
            pred_values = imputation[quantile][variable].values

            # Get unique labels from test data
            labels = np.unique(test_values)

            # Compute loss
            # Note: If pred_values contains class labels instead of probabilities,
            # they will be converted with a warning
            _, mean_loss = compute_loss(
                test_values, pred_values, "log_loss", labels=labels
            )
            categorical_losses.append(mean_loss)

            # Add variable-specific result
            results.append(
                {
                    "Method": method,
                    "Imputed Variable": variable,
                    "Percentile": quantile,
                    "Loss": mean_loss,
                    "Metric": "log_loss",
                }
            )

            log.debug(
                f"Log loss for {method}/{variable} at q={quantile}: {mean_loss:.6f} (note that log loss does not depend on quantile and should remain constant across them)"
            )

        # Add average for quantile loss variables at this quantile
        if quantile_losses:
            avg_quantile_loss = np.mean(quantile_losses)
            results.append(
                {
                    "Method": method,
                    "Imputed Variable": "mean_quantile_loss",
                    "Percentile": quantile,
                    "Loss": avg_quantile_loss,
                    "Metric": "quantile_loss",
                }
            )

        # Add average for categorical variables at this quantile
        if categorical_losses:
            avg_categorical_loss = np.mean(categorical_losses)
            results.append(
                {
                    "Method": method,
                    "Imputed Variable": "mean_log_loss",
                    "Percentile": quantile,
                    "Loss": avg_categorical_loss,
                    "Metric": "log_loss",
                }
            )

    # Add overall average across all quantiles for quantile loss variables
    all_quantile_losses = [
        r["Loss"]
        for r in results
        if r["Imputed Variable"] == "mean_quantile_loss"
        and r["Percentile"] != "mean_loss"
    ]
    if all_quantile_losses:
        avg_quant_loss = np.mean(all_quantile_losses)
        results.append(
            {
                "Method": method,
                "Imputed Variable": "mean_quantile_loss",
                "Percentile": "mean_loss",
                "Loss": avg_quant_loss,
                "Metric": "quantile_loss",
            }
        )

    # Add overall average across all quantiles for log loss variables
    all_categorical_losses = [
        r["Loss"]
        for r in results
        if r["Imputed Variable"] == "mean_log_loss"
        and r["Percentile"] != "mean_loss"
    ]
    if all_categorical_losses:
        avg_cat_loss = np.mean(all_categorical_losses)
        results.append(
            {
                "Method": method,
                "Imputed Variable": "mean_log_loss",
                "Percentile": "mean_loss",
                "Loss": avg_cat_loss,
                "Metric": "log_loss",
            }
        )

    return results


@validate_call(config=VALIDATE_CONFIG)
def compare_metrics(
    test_y: pd.DataFrame,
    method_imputations: Dict[str, Dict[float, pd.DataFrame]],
    imputed_variables: List[str],
) -> pd.DataFrame:
    """Compare metrics across different imputation methods.

    Automatically detects which metric to use for each variable based on its type.

    Args:
        test_y: DataFrame containing true values.
        method_imputations: Nested dictionary mapping method names
            to dictionaries mapping quantiles to imputation values.
        imputed_variables: List of variables to evaluate.

    Returns:
        pd.DataFrame: Results dataframe with columns 'Method', 'Imputed Variable',
            'Percentile', 'Loss', and 'Metric' containing the metrics for each
            method, variable, and percentile.

    Raises:
        ValueError: If input data formats are invalid.
        RuntimeError: If comparison operation fails.
    """
    try:
        log.info(
            f"Comparing metrics for {len(method_imputations)} methods: {list(method_imputations.keys())}"
        )
        log.info(f"Using {len(QUANTILES)} quantiles: {QUANTILES}")
        log.info(f"True values shape: {test_y.shape}")

        # Validate inputs
        validate_columns_exist(test_y, imputed_variables, "test_y")

        # Detect metric type for each variable
        variable_metrics = {}
        for var in imputed_variables:
            metric_type = get_metric_for_variable_type(test_y[var], var)
            variable_metrics[var] = metric_type
            log.info(f"Variable '{var}' will use metric: {metric_type}")

        # Collect all results in a list first
        all_results = []

        # Process each method
        for method, imputation in method_imputations.items():
            method_results = _compute_method_losses(
                method,
                imputation,
                test_y,
                imputed_variables,
                QUANTILES,
                variable_metrics,
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
        log.error(f"Error in metrics comparison: {str(e)}")
        raise RuntimeError(f"Failed to compare metrics: {str(e)}") from e


def kl_divergence(
    donor_values: np.ndarray,
    receiver_values: np.ndarray,
    donor_weights: Optional[np.ndarray] = None,
    receiver_weights: Optional[np.ndarray] = None,
) -> float:
    """Calculate Kullback-Leibler (KL) Divergence between two categorical distributions.

    KL divergence measures the difference between two probability distributions.
    For categorical variables, it is calculated as:
    KL(P||Q) = sum(P(x) * log(P(x) / Q(x))) for all categories x

    This implementation uses the donor distribution as P (reference) and
    receiver distribution as Q (approximation), measuring how well the
    receiver distribution approximates the donor distribution.

    Args:
        donor_values: Array of categorical values from donor data (reference distribution P).
        receiver_values: Array of categorical values from receiver data (approximation Q).
        donor_weights: Optional weights for donor values. If provided, computes
            weighted probability distribution.
        receiver_weights: Optional weights for receiver values. If provided,
            computes weighted probability distribution.

    Returns:
        KL divergence value >= 0, where 0 indicates identical distributions
        and larger values indicate greater divergence. Note: KL divergence is
        unbounded and can be infinite if Q(x) = 0 for some x where P(x) > 0.

    Raises:
        ValueError: If inputs are empty or invalid.

    Note:
        - KL divergence is not symmetric: KL(P||Q) != KL(Q||P)
        - To handle zero probabilities, a small epsilon is added to avoid log(0)
        - Uses scipy.special.rel_entr for numerical stability
    """
    if len(donor_values) == 0 or len(receiver_values) == 0:
        raise ValueError(
            "Both donor and receiver values must be non-empty arrays"
        )

    # Get all unique categories from both distributions
    all_categories = np.union1d(
        np.unique(donor_values), np.unique(receiver_values)
    )

    # Calculate probability distributions (weighted if weights provided)
    if donor_weights is not None:
        # Compute weighted probabilities
        donor_df = pd.DataFrame(
            {"value": donor_values, "weight": donor_weights}
        )
        donor_grouped = donor_df.groupby("value")["weight"].sum()
        donor_total = donor_grouped.sum()
        donor_counts = donor_grouped / donor_total
    else:
        donor_counts = pd.Series(donor_values).value_counts(normalize=True)

    if receiver_weights is not None:
        # Compute weighted probabilities
        receiver_df = pd.DataFrame(
            {"value": receiver_values, "weight": receiver_weights}
        )
        receiver_grouped = receiver_df.groupby("value")["weight"].sum()
        receiver_total = receiver_grouped.sum()
        receiver_counts = receiver_grouped / receiver_total
    else:
        receiver_counts = pd.Series(receiver_values).value_counts(
            normalize=True
        )

    # Create probability arrays for all categories
    p_donor = np.array([donor_counts.get(cat, 0.0) for cat in all_categories])
    q_receiver = np.array(
        [receiver_counts.get(cat, 0.0) for cat in all_categories]
    )

    # Add small epsilon to avoid log(0) and division by zero
    epsilon = 1e-10
    q_receiver = np.maximum(q_receiver, epsilon)

    # Calculate KL divergence using scipy.special.kl_div
    # kl_div(p, q) computes p * log(p/q) element-wise
    kl_values = rel_entr(p_donor, q_receiver)

    # Sum over all categories to get total KL divergence
    return np.sum(kl_values)


@validate_call(config=VALIDATE_CONFIG)
def compare_distributions(
    donor_data: pd.DataFrame,
    receiver_data: pd.DataFrame,
    imputed_variables: List[str],
    donor_weights: Optional[Union[pd.Series, np.ndarray]] = None,
    receiver_weights: Optional[Union[pd.Series, np.ndarray]] = None,
) -> pd.DataFrame:
    """Compare distributions between donor and receiver data for imputed variables.

    Evaluates distributional similarity using appropriate metrics:
    - Wasserstein Distance for numerical variables
    - KL Divergence for categorical variables

    Args:
        donor_data: DataFrame containing original donor data.
        receiver_data: DataFrame containing receiver data with imputations.
        imputed_variables: List of variable names to compare.
        donor_weights: Optional array or Series of sample weights for donor data.
            Must have same length as donor_data.
        receiver_weights: Optional array or Series of sample weights for receiver
            data. Must have same length as receiver_data.

    Returns:
        DataFrame with columns 'Variable', 'Metric', and 'Distance' containing
        the distributional similarity metrics for each variable.

    Raises:
        ValueError: If variables don't exist in both DataFrames or if data is invalid.
        RuntimeError: If distribution comparison fails.

    Example:
        >>> donor_df = pd.DataFrame({'income': [1000, 2000, 3000],
        ...                          'region': ['A', 'B', 'A']})
        >>> receiver_df = pd.DataFrame({'income': [1100, 1900, 3100],
        ...                             'region': ['A', 'A', 'B']})
        >>> result = compare_distributions(donor_df, receiver_df,
        ...                                ['income', 'region'])
        >>> print(result)
           Variable                 Metric  Distance
        0    income  wasserstein_distance  66.666667
        1    region          kl_divergence    0.166667
    """
    try:
        log.info(
            f"Comparing distributions for {len(imputed_variables)} variables"
        )
        log.info(f"Donor data shape: {donor_data.shape}")
        log.info(f"Receiver data shape: {receiver_data.shape}")

        # Validate inputs
        validate_columns_exist(donor_data, imputed_variables, "donor_data")
        validate_columns_exist(
            receiver_data, imputed_variables, "receiver_data"
        )

        # Convert weights to numpy arrays if provided
        donor_weights_arr = None
        receiver_weights_arr = None
        if donor_weights is not None:
            donor_weights_arr = np.asarray(donor_weights)
            if len(donor_weights_arr) != len(donor_data):
                raise ValueError(
                    f"donor_weights length ({len(donor_weights_arr)}) must match "
                    f"donor_data length ({len(donor_data)})"
                )
        if receiver_weights is not None:
            receiver_weights_arr = np.asarray(receiver_weights)
            if len(receiver_weights_arr) != len(receiver_data):
                raise ValueError(
                    f"receiver_weights length ({len(receiver_weights_arr)}) must "
                    f"match receiver_data length ({len(receiver_data)})"
                )

        results = []

        # Detect metric type and compute distance for each variable
        detector = VariableTypeDetector()
        for var in imputed_variables:
            donor_values = donor_data[var].values
            receiver_values = receiver_data[var].values

            # Check for null values - these are not allowed when comparing
            if np.any(pd.isna(donor_values)):
                raise ValueError(
                    f"Variable '{var}' in donor_data contains null values. "
                    "Please remove or impute null values before comparing "
                    "distributions."
                )
            if np.any(pd.isna(receiver_values)):
                raise ValueError(
                    f"Variable '{var}' in receiver_data contains null values. "
                    "Please remove or impute null values before comparing "
                    "distributions."
                )

            if len(donor_values) == 0 or len(receiver_values) == 0:
                log.warning(
                    f"Skipping variable '{var}' due to insufficient data "
                    f"(donor: {len(donor_values)}, receiver: {len(receiver_values)})"
                )
                continue

            # Detect variable type using donor data
            var_type, _ = detector.categorize_variable(
                donor_data[var], var, log
            )

            # Choose appropriate metric
            if var_type in ["bool", "categorical", "numeric_categorical"]:
                # Use KL Divergence for categorical
                metric_name = "kl_divergence"
                distance = kl_divergence(
                    donor_values,
                    receiver_values,
                    donor_weights=donor_weights_arr,
                    receiver_weights=receiver_weights_arr,
                )
                log.debug(
                    f"KL divergence for categorical variable '{var}': {distance:.6f}"
                )
            else:
                # Use Wasserstein Distance for numerical
                metric_name = "wasserstein_distance"
                distance = wasserstein_distance(
                    donor_values,
                    receiver_values,
                    u_weights=donor_weights_arr,
                    v_weights=receiver_weights_arr,
                )
                log.debug(
                    f"Wasserstein distance for numerical variable '{var}': {distance:.6f}"
                )

            results.append(
                {
                    "Variable": var,
                    "Metric": metric_name,
                    "Distance": distance,
                }
            )

        if not results:
            raise ValueError(
                "No valid distribution comparisons could be computed. "
                "Check that variables have sufficient non-null data."
            )

        results_df = pd.DataFrame(results)
        log.info(
            f"Distribution comparison complete. Computed {len(results_df)} metrics."
        )

        return results_df

    except ValueError as e:
        # Re-raise validation errors
        raise e
    except Exception as e:
        log.error(f"Error comparing distributions: {str(e)}")
        raise RuntimeError(f"Failed to compare distributions: {str(e)}") from e
