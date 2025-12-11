"""Cross-validation utilities with dual metric support for imputation model evaluation.

This module provides functions for evaluating imputation models using k-fold
cross-validation with separate quantile loss and log loss metrics.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import joblib
import numpy as np
import pandas as pd
from pydantic import validate_call
from sklearn.model_selection import KFold

from microimpute.comparisons.metrics import (
    compute_loss,
    get_metric_for_variable_type,
)
from microimpute.comparisons.validation import (
    validate_columns_exist,
    validate_quantiles,
)
from microimpute.config import QUANTILES, RANDOM_STATE, VALIDATE_CONFIG

try:
    from microimpute.models.matching import Matching
except ImportError:  # optional dependency
    Matching = None
from microimpute.models.quantreg import QuantReg

log = logging.getLogger(__name__)


def _process_single_fold(
    fold_idx_pair: Tuple[int, Tuple[np.ndarray, np.ndarray]],
    data: pd.DataFrame,
    model_class: Type,
    predictors: List[str],
    imputed_variables: List[str],
    weight_col: Optional[str],
    quantiles: List[float],
    model_hyperparams: Optional[dict],
    tune_hyperparameters: bool,
    variable_metrics: Dict[str, str],
) -> Tuple[
    int,
    Dict,
    Dict,
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Optional[dict],
]:
    """Process a single CV fold and return results organized by variable."""
    fold_idx, (train_idx, test_idx) = fold_idx_pair
    log.info(f"Processing fold {fold_idx+1}")

    # Split data for this fold
    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]

    # Store actual values for this fold organized by variable
    train_y = {var: train_data[var].values for var in imputed_variables}
    test_y = {var: test_data[var].values for var in imputed_variables}

    # Instantiate and fit the model
    model = model_class()
    fold_tuned_params = None

    # Fit model with appropriate parameters
    fitted_model, fold_tuned_params = _fit_model_for_fold(
        model,
        model_class,
        train_data,
        predictors,
        imputed_variables,
        weight_col,
        quantiles,
        model_hyperparams,
        tune_hyperparameters,
    )

    # Check if model fitting failed (incompatible with variable types)
    if fitted_model is None:
        log.info(
            f"Model {model_class.__name__} incompatible with variable types, skipping fold"
        )
        return fold_idx, None, None, test_y, train_y, None

    # Check if we need to use return_probs for categorical variables
    has_categorical = any(
        variable_metrics.get(var) == "log_loss" for var in imputed_variables
    )

    # Get predictions for this fold
    log.info(f"Generating predictions for train and test data")
    if has_categorical:
        # Use return_probs=True for categorical predictions
        fold_test_imputations = fitted_model.predict(
            test_data, quantiles, return_probs=True
        )
        fold_train_imputations = fitted_model.predict(
            train_data, quantiles, return_probs=True
        )
    else:
        fold_test_imputations = fitted_model.predict(test_data, quantiles)
        fold_train_imputations = fitted_model.predict(train_data, quantiles)

    return (
        fold_idx,
        fold_test_imputations,
        fold_train_imputations,
        test_y,
        train_y,
        fold_tuned_params,
    )


def _fit_model_for_fold(
    model: Any,
    model_class: Type,
    train_data: pd.DataFrame,
    predictors: List[str],
    imputed_variables: List[str],
    weight_col: Optional[str],
    quantiles: List[float],
    model_hyperparams: Optional[dict],
    tune_hyperparameters: bool,
) -> Tuple[Any, Optional[dict]]:
    """Fit a model for a single fold with appropriate parameters.

    Returns None for fitted_model if the model cannot handle the variable types.
    """
    model_name = model_class.__name__
    fold_tuned_params = None

    # Special handling for QuantReg with categorical variables
    if model_name == "QuantReg":
        # Check if any imputed variables are categorical
        from microimpute.comparisons.metrics import (
            get_metric_for_variable_type,
        )

        for var in imputed_variables:
            if (
                get_metric_for_variable_type(train_data[var], var)
                == "log_loss"
            ):
                log.warning(
                    f"QuantReg does not support categorical variable '{var}'. "
                    f"Skipping QuantReg for this fold."
                )
                return None, None

    # Handle model-specific hyperparameters
    if model_hyperparams:
        try:
            log.info(
                f"Fitting {model_name} with hyperparameters: {model_hyperparams}"
            )
            fitted_model = model.fit(
                X_train=train_data,
                predictors=predictors,
                imputed_variables=imputed_variables,
                weight_col=weight_col,
                **model_hyperparams,
            )
        except ValueError as e:
            # Check if it's due to categorical incompatibility
            if "QuantReg does not support categorical" in str(e):
                log.warning(
                    f"{model_name} incompatible with variable types: {str(e)}"
                )
                return None, None
            raise e
        except TypeError as e:
            log.warning(
                f"Invalid hyperparameters for {model_name}, using defaults: {str(e)}"
            )
            fitted_model = model.fit(
                X_train=train_data,
                predictors=predictors,
                imputed_variables=imputed_variables,
                weight_col=weight_col,
            )
            raise ValueError(
                f"Invalid hyperparameters for {model_name}"
            ) from e

    # Handle QuantReg which needs explicit quantiles
    elif model_class == QuantReg:
        try:
            log.info(f"Fitting QuantReg model with explicit quantiles")
            fitted_model = model.fit(
                train_data,
                predictors,
                imputed_variables,
                weight_col=weight_col,
                quantiles=quantiles,
            )
        except ValueError as e:
            if "QuantReg does not support categorical" in str(e):
                log.warning(
                    f"QuantReg incompatible with variable types: {str(e)}"
                )
                return None, None
            raise e

    # Handle hyperparameter tuning for QRF, Matching, and MDN
    elif tune_hyperparameters and model_name in ["QRF", "Matching", "MDN"]:
        log.info(f"Tuning {model_name} hyperparameters during fitting")
        fitted_model, fold_tuned_params = model.fit(
            train_data,
            predictors,
            imputed_variables,
            weight_col=weight_col,
            tune_hyperparameters=True,
        )

    # Default fitting
    else:
        try:
            log.info(f"Fitting {model_name} model with default parameters")
            fitted_model = model.fit(
                train_data,
                predictors,
                imputed_variables,
                weight_col=weight_col,
            )
        except ValueError as e:
            if (
                "QuantReg does not support categorical" in str(e)
                and model_name == "QuantReg"
            ):
                log.warning(
                    f"QuantReg incompatible with variable types: {str(e)}"
                )
                return None, None
            raise e

    return fitted_model, fold_tuned_params


def _compute_fold_loss_by_metric(
    fold_idx: int,
    quantile: float,
    test_y_values: Dict[str, List[np.ndarray]],
    train_y_values: Dict[str, List[np.ndarray]],
    test_results: Dict[float, List],
    train_results: Dict[float, List],
    variable_metrics: Dict[str, str],
    imputed_variables: List[str],
    test_probabilities: Dict[str, List] = None,
    train_probabilities: Dict[str, List] = None,
) -> Dict[str, Any]:
    """Compute loss for a specific fold and quantile, separated by metric type."""
    result = {
        "fold": fold_idx,
        "quantile": quantile,
        "quantile_loss": {"test": None, "train": None, "variables": []},
        "log_loss": {"test": None, "train": None, "variables": []},
    }

    # Separate variables by metric type
    for var in imputed_variables:
        metric_type = variable_metrics[var]

        # Get data for this variable
        test_y_var = test_y_values[var][fold_idx]
        train_y_var = train_y_values[var][fold_idx]
        test_pred_var = test_results[quantile][fold_idx][var].values
        train_pred_var = train_results[quantile][fold_idx][var].values

        # Compute loss based on metric type
        if metric_type == "quantile_loss":
            _, test_loss = compute_loss(
                test_y_var, test_pred_var, "quantile_loss", q=quantile
            )
            _, train_loss = compute_loss(
                train_y_var, train_pred_var, "quantile_loss", q=quantile
            )

            if result["quantile_loss"]["test"] is None:
                result["quantile_loss"]["test"] = []
                result["quantile_loss"]["train"] = []

            result["quantile_loss"]["test"].append(test_loss)
            result["quantile_loss"]["train"].append(train_loss)
            result["quantile_loss"]["variables"].append(var)

        else:  # log_loss
            # Use probabilities if available, otherwise use class predictions
            if (
                test_probabilities
                and test_probabilities[var][fold_idx] is not None
            ):
                # Get probabilities and classes for this variable
                test_prob_info = test_probabilities[var][fold_idx]
                train_prob_info = train_probabilities[var][fold_idx]

                if (
                    isinstance(test_prob_info, dict)
                    and "probabilities" in test_prob_info
                ):
                    # Extract probabilities and classes
                    test_probs = test_prob_info["probabilities"]
                    train_probs = train_prob_info["probabilities"]
                    model_classes = test_prob_info["classes"]

                    # Import the ordering function
                    from microimpute.comparisons.metrics import (
                        order_probabilities_alphabetically,
                    )

                    # Order probabilities alphabetically
                    test_probs_ordered, alphabetical_labels = (
                        order_probabilities_alphabetically(
                            test_probs, model_classes
                        )
                    )
                    train_probs_ordered, _ = (
                        order_probabilities_alphabetically(
                            train_probs, model_classes
                        )
                    )

                    # Compute log loss with properly ordered probabilities
                    _, test_loss = compute_loss(
                        test_y_var,
                        test_probs_ordered,
                        "log_loss",
                        labels=alphabetical_labels,
                    )
                    _, train_loss = compute_loss(
                        train_y_var,
                        train_probs_ordered,
                        "log_loss",
                        labels=alphabetical_labels,
                    )
                else:
                    # Fallback for old format or if probabilities not available
                    log.warning(
                        f"Probabilities not in expected format for variable {var}, using class predictions"
                    )
                    labels = np.unique(
                        np.concatenate([test_y_var, train_y_var])
                    )
                    labels = np.sort(labels)  # Ensure alphabetical order
                    _, test_loss = compute_loss(
                        test_y_var, test_pred_var, "log_loss", labels=labels
                    )
                    _, train_loss = compute_loss(
                        train_y_var, train_pred_var, "log_loss", labels=labels
                    )
            else:
                # Fall back to using class predictions (less accurate)
                labels = np.unique(np.concatenate([test_y_var, train_y_var]))
                labels = np.sort(labels)  # Ensure alphabetical order
                _, test_loss = compute_loss(
                    test_y_var, test_pred_var, "log_loss", labels=labels
                )
                _, train_loss = compute_loss(
                    train_y_var, train_pred_var, "log_loss", labels=labels
                )

            if result["log_loss"]["test"] is None:
                result["log_loss"]["test"] = []
                result["log_loss"]["train"] = []

            result["log_loss"]["test"].append(test_loss)
            result["log_loss"]["train"].append(train_loss)
            result["log_loss"]["variables"].append(var)

    # Average losses for each metric type
    for metric_type in ["quantile_loss", "log_loss"]:
        if result[metric_type]["test"] is not None:
            result[metric_type]["test"] = np.mean(result[metric_type]["test"])
            result[metric_type]["train"] = np.mean(
                result[metric_type]["train"]
            )
        else:
            # No variables of this type
            result[metric_type]["test"] = np.nan
            result[metric_type]["train"] = np.nan

    return result


def _compute_losses_parallel(
    test_y_values: Dict[str, List[np.ndarray]],
    train_y_values: Dict[str, List[np.ndarray]],
    test_results: Dict[float, List],
    train_results: Dict[float, List],
    quantiles: List[float],
    variable_metrics: Dict[str, str],
    imputed_variables: List[str],
    n_jobs: int,
    test_probabilities: Dict[str, List] = None,
    train_probabilities: Dict[str, List] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute losses in parallel for all folds and quantiles, separated by metric type."""
    n_folds = len(next(iter(test_y_values.values())))
    loss_tasks = [(k, q) for k in range(n_folds) for q in quantiles]

    # Only parallelize if worthwhile
    if len(loss_tasks) > 10 and n_jobs != 1:
        with joblib.Parallel(n_jobs=n_jobs) as parallel:
            loss_results = parallel(
                joblib.delayed(_compute_fold_loss_by_metric)(
                    fold_idx,
                    q,
                    test_y_values,
                    train_y_values,
                    test_results,
                    train_results,
                    variable_metrics,
                    imputed_variables,
                    test_probabilities,
                    train_probabilities,
                )
                for fold_idx, q in loss_tasks
            )
    else:
        # Sequential computation for smaller tasks
        loss_results = [
            _compute_fold_loss_by_metric(
                fold_idx,
                q,
                test_y_values,
                train_y_values,
                test_results,
                train_results,
                variable_metrics,
                imputed_variables,
                test_probabilities,
                train_probabilities,
            )
            for fold_idx, q in loss_tasks
        ]

    # Organize results by metric type
    results = {
        "quantile_loss": {
            "test": {q: [] for q in quantiles},
            "train": {q: [] for q in quantiles},
            "variables": [],
        },
        "log_loss": {
            "test": {q: [] for q in quantiles},
            "train": {q: [] for q in quantiles},
            "variables": [],
        },
    }

    # Process results
    for result in loss_results:
        q = result["quantile"]
        fold_idx = result["fold"]

        for metric_type in ["quantile_loss", "log_loss"]:
            if not np.isnan(result[metric_type]["test"]):
                results[metric_type]["test"][q].append(
                    result[metric_type]["test"]
                )
                results[metric_type]["train"][q].append(
                    result[metric_type]["train"]
                )

                # Store variable list (only once)
                if fold_idx == 0 and q == quantiles[0]:
                    results[metric_type]["variables"] = result[metric_type][
                        "variables"
                    ]

    return results


@validate_call(config=VALIDATE_CONFIG)
def cross_validate_model(
    model_class: Type,
    data: pd.DataFrame,
    predictors: List[str],
    imputed_variables: List[str],
    weight_col: Optional[str] = None,
    quantiles: Optional[List[float]] = QUANTILES,
    n_splits: Optional[int] = 5,
    random_state: Optional[int] = RANDOM_STATE,
    model_hyperparams: Optional[dict] = None,
    tune_hyperparameters: Optional[bool] = False,
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict]]:
    """Perform cross-validation with dual metric support.

    Returns:
        Dictionary containing separate results for quantile_loss and log_loss:
        {
            "quantile_loss": {
                "results": pd.DataFrame,  # rows: ["train", "test"], cols: quantiles
                "mean_train": float,
                "mean_test": float,
                "variables": List[str]
            },
            "log_loss": {
                "results": pd.DataFrame,  # rows: ["train", "test"], cols: quantiles (constant values)
                "mean_train": float,
                "mean_test": float,
                "variables": List[str]
            }
        }
        If tune_hyperparameters is True, returns tuple of (results_dict, best_hyperparameters).
    """
    # Use shared validation utilities
    validate_columns_exist(data, predictors, "data")
    validate_columns_exist(data, imputed_variables, "data")
    if weight_col:
        validate_columns_exist(data, [weight_col], "data")
    if quantiles:
        validate_quantiles(quantiles)

    # Set up parallel processing
    n_jobs = 1 if (Matching is not None and model_class == Matching) else -1

    try:
        log.info(
            f"Starting {n_splits}-fold cross-validation for {model_class.__name__}"
        )
        log.info(f"Evaluating at {len(quantiles)} quantiles: {quantiles}")

        # Detect variable types
        variable_metrics = {}
        for var in imputed_variables:
            metric_type = get_metric_for_variable_type(data[var], var)
            variable_metrics[var] = (
                "quantile_loss"
                if metric_type == "quantile_loss"
                else "log_loss"
            )
            log.info(
                f"Variable '{var}' will use metric: {variable_metrics[var]}"
            )

        # Set up k-fold cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_indices = list(kf.split(data))

        # Execute folds in parallel
        with joblib.Parallel(n_jobs=n_jobs, verbose=10) as parallel:
            fold_results = parallel(
                joblib.delayed(_process_single_fold)(
                    (i, fold_pair),
                    data,
                    model_class,
                    predictors,
                    imputed_variables,
                    weight_col,
                    quantiles,
                    model_hyperparams,
                    tune_hyperparameters,
                    variable_metrics,
                )
                for i, fold_pair in enumerate(fold_indices)
            )

        # Filter out None results (from incompatible model-variable combinations)
        valid_fold_results = [r for r in fold_results if r[1] is not None]

        if not valid_fold_results:
            # Model cannot handle any of the variables
            log.warning(
                f"{model_class.__name__} cannot handle the provided variable types. "
                f"Returning NaN results."
            )
            # Return empty results structure
            return {
                "quantile_loss": {
                    "results": pd.DataFrame(),  # Empty DataFrame
                    "mean_train": np.nan,
                    "mean_test": np.nan,
                    "variables": [],
                },
                "log_loss": {
                    "results": pd.DataFrame(),  # Empty DataFrame
                    "mean_train": np.nan,
                    "mean_test": np.nan,
                    "variables": [],
                },
            }

        # Sort valid results by fold index
        valid_fold_results.sort(key=lambda x: x[0])

        # Extract and organize results
        test_results = {q: [] for q in quantiles}
        train_results = {q: [] for q in quantiles}
        test_y_values = {var: [] for var in imputed_variables}
        train_y_values = {var: [] for var in imputed_variables}
        # Store probabilities separately for categorical variables
        test_probabilities = {var: [] for var in imputed_variables}
        train_probabilities = {var: [] for var in imputed_variables}
        tuned_hyperparameters = {}

        for (
            fold_idx,
            fold_test_imp,
            fold_train_imp,
            test_y,
            train_y,
            fold_tuned_params,
        ) in valid_fold_results:
            for var in imputed_variables:
                test_y_values[var].append(test_y[var])
                train_y_values[var].append(train_y[var])

            if tune_hyperparameters and fold_tuned_params:
                tuned_hyperparameters[fold_idx] = fold_tuned_params

            # Extract probabilities if available (for categorical variables)
            if "probabilities" in fold_test_imp:
                for var in imputed_variables:
                    if var in fold_test_imp["probabilities"]:
                        test_probabilities[var].append(
                            fold_test_imp["probabilities"][var]
                        )
                        train_probabilities[var].append(
                            fold_train_imp["probabilities"][var]
                        )
                    else:
                        # Not a categorical variable, no probabilities
                        test_probabilities[var].append(None)
                        train_probabilities[var].append(None)
            else:
                # No probabilities returned (all numerical variables)
                for var in imputed_variables:
                    test_probabilities[var].append(None)
                    train_probabilities[var].append(None)

            for q in quantiles:
                test_results[q].append(fold_test_imp[q])
                train_results[q].append(fold_train_imp[q])

        # Compute losses with dual metrics
        metric_results = _compute_losses_parallel(
            test_y_values,
            train_y_values,
            test_results,
            train_results,
            quantiles,
            variable_metrics,
            imputed_variables,
            n_jobs,
            test_probabilities,
            train_probabilities,
        )

        # Create structured results
        final_results = {}

        for metric_type in ["quantile_loss", "log_loss"]:
            if metric_results[metric_type]["variables"]:
                # Create a single DataFrame with train and test as rows
                # This matches the original format and is more convenient
                combined_df = pd.DataFrame(
                    [
                        {
                            q: np.mean(values)
                            for q, values in metric_results[metric_type][
                                "train"
                            ].items()
                        },
                        {
                            q: np.mean(values)
                            for q, values in metric_results[metric_type][
                                "test"
                            ].items()
                        },
                    ],
                    index=["train", "test"],
                )

                # Calculate means
                mean_test = combined_df.loc["test"].mean()
                mean_train = combined_df.loc["train"].mean()

                final_results[metric_type] = {
                    "results": combined_df,  # Single DataFrame with train/test rows
                    "mean_train": mean_train,
                    "mean_test": mean_test,
                    "variables": metric_results[metric_type]["variables"],
                }

                log.info(
                    f"{metric_type} - Mean Train: {mean_train:.6f}, Mean Test: {mean_test:.6f}"
                )
            else:
                # No variables use this metric
                final_results[metric_type] = {
                    "results": pd.DataFrame(),  # Empty DataFrame
                    "mean_train": np.nan,
                    "mean_test": np.nan,
                    "variables": [],
                }

        # Return results with optional hyperparameters
        if tune_hyperparameters and tuned_hyperparameters:
            # Select best hyperparameters based on primary metric
            primary_metric = (
                "quantile_loss"
                if len(final_results["quantile_loss"]["variables"])
                >= len(final_results["log_loss"]["variables"])
                else "log_loss"
            )

            # Use median quantile (0.5) for selection
            best_fold = 0
            best_loss = float("inf")

            if 0.5 in quantiles:
                for fold_idx in range(n_splits):
                    fold_loss = metric_results[primary_metric]["test"][0.5][
                        fold_idx
                    ]
                    if fold_loss < best_loss:
                        best_loss = fold_loss
                        best_fold = fold_idx

            best_hyperparams = tuned_hyperparameters.get(best_fold)
            return final_results, best_hyperparams
        else:
            return final_results

    except ValueError as e:
        raise e
    except (KeyError, TypeError, AttributeError, ImportError) as e:
        log.error(f"Error during cross-validation: {str(e)}")
        raise RuntimeError(f"Cross-validation failed: {str(e)}") from e
