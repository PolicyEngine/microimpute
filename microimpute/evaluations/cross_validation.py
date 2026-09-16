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
from microimpute.utils.data import (
    preprocess_data,
    apply_transformations,
    reverse_transformations,
)
from microimpute.utils.type_handling import declare_target_types

try:
    from microimpute.models.matching import Matching
except ImportError:  # optional dependency
    Matching = None
from microimpute.models.quantreg import QuantReg
from microimpute.models.imputer import create_distributional_model

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
    preprocessing: Optional[Dict[str, str]] = None,
    target_types: Optional[Dict[str, str]] = None,
    random_state: int = RANDOM_STATE,
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
    log.info(f"Processing fold {fold_idx + 1}")

    # Split data for this fold
    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]

    # Store actual values for this fold organized by variable
    train_y = {var: train_data[var].values for var in imputed_variables}
    test_y = {var: test_data[var].values for var in imputed_variables}

    transform_params = {}
    if preprocessing:
        train_data, transform_params = preprocess_data(
            train_data, full_data=True, **_preprocessing_kwargs(preprocessing)
        )
        test_data = apply_transformations(test_data, transform_params)
    # Instantiate with the caller's seed for fitting and prediction sampling.
    model = create_distributional_model(model_class, seed=random_state)
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
        target_types,
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

    if has_categorical:
        for predictions, frame in [
            (fold_test_imputations, test_data),
            (fold_train_imputations, train_data),
        ]:
            probabilities = predictions.setdefault("probabilities", {})
            for variable, info in model.constant_targets.items():
                if variable_metrics[variable] == "log_loss":
                    probabilities[variable] = {
                        "probabilities": np.ones((len(frame), 1)),
                        "classes": np.asarray([info["value"]]),
                    }
    if transform_params:
        for predictions in [fold_test_imputations, fold_train_imputations]:
            for quantile in quantiles:
                predictions[quantile] = reverse_transformations(
                    predictions[quantile], transform_params
                )

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
    target_types: Optional[Dict[str, str]] = None,
) -> Tuple[Any, Optional[dict]]:
    """Fit a model for a single fold with appropriate parameters.

    Returns None for fitted_model if the model cannot handle the variable types.
    """
    model_name = model_class.__name__
    if model_name == "Matching":
        log.warning(
            "Matching provides donor samples, not quantiles or class probabilities; skipping distributional scoring"
        )
        return None, None
    metric_types = {
        var: ("quantile_loss" if target_types[var] == "numeric" else "log_loss")
        if target_types and var in target_types
        else get_metric_for_variable_type(train_data[var], var)
        for var in imputed_variables
    }
    if model_name == "QuantReg" and "log_loss" in metric_types.values():
        log.warning("QuantReg does not support categorical targets; skipping")
        return None, None
    params = dict(model_hyperparams or {})
    params["target_types"] = target_types
    if model_name == "QuantReg":
        params["quantiles"] = quantiles
    if tune_hyperparameters and model_name in ["QRF", "MDN"]:
        params["tune_hyperparameters"] = True
    fitted = model.fit(
        train_data, predictors, imputed_variables, weight_col=weight_col, **params
    )
    if isinstance(fitted, tuple):
        return fitted
    return fitted, None


def _preprocessing_kwargs(preprocessing: Dict[str, str]) -> dict:
    valid = {
        "normalize": "normalize",
        "log": "log_transform",
        "asinh": "asinh_transform",
    }
    if set(preprocessing.values()) - set(valid):
        raise ValueError("Unknown preprocessing transformation")
    return {
        argument: [col for col, transform in preprocessing.items() if transform == name]
        or False
        for name, argument in valid.items()
    }


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

        # Get data for this variable, converting to numpy to handle
        # Arrow-backed dtypes (e.g. ArrowStringArray) that pydantic
        # won't accept as np.ndarray.
        test_y_var = np.asarray(test_y_values[var][fold_idx])
        train_y_var = np.asarray(train_y_values[var][fold_idx])
        test_pred_var = np.asarray(test_results[quantile][fold_idx][var])
        train_pred_var = np.asarray(train_results[quantile][fold_idx][var])

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
            if not test_probabilities or test_probabilities[var][fold_idx] is None:
                raise ValueError(
                    f"Log loss for '{var}' requires predicted probabilities"
                )
            losses = []
            labels = np.unique(np.concatenate([test_y_var, train_y_var]))
            for truth, info in [
                (test_y_var, test_probabilities[var][fold_idx]),
                (train_y_var, train_probabilities[var][fold_idx]),
            ]:
                classes = np.asarray(info["classes"])
                all_labels = np.union1d(labels, classes)
                probabilities = np.zeros((len(truth), len(all_labels)))
                for idx, label in enumerate(classes):
                    probabilities[:, np.flatnonzero(all_labels == label)[0]] = (
                        np.asarray(info["probabilities"])[:, idx]
                    )
                if len(all_labels) == 1:
                    losses.append(0.0)
                else:
                    losses.append(
                        compute_loss(
                            truth, probabilities, "log_loss", labels=all_labels
                        )[1]
                    )
            test_loss, train_loss = losses

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
            result[metric_type]["train"] = np.mean(result[metric_type]["train"])
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
                results[metric_type]["test"][q].append(result[metric_type]["test"])
                results[metric_type]["train"][q].append(result[metric_type]["train"])

                # Store variable list (only once)
                if fold_idx == 0 and q == quantiles[0]:
                    results[metric_type]["variables"] = result[metric_type]["variables"]

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
    preprocessing: Optional[Dict[str, str]] = None,
    target_types: Optional[Dict[str, str]] = None,
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict]]:
    """Perform cross-validation with dual metric support.

    Returns:
        Dictionary containing separate results for quantile_loss and log_loss:
        {
            "quantile_loss": {
                "results": pd.DataFrame,  # rows: ["train", "test"], cols: quantiles (mean across folds)
                "results_std": pd.DataFrame,  # rows: ["train", "test"], cols: quantiles (std across folds)
                "mean_train": float,
                "mean_test": float,
                "std_train": float,  # std of mean loss across folds
                "std_test": float,  # std of mean loss across folds
                "variables": List[str]
            },
            "log_loss": {
                "results": pd.DataFrame,  # rows: ["train", "test"], cols: quantiles (constant values)
                "results_std": pd.DataFrame,  # rows: ["train", "test"], cols: quantiles (std across folds)
                "mean_train": float,
                "mean_test": float,
                "std_train": float,
                "std_test": float,
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
    quantiles = QUANTILES if quantiles is None else quantiles
    validate_quantiles(quantiles)

    data = declare_target_types(data, imputed_variables, target_types)

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
                "quantile_loss" if metric_type == "quantile_loss" else "log_loss"
            )
            log.info(f"Variable '{var}' will use metric: {variable_metrics[var]}")

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
                    preprocessing,
                    target_types,
                    random_state,
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
                            for q, values in metric_results[metric_type]["test"].items()
                        },
                    ],
                    index=["train", "test"],
                )

                # Create std DataFrame for error bars
                std_df = pd.DataFrame(
                    [
                        {
                            q: np.std(values) if len(values) > 1 else 0.0
                            for q, values in metric_results[metric_type][
                                "train"
                            ].items()
                        },
                        {
                            q: np.std(values) if len(values) > 1 else 0.0
                            for q, values in metric_results[metric_type]["test"].items()
                        },
                    ],
                    index=["train", "test"],
                )

                # Calculate means and stds across all quantiles
                mean_test = combined_df.loc["test"].mean()
                mean_train = combined_df.loc["train"].mean()
                std_test = float(
                    np.std(
                        np.mean(
                            [metric_results[metric_type]["test"][q] for q in quantiles],
                            axis=0,
                        )
                    )
                )
                std_train = float(
                    np.std(
                        np.mean(
                            [
                                metric_results[metric_type]["train"][q]
                                for q in quantiles
                            ],
                            axis=0,
                        )
                    )
                )

                final_results[metric_type] = {
                    "results": combined_df,  # Single DataFrame with train/test rows
                    "results_std": std_df,  # Std across folds for each quantile
                    "mean_train": mean_train,
                    "mean_test": mean_test,
                    "std_train": std_train,
                    "std_test": std_test,
                    "variables": metric_results[metric_type]["variables"],
                }

                log.info(
                    f"{metric_type} - Mean Train: {mean_train:.6f} (±{std_train:.6f}), "
                    f"Mean Test: {mean_test:.6f} (±{std_test:.6f})"
                )
            else:
                # No variables use this metric
                final_results[metric_type] = {
                    "results": pd.DataFrame(),  # Empty DataFrame
                    "results_std": pd.DataFrame(),  # Empty DataFrame
                    "mean_train": np.nan,
                    "mean_test": np.nan,
                    "std_train": np.nan,
                    "std_test": np.nan,
                    "variables": [],
                }

        # Return results with optional hyperparameters
        if tune_hyperparameters and tuned_hyperparameters:
            # Outer test folds estimate performance only. Select final parameters
            # with a fresh internal tuning run on all available training rows.
            tuning_data = data
            if preprocessing:
                tuning_data, _ = preprocess_data(
                    data, full_data=True, **_preprocessing_kwargs(preprocessing)
                )
            _, best_hyperparams = _fit_model_for_fold(
                create_distributional_model(model_class, seed=random_state),
                model_class,
                tuning_data,
                predictors,
                imputed_variables,
                weight_col,
                quantiles,
                model_hyperparams,
                True,
                target_types,
            )
            return final_results, best_hyperparams
        else:
            return final_results

    except ValueError as e:
        raise e
    except (KeyError, TypeError, AttributeError, ImportError) as e:
        log.error(f"Error during cross-validation: {str(e)}")
        raise RuntimeError(f"Cross-validation failed: {str(e)}") from e
