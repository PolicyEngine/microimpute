"""Cross-validation utilities for imputation model evaluation.

This module provides functions for evaluating imputation models using k-fold
cross-validation. It calculates train and test quantile loss metrics for
each fold to provide robust performance estimates.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Type

import joblib
import numpy as np
import pandas as pd
from pydantic import validate_call
from sklearn.model_selection import KFold

from microimpute.comparisons.quantile_loss import quantile_loss
from microimpute.comparisons.validation import (
    validate_columns_exist,
    validate_quantiles,
)
from microimpute.config import QUANTILES, RANDOM_STATE, VALIDATE_CONFIG

try:
    from microimpute.models.matching import Matching
except ImportError:  # optional dependency
    Matching = None
from microimpute.models.qrf import QRF
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
) -> Tuple[int, Dict, Dict, np.ndarray, np.ndarray, Optional[dict]]:
    """Process a single CV fold and return results.

    Args:
        fold_idx_pair: Tuple of (fold_index, (train_indices, test_indices))
        data: Full dataset
        model_class: Model class to evaluate
        predictors: Predictor column names
        imputed_variables: Variables to impute
        weight_col: Optional weight column
        quantiles: List of quantiles to evaluate
        model_hyperparams: Optional model hyperparameters
        tune_hyperparameters: Whether to tune hyperparameters

    Returns:
        Tuple containing fold results
    """
    fold_idx, (train_idx, test_idx) = fold_idx_pair
    log.info(f"Processing fold {fold_idx+1}")

    # Split data for this fold
    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]

    # Store actual values for this fold
    train_y = train_data[imputed_variables].values
    test_y = test_data[imputed_variables].values

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

    # Get predictions for this fold
    log.info(f"Generating predictions for train and test data")
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

    Returns:
        Tuple of (fitted_model, tuned_hyperparameters or None)
    """
    model_name = model_class.__name__
    fold_tuned_params = None

    # Handle model-specific hyperparameters
    if model_hyperparams and model_name in model_hyperparams:
        try:
            log.info(
                f"Fitting {model_name} with hyperparameters: {model_hyperparams[model_name]}"
            )
            fitted_model = model.fit(
                X_train=train_data,
                predictors=predictors,
                imputed_variables=imputed_variables,
                weight_col=weight_col,
                **model_hyperparams[model_name],
            )
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
        log.info(f"Fitting QuantReg model with explicit quantiles")
        fitted_model = model.fit(
            train_data,
            predictors,
            imputed_variables,
            weight_col=weight_col,
            quantiles=quantiles,
        )

    # Handle hyperparameter tuning for QRF and Matching
    elif tune_hyperparameters and model_name in ["QRF", "Matching"]:
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
        log.info(f"Fitting {model_name} model with default parameters")
        fitted_model = model.fit(
            train_data, predictors, imputed_variables, weight_col=weight_col
        )

    return fitted_model, fold_tuned_params


def _compute_fold_loss(
    fold_idx: int,
    quantile: float,
    test_y_values: List[np.ndarray],
    train_y_values: List[np.ndarray],
    test_results: Dict[float, List],
    train_results: Dict[float, List],
) -> Dict[str, Any]:
    """Compute loss for a specific fold and quantile.

    Returns:
        Dictionary with fold, quantile, and loss metrics
    """
    # Flatten arrays for calculation
    test_y_flat = test_y_values[fold_idx].flatten()
    train_y_flat = train_y_values[fold_idx].flatten()
    test_pred_flat = test_results[quantile][fold_idx].values.flatten()
    train_pred_flat = train_results[quantile][fold_idx].values.flatten()

    # Calculate loss
    test_loss = quantile_loss(quantile, test_y_flat, test_pred_flat)
    train_loss = quantile_loss(quantile, train_y_flat, train_pred_flat)

    return {
        "fold": fold_idx,
        "quantile": quantile,
        "test_loss": test_loss.mean(),
        "train_loss": train_loss.mean(),
    }


def _compute_losses_parallel(
    test_y_values: List[np.ndarray],
    train_y_values: List[np.ndarray],
    test_results: Dict[float, List],
    train_results: Dict[float, List],
    quantiles: List[float],
    n_jobs: int,
) -> Tuple[Dict[float, List[float]], Dict[float, List[float]]]:
    """Compute losses in parallel for all folds and quantiles.

    Returns:
        Tuple of (test_losses_by_quantile, train_losses_by_quantile)
    """
    loss_tasks = [(k, q) for k in range(len(test_y_values)) for q in quantiles]

    # Only parallelize if worthwhile
    if len(loss_tasks) > 10 and n_jobs != 1:
        with joblib.Parallel(n_jobs=n_jobs) as parallel:
            loss_results = parallel(
                joblib.delayed(_compute_fold_loss)(
                    fold_idx,
                    q,
                    test_y_values,
                    train_y_values,
                    test_results,
                    train_results,
                )
                for fold_idx, q in loss_tasks
            )
    else:
        # Sequential computation for smaller tasks
        loss_results = [
            _compute_fold_loss(
                fold_idx,
                q,
                test_y_values,
                train_y_values,
                test_results,
                train_results,
            )
            for fold_idx, q in loss_tasks
        ]

    # Organize results
    avg_test_losses = {q: [] for q in quantiles}
    avg_train_losses = {q: [] for q in quantiles}

    for result in loss_results:
        q = result["quantile"]
        fold_idx = result["fold"]
        avg_test_losses[q].append(result["test_loss"])
        avg_train_losses[q].append(result["train_loss"])

        log.debug(
            f"Fold {fold_idx+1}, q={q}: Train loss = {result['train_loss']:.6f}, "
            f"Test loss = {result['test_loss']:.6f}"
        )

    return avg_test_losses, avg_train_losses


def _select_best_hyperparameters(
    loss_results: List[Dict], tuned_hyperparameters: Dict[int, Any]
) -> Any:
    """Select best hyperparameters based on median quantile test loss.

    Args:
        loss_results: List of loss result dictionaries
        tuned_hyperparameters: Dictionary mapping fold index to tuned parameters

    Returns:
        Best hyperparameters
    """
    best_fold = 0
    best_loss = float("inf")

    for result in loss_results:
        if result["quantile"] == 0.5 and result["test_loss"] < best_loss:
            best_loss = result["test_loss"]
            best_fold = result["fold"]

    return tuned_hyperparameters.get(best_fold)


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
) -> pd.DataFrame:
    """Perform cross-validation for an imputation model.

    Args:
        model_class: Model class to evaluate (e.g., QRF, OLS, QuantReg, Matching).
        data: Full dataset to split into training and testing folds.
        predictors: Names of columns to use as predictors.
        imputed_variables: Names of columns to impute.
        weight_col: Optional column name for sample weights.
        quantiles: List of quantiles to evaluate. Defaults to standard set if None.
        n_splits: Number of cross-validation folds.
        random_state: Random seed for reproducibility.
        model_hyperparams: Hyperparameters for the model class.
        tune_hyperparameters: Whether to tune hyperparameters for QRF/Matching models.

    Returns:
        DataFrame with train and test rows, quantiles as columns, and average loss values.
        If tune_hyperparameters is True, returns tuple of (DataFrame, best_hyperparameters).

    Raises:
        ValueError: If input data is invalid or missing required columns.
        RuntimeError: If cross-validation fails.
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
                )
                for i, fold_pair in enumerate(fold_indices)
            )

        # Sort results by fold index
        fold_results.sort(key=lambda x: x[0])

        # Extract and organize results
        test_results = {q: [] for q in quantiles}
        train_results = {q: [] for q in quantiles}
        test_y_values = []
        train_y_values = []
        tuned_hyperparameters = {}

        for (
            fold_idx,
            fold_test_imp,
            fold_train_imp,
            test_y,
            train_y,
            fold_tuned_params,
        ) in fold_results:
            test_y_values.append(test_y)
            train_y_values.append(train_y)

            if tune_hyperparameters and fold_tuned_params:
                tuned_hyperparameters[fold_idx] = fold_tuned_params

            for q in quantiles:
                test_results[q].append(fold_test_imp[q])
                train_results[q].append(fold_train_imp[q])

        # Compute losses
        log.info("Computing loss metrics across all folds")
        avg_test_losses, avg_train_losses = _compute_losses_parallel(
            test_y_values,
            train_y_values,
            test_results,
            train_results,
            quantiles,
            n_jobs,
        )

        # Calculate final average metrics
        log.info("Calculating final average metrics")
        final_test_losses = {
            q: np.mean(losses) for q, losses in avg_test_losses.items()
        }
        final_train_losses = {
            q: np.mean(losses) for q, losses in avg_train_losses.items()
        }

        # Create results DataFrame
        final_results = pd.DataFrame(
            [final_train_losses, final_test_losses], index=["train", "test"]
        )

        # Log summary statistics
        train_mean = final_results.loc["train"].mean()
        test_mean = final_results.loc["test"].mean()
        log.info(f"Cross-validation completed for {model_class.__name__}")
        log.info(f"Average Train Loss: {train_mean:.6f}")
        log.info(f"Average Test Loss: {test_mean:.6f}")
        log.info(f"Train/Test Ratio: {train_mean / test_mean:.6f}")

        # Return results with optional hyperparameters
        if tune_hyperparameters and tuned_hyperparameters:
            # Create simplified loss results for hyperparameter selection
            loss_results = []
            for fold_idx in range(len(test_y_values)):
                for q in quantiles:
                    loss_results.append(
                        {
                            "fold": fold_idx,
                            "quantile": q,
                            "test_loss": avg_test_losses[q][fold_idx],
                        }
                    )
            best_hyperparams = _select_best_hyperparameters(
                loss_results, tuned_hyperparameters
            )
            return final_results, best_hyperparams
        else:
            return final_results

    except ValueError as e:
        raise e
    except (KeyError, TypeError, AttributeError, ImportError) as e:
        log.error(f"Error during cross-validation: {str(e)}")
        raise RuntimeError(f"Cross-validation failed: {str(e)}") from e
