"""Cross-validation utilities for imputation model evaluation.

This module provides functions for evaluating imputation models using k-fold
cross-validation. It calculates train and test quantile loss metrics for
each fold to provide robust performance estimates.
"""

import logging
from typing import Dict, List, Optional, Tuple, Type

import joblib
import numpy as np
import pandas as pd
from pydantic import validate_call
from sklearn.model_selection import KFold

from microimpute.comparisons.quantile_loss import quantile_loss
from microimpute.config import QUANTILES, RANDOM_STATE, VALIDATE_CONFIG

try:
    from microimpute.models.matching import Matching
except ImportError:  # optional dependency
    Matching = None
from microimpute.models.qrf import QRF
from microimpute.models.quantreg import QuantReg

log = logging.getLogger(__name__)


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
        model_class: Model class to evaluate (e.g., QRF, OLS, QuantReg,
                   Matching).
        data: Full dataset to split into training and testing folds.
        predictors: Names of columns to use as predictors.
        imputed_variables: Names of columns to impute.
        weight_col: Optional column name for sample weights.
        quantiles: List of quantiles to evaluate. Defaults to standard
            set if None.
        n_splits: Number of cross-validation folds.
        random_state: Random seed for reproducibility.
        model_hyperparams: Hyperparameters for the model class.
            Defaults to None and uses default model hyperparameters then.
        tune_hyperparameters: Whether to tune hyperparameters for QRF
            model. Defaults to False.

    Returns:
        DataFrame with train and test rows, quantiles as columns, and average
        loss values

    Raises:
        ValueError: If input data is invalid or missing required columns.
        RuntimeError: If cross-validation fails.
    """
    # Set up parallel processing
    # Disable parallel processing for Matching (R/rpy2 doesn't work well with multiprocessing)
    if Matching is not None and model_class == Matching:
        n_jobs: Optional[int] = 1  # Sequential processing for R-based models
    else:
        n_jobs: Optional[int] = (
            -1
        )  # Parallel processing for Python-only models

    try:
        # Validate predictor and imputed variable columns exist
        missing_predictors = [
            col for col in predictors if col not in data.columns
        ]
        if missing_predictors:
            error_msg = f"Missing predictor columns: {missing_predictors}"
            log.error(error_msg)
            raise ValueError(error_msg)

        missing_imputed = [
            col for col in imputed_variables if col not in data.columns
        ]
        if missing_imputed:
            error_msg = f"Missing imputed variable columns: {missing_imputed}"
            log.error(error_msg)
            raise ValueError(error_msg)

        if quantiles:
            invalid_quantiles = [q for q in quantiles if not 0 <= q <= 1]
            if invalid_quantiles:
                error_msg = f"Invalid quantiles (must be between 0 and 1): {invalid_quantiles}"
                log.error(error_msg)
                raise ValueError(error_msg)

        # Set up results containers
        test_results = {q: [] for q in quantiles}
        train_results = {q: [] for q in quantiles}
        train_y_values = []
        test_y_values = []

        log.info(
            f"Starting {n_splits}-fold cross-validation for {model_class.__name__}"
        )
        log.info(f"Evaluating at {len(quantiles)} quantiles: {quantiles}")

        # Set up k-fold cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        # Create parallel-ready fold indices
        fold_indices = list(kf.split(data))

        # Initialize tuned_hyperparameters
        tuned_hyperparameters = {} if tune_hyperparameters else None
        best_tuned_hyperparams = None

        # Define the function to process a single fold
        def process_single_fold(
            fold_idx_pair,
        ) -> Tuple[Dict, Dict, np.ndarray, np.ndarray]:
            """Process a single CV fold and return results.

            Args:
                fold_idx_pair: Tuple of (train_indices, test_indices)

            Returns:
                Tuple containing:
                - Dictionary of test predictions for each quantile
                - Dictionary of train predictions for each quantile
                - Array of actual train values
                - Array of actual test values
            """
            fold_idx, (train_idx, test_idx) = fold_idx_pair
            log.info(f"Processing fold {fold_idx+1}/{n_splits}")

            # Split data for this fold
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]

            # Store actual values for this fold
            train_y = train_data[imputed_variables].values
            test_y = test_data[imputed_variables].values

            # Instantiate the model
            log.info(f"Initializing {model_class.__name__} model")
            model = model_class()

            fold_tuned_params = None

            # Handle different model fitting requirements
            if (
                model_hyperparams
                and model_class.__name__ == "QRF"
                and ("QRF" in model_hyperparams)
            ):
                try:
                    log.info(
                        f"Fitting {model_class.__name__} model with hyperparameters: {model_hyperparams}"
                    )
                    fitted_model = model.fit(
                        X_train=train_data,
                        predictors=predictors,
                        imputed_variables=imputed_variables,
                        weight_col=weight_col,
                        **model_hyperparams["QRF"],
                    )
                except TypeError as e:
                    log.warning(
                        f"Invalid hyperparameters, using defaults: {str(e)}"
                    )
                    fitted_model = model.fit(
                        X_train=train_data,
                        predictors=predictors,
                        imputed_variables=imputed_variables,
                        weight_col=weight_col,
                    )
                    raise ValueError(
                        f"Invalid hyperparameters for model initialization. Current model hyperparameters: {fitted_model.models[imputed_variables[0]].qrf.get_params()}"
                    ) from e
            elif (
                model_hyperparams
                and Matching is not None
                and model_class.__name__ == "Matching"
                and ("Matching" in model_hyperparams)
            ):
                try:
                    log.info(
                        f"Fitting {model_class.__name__} model with hyperparameters: {model_hyperparams}"
                    )
                    fitted_model = model.fit(
                        X_train=train_data,
                        predictors=predictors,
                        imputed_variables=imputed_variables,
                        weight_col=weight_col,
                        **model_hyperparams["Matching"],
                    )
                except TypeError as e:
                    log.warning(
                        f"Invalid hyperparameters, using defaults: {str(e)}"
                    )
                    fitted_model = model.fit(
                        X_train=train_data,
                        predictors=predictors,
                        imputed_variables=imputed_variables,
                        weight_col=weight_col,
                    )
                    raise ValueError(
                        f"Invalid hyperparameters for model initialization. Current model hyperparameters: dist_fun=Manhattan, constrained=False"
                    ) from e
            else:
                if model_class == QuantReg:
                    log.info(f"Fitting QuantReg model with explicit quantiles")
                    fitted_model = model.fit(
                        train_data,
                        predictors,
                        imputed_variables,
                        weight_col=weight_col,
                        quantiles=quantiles,
                    )
                elif (
                    model_class.__name__ == "QRF"
                    or (
                        Matching is not None
                        and model_class.__name__ == "Matching"
                    )
                ) and tune_hyperparameters == True:
                    log.info(
                        f"Tuning {model_class.__name__} hyperparameters when fitting"
                    )
                    fitted_model, best_tuned_params = model.fit(
                        train_data,
                        predictors,
                        imputed_variables,
                        weight_col=weight_col,
                        tune_hyperparameters=True,
                    )
                    fold_tuned_params = best_tuned_params

                else:
                    log.info(f"Fitting {model_class.__name__} model")
                    fitted_model = model.fit(
                        train_data,
                        predictors,
                        imputed_variables,
                        weight_col=weight_col,
                    )

            # Get predictions for this fold
            log.info(f"Generating predictions for train and test data")
            fold_test_imputations = fitted_model.predict(test_data, quantiles)
            fold_train_imputations = fitted_model.predict(
                train_data, quantiles
            )

            # Return results for this fold
            return (
                fold_idx,
                fold_test_imputations,
                fold_train_imputations,
                test_y,
                train_y,
                fold_tuned_params,
            )

        # Execute folds in parallel
        fold_results = []
        with joblib.Parallel(n_jobs=n_jobs, verbose=10) as parallel:
            fold_results = parallel(
                joblib.delayed(process_single_fold)((i, fold_pair))
                for i, fold_pair in enumerate(fold_indices)
            )

        # Organize results
        test_results = {q: [] for q in quantiles}
        train_results = {q: [] for q in quantiles}
        test_y_values = []
        train_y_values = []

        # Sort results by fold index to maintain order
        fold_results.sort(key=lambda x: x[0])

        # Extract results
        for (
            fold_idx,
            fold_test_imputations,
            fold_train_imputations,
            test_y,
            train_y,
            fold_tuned_params,
        ) in fold_results:
            test_y_values.append(test_y)
            train_y_values.append(train_y)
            if tune_hyperparameters:
                tuned_hyperparameters[fold_idx] = fold_tuned_params

            # Store results for each quantile
            for q in quantiles:
                test_results[q].append(fold_test_imputations[q])
                train_results[q].append(fold_train_imputations[q])

        # Calculate loss metrics (this can also be parallelized for large datasets)
        log.info("Computing loss metrics across all folds")

        # Define a function to compute loss for a specific fold and quantile
        def compute_fold_loss(fold_idx, quantile):
            # Flatten arrays for easier calculation
            test_y_flat = test_y_values[fold_idx].flatten()
            train_y_flat = train_y_values[fold_idx].flatten()
            test_pred_flat = test_results[quantile][fold_idx].values.flatten()
            train_pred_flat = train_results[quantile][
                fold_idx
            ].values.flatten()

            # Calculate the loss for this fold and quantile
            test_loss = quantile_loss(quantile, test_y_flat, test_pred_flat)
            train_loss = quantile_loss(quantile, train_y_flat, train_pred_flat)

            return {
                "fold": fold_idx,
                "quantile": quantile,
                "test_loss": test_loss.mean(),
                "train_loss": train_loss.mean(),
            }

        # Create tasks for parallel loss computation
        loss_tasks = [
            (k, q) for k in range(len(test_y_values)) for q in quantiles
        ]

        # Compute losses in parallel if there are many folds/quantiles
        if (
            len(loss_tasks) > 10 and n_jobs != 1
        ):  # Only parallelize if it's worth it
            with joblib.Parallel(n_jobs=n_jobs) as parallel:
                loss_results = parallel(
                    joblib.delayed(compute_fold_loss)(fold_idx, q)
                    for fold_idx, q in loss_tasks
                )

            # Organize loss results
            avg_test_losses = {q: [] for q in quantiles}
            avg_train_losses = {q: [] for q in quantiles}

            if (
                model_class == QRF
                or (Matching is not None and model_class == Matching)
            ) and tune_hyperparameters == True:
                best_fold = fold_indices[0][0]
                best_loss = float("inf")
                for result in loss_results:
                    q = result["quantile"]
                    fold_idx = result["fold"]
                    avg_test_losses[q].append(result["test_loss"])
                    avg_train_losses[q].append(result["train_loss"])

                    log.debug(
                        f"Fold {fold_idx+1}, q={q}: Train loss = {result['train_loss']:.6f}, Test loss = {result['test_loss']:.6f}"
                    )
                    if q == 0.5:
                        if result["test_loss"] < best_loss:
                            best_loss = result["test_loss"]
                            best_fold = fold_idx
                best_tuned_hyperparams = tuned_hyperparameters[best_fold]
            else:
                for result in loss_results:
                    q = result["quantile"]
                    fold_idx = result["fold"]
                    avg_test_losses[q].append(result["test_loss"])
                    avg_train_losses[q].append(result["train_loss"])

                    log.debug(
                        f"Fold {fold_idx+1}, q={q}: Train loss = {result['train_loss']:.6f}, Test loss = {result['test_loss']:.6f}"
                    )
        else:
            # Calculate losses sequentially for simpler cases
            avg_test_losses = {q: [] for q in quantiles}
            avg_train_losses = {q: [] for q in quantiles}

            if (
                model_class == QRF
                or (Matching is not None and model_class == Matching)
            ) and tune_hyperparameters == True:
                best_fold = fold_indices[0][0]
                best_loss = float("inf")
                for k in range(len(test_y_values)):
                    for q in quantiles:
                        result = compute_fold_loss(k, q)
                        avg_test_losses[q].append(result["test_loss"])
                        avg_train_losses[q].append(result["train_loss"])

                        log.debug(
                            f"Fold {k+1}, q={q}: Train loss = {result['train_loss']:.6f}, Test loss = {result['test_loss']:.6f}"
                        )
                    if result["test_loss"] < best_loss:
                        best_fold = k
                        best_loss = result["test_loss"]
                best_tuned_hyperparams = tuned_hyperparameters[best_fold]
            else:
                for k in range(len(test_y_values)):
                    for q in quantiles:
                        result = compute_fold_loss(k, q)
                        avg_test_losses[q].append(result["test_loss"])
                        avg_train_losses[q].append(result["train_loss"])

                        log.debug(
                            f"Fold {k+1}, q={q}: Train loss = {result['train_loss']:.6f}, Test loss = {result['test_loss']:.6f}"
                        )

        # Calculate the average loss across all folds for each quantile
        log.info("Calculating final average metrics")
        final_test_losses = {
            q: np.mean(losses) for q, losses in avg_test_losses.items()
        }
        final_train_losses = {
            q: np.mean(losses) for q, losses in avg_train_losses.items()
        }

        # Create DataFrame with quantiles as columns
        final_results = pd.DataFrame(
            [final_train_losses, final_test_losses], index=["train", "test"]
        )

        # Generate summary statistics
        train_mean = final_results.loc["train"].mean()
        test_mean = final_results.loc["test"].mean()
        train_test_ratio = train_mean / test_mean

        log.info(f"Cross-validation completed for {model_class.__name__}")
        log.info(f"Average Train Loss: {train_mean:.6f}")
        log.info(f"Average Test Loss: {test_mean:.6f}")
        log.info(f"Train/Test Ratio: {train_test_ratio:.6f}")

        if tune_hyperparameters:
            return final_results, best_tuned_hyperparams
        else:
            return final_results

    except ValueError as e:
        # Re-raise validation errors directly
        raise e
    except Exception as e:
        log.error(f"Error during cross-validation: {str(e)}")
        raise RuntimeError(f"Cross-validation failed: {str(e)}") from e
