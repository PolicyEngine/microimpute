"""Helper functions for automated imputation

This module provides utility functions that support the autoimpute workflow,
including input validation, data preparation, model evaluation, and result selection.
These functions are extracted from the main autoimpute module to improve code
organization and maintainability.

Key functions:
    - validate_autoimpute_inputs: comprehensive input validation
    - prepare_data_for_imputation: data preprocessing and normalization
    - evaluate_model: cross-validation evaluation for a single model
    - fit_and_predict_model: model fitting and prediction generation
    - select_best_model: selection of best performing model
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from microimpute.comparisons.validation import (
    validate_imputation_inputs,
    validate_quantiles,
)
from microimpute.evaluations import cross_validate_model
from microimpute.models import Imputer
from microimpute.utils.data import preprocess_data

log = logging.getLogger(__name__)


def validate_autoimpute_inputs(
    donor_data: pd.DataFrame,
    receiver_data: pd.DataFrame,
    predictors: List[str],
    imputed_variables: List[str],
    weight_col: Optional[str],
    quantiles: Optional[List[float]],
    hyperparameters: Optional[Dict[str, Dict[str, Any]]],
    tune_hyperparameters: bool,
    log_level: str,
) -> None:
    """Validate all inputs for the autoimpute function.

    Args:
        donor_data: Training data.
        receiver_data: Data to impute.
        predictors: Predictor column names.
        imputed_variables: Variables to impute.
        weight_col: Optional weight column.
        quantiles: Optional quantiles list.
        hyperparameters: Optional model hyperparameters.
        tune_hyperparameters: Whether to tune hyperparameters.
        log_level: Logging level string.

    Raises:
        ValueError: If validation fails.
    """
    # Validate log level
    if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        error_msg = f"Invalid log_level: {log_level}. Must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL."
        log.error(error_msg)
        raise ValueError(error_msg)

    # Validate quantiles if provided
    if quantiles:
        validate_quantiles(quantiles)

    # Validate data and columns
    validate_imputation_inputs(
        donor_data, receiver_data, predictors, imputed_variables, weight_col
    )

    # Validate hyperparameter settings
    if hyperparameters is not None and tune_hyperparameters:
        error_msg = "Cannot specify both model_hyperparams and request to automatically tune hyperparameters, please select one or the other."
        log.error(error_msg)
        raise ValueError(error_msg)


def prepare_data_for_imputation(
    donor_data: pd.DataFrame,
    receiver_data: pd.DataFrame,
    predictors: List[str],
    imputed_variables: List[str],
    weight_col: Optional[str],
    train_size: float,
    test_size: float,
    preprocessing: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[dict]]:
    """Prepare training and imputing data, optionally with transformations.

    Args:
        donor_data: Original donor data.
        receiver_data: Original receiver data.
        predictors: Predictor columns.
        imputed_variables: Variables to impute.
        weight_col: Optional weight column.
        train_size: Training data proportion.
        test_size: Test data proportion.
        preprocessing: Dict mapping variable names to transformation type.
            Supported: "normalize", "log", "asinh". If None, no transformation.

    Returns:
        Tuple of (training_data, imputing_data, transform_params or None)
        transform_params contains info needed to reverse transformations on
        imputed variables.
    """
    # Remove imputed variables from receiver if present
    receiver_data = receiver_data.drop(
        columns=imputed_variables, errors="ignore"
    )

    training_data = donor_data.copy()
    imputing_data = receiver_data.copy()

    if preprocessing:
        all_training_cols = predictors + imputed_variables
        all_cols = set(all_training_cols)

        # Validate preprocessing keys
        invalid_cols = set(preprocessing.keys()) - all_cols
        if invalid_cols:
            error_msg = (
                f"Preprocessing specified for unknown columns: {invalid_cols}. "
                f"Valid columns are: {all_cols}"
            )
            log.error(error_msg)
            raise ValueError(error_msg)

        # Validate transformation types
        valid_transforms = {"normalize", "log", "asinh"}
        for col, transform in preprocessing.items():
            if transform not in valid_transforms:
                error_msg = (
                    f"Invalid transformation '{transform}' for column '{col}'. "
                    f"Valid transformations are: {valid_transforms}"
                )
                log.error(error_msg)
                raise ValueError(error_msg)

        # Group columns by transformation type
        normalize_cols = [
            col for col, t in preprocessing.items() if t == "normalize"
        ]
        log_cols = [col for col, t in preprocessing.items() if t == "log"]
        asinh_cols = [col for col, t in preprocessing.items() if t == "asinh"]

        # Apply transformations to training data
        transformed_training, transform_result = preprocess_data(
            training_data[all_training_cols],
            full_data=True,
            train_size=train_size,
            test_size=test_size,
            normalize=normalize_cols if normalize_cols else False,
            log_transform=log_cols if log_cols else False,
            asinh_transform=asinh_cols if asinh_cols else False,
        )

        # Apply same transformations to predictors in imputing data
        predictor_normalize = [c for c in normalize_cols if c in predictors]
        predictor_log = [c for c in log_cols if c in predictors]
        predictor_asinh = [c for c in asinh_cols if c in predictors]

        if predictor_normalize or predictor_log or predictor_asinh:
            transformed_imputing, _ = preprocess_data(
                imputing_data[predictors],
                full_data=True,
                train_size=train_size,
                test_size=test_size,
                normalize=(
                    predictor_normalize if predictor_normalize else False
                ),
                log_transform=predictor_log if predictor_log else False,
                asinh_transform=predictor_asinh if predictor_asinh else False,
            )
        else:
            transformed_imputing = imputing_data[predictors].copy()

        training_data = transformed_training
        if weight_col:
            training_data[weight_col] = donor_data[weight_col]

        imputing_data = transformed_imputing

        # Extract transform params only for imputed variables
        imputed_transform_params = {
            "normalization": {
                col: transform_result["normalization"].get(col, {})
                for col in imputed_variables
                if col in transform_result.get("normalization", {})
            },
            "log_transform": {
                col: transform_result["log_transform"].get(col, {})
                for col in imputed_variables
                if col in transform_result.get("log_transform", {})
            },
            "asinh_transform": {
                col: transform_result["asinh_transform"].get(col, {})
                for col in imputed_variables
                if col in transform_result.get("asinh_transform", {})
            },
        }

        # Only return params if there are transformations to reverse
        has_transforms = any(
            imputed_transform_params[key]
            for key in ["normalization", "log_transform", "asinh_transform"]
        )

        if has_transforms:
            transform_params = {
                "type": "preprocessing",
                "params": imputed_transform_params,
            }
            return training_data, imputing_data, transform_params
        else:
            return training_data, imputing_data, None

    else:
        # No transformation needed
        training_data = preprocess_data(
            training_data[predictors + imputed_variables],
            full_data=True,
            train_size=train_size,
            test_size=test_size,
            normalize=False,
        )

        imputing_data = preprocess_data(
            imputing_data[predictors],
            full_data=True,
            train_size=train_size,
            test_size=test_size,
            normalize=False,
        )

        if weight_col:
            training_data[weight_col] = donor_data[weight_col]

        return training_data, imputing_data, None


def evaluate_model(
    model: Type[Imputer],
    data: pd.DataFrame,
    predictors: List[str],
    imputed_variables: List[str],
    weight_col: Optional[str],
    quantiles: List[float],
    k_folds: int,
    random_state: int,
    tune_hyperparams: bool,
    hyperparameters: Optional[Dict[str, Any]],
) -> tuple:
    """Evaluate a single imputation model with cross-validation.

    Args:
        model: The imputation model class to evaluate.
        data: The dataset to use for evaluation.
        predictors: List of predictor column names.
        imputed_variables: List of columns to impute.
        weight_col: Optional weight column.
        quantiles: List of quantiles to evaluate.
        k_folds: Number of cross-validation folds.
        random_state: Random seed for reproducibility.
        tune_hyperparams: Whether to tune hyperparameters.
        hyperparameters: Optional model-specific hyperparameters.

    Returns:
        Tuple containing model name and cross-validation results.
        Results are now a dict with 'quantile_loss' and 'log_loss' keys.
    """
    model_name = model.__name__
    log.info(f"Evaluating {model_name}...")

    cv_result = cross_validate_model(
        model_class=model,
        data=data,
        predictors=predictors,
        imputed_variables=imputed_variables,
        weight_col=weight_col,
        quantiles=quantiles,
        n_splits=k_folds,
        random_state=random_state,
        tune_hyperparameters=tune_hyperparams,
        model_hyperparams=hyperparameters,
    )

    if (
        tune_hyperparams
        and isinstance(cv_result, tuple)
        and len(cv_result) == 2
    ):
        final_results, best_params = cv_result
        return model_name, final_results, best_params
    else:
        return model_name, cv_result


def fit_and_predict_model(
    model_class: Type[Imputer],
    training_data: pd.DataFrame,
    imputing_data: pd.DataFrame,
    predictors: List[str],
    imputed_variables: List[str],
    weight_col: Optional[str],
    quantile: float,
    hyperparams: Optional[Dict[str, Any]] = None,
    log_level: str = "WARNING",
) -> Tuple[Any, Dict[float, pd.DataFrame]]:
    """Fit a model and generate predictions.

    Args:
        model_class: The model class to use.
        training_data: Training data.
        imputing_data: Data to make predictions on.
        predictors: Predictor columns.
        imputed_variables: Variables to impute.
        weight_col: Optional weight column.
        quantile: Quantile to predict.
        hyperparams: Optional model hyperparameters.
        log_level: Logging level.

    Returns:
        Tuple of (fitted_model, predictions_dict)
    """
    model_name = model_class.__name__
    model = model_class(log_level=log_level)

    # Check for categorical variables
    from microimpute.comparisons.metrics import get_metric_for_variable_type

    has_categorical = any(
        get_metric_for_variable_type(training_data[var], var) == "log_loss"
        for var in imputed_variables
    )

    # Special check for QuantReg with categorical variables
    if model_name == "QuantReg" and has_categorical:
        categorical_vars = [
            var
            for var in imputed_variables
            if get_metric_for_variable_type(training_data[var], var)
            == "log_loss"
        ]
        error_msg = (
            f"QuantReg does not support categorical variables: {categorical_vars}. "
            f"Please use QRF, OLS, or Matching models instead."
        )
        log.error(error_msg)
        raise ValueError(error_msg)

    # Fit the model
    if model_name == "QuantReg":
        # QuantReg needs explicit quantiles during fitting
        fitted_model = model.fit(
            training_data,
            predictors,
            imputed_variables,
            weight_col=weight_col,
            quantiles=[quantile],
        )
    elif hyperparams and model_name in ["Matching", "QRF", "MDN"]:
        # Apply hyperparameters for specific models
        fitted_model = model.fit(
            training_data,
            predictors,
            imputed_variables,
            weight_col=weight_col,
            **hyperparams,
        )
    else:
        fitted_model = model.fit(
            training_data,
            predictors,
            imputed_variables,
            weight_col=weight_col,
        )

    # Generate predictions with return_probs for categorical variables
    if has_categorical:
        imputations = fitted_model.predict(
            imputing_data, quantiles=[quantile], return_probs=True
        )
    else:
        imputations = fitted_model.predict(imputing_data, quantiles=[quantile])

    # Handle case where predict returns a DataFrame directly
    if isinstance(imputations, pd.DataFrame):
        imputations = {quantile: imputations}

    return fitted_model, imputations


def select_best_model_dual_metrics(
    method_results: Dict[str, Dict[str, Any]],
    metric_priority: str = "auto",
) -> Tuple[str, Dict[str, float]]:
    """Select the best model based on dual metric cross-validation results.

    Args:
        method_results: Dictionary with model names as keys and CV results as values.
                       Each result contains 'quantile_loss' and 'log_loss' subdicts.
        metric_priority: 'auto' (rank-based), 'numerical', 'categorical', or 'combined'.

    Returns:
        Tuple of (best_method_name, metrics_dict)
    """
    # Extract metrics for each model
    model_metrics = {}
    for model_name, results in method_results.items():
        model_metrics[model_name] = {
            "quantile_loss": results.get("quantile_loss", {}).get(
                "mean_test", np.inf
            ),
            "log_loss": results.get("log_loss", {}).get("mean_test", np.inf),
            "n_quantile_vars": len(
                results.get("quantile_loss", {}).get("variables", [])
            ),
            "n_log_vars": len(
                results.get("log_loss", {}).get("variables", [])
            ),
        }

    # Select based on priority
    if metric_priority == "numerical":
        # Check if any model has numerical variables
        has_numerical = any(
            model_metrics[m]["n_quantile_vars"] > 0 for m in model_metrics
        )
        if not has_numerical:
            error_msg = (
                "No numerical variables found for evaluation with 'numerical' metric priority. "
                "Please check your imputed_variables or use a different metric_priority."
            )
            log.error(error_msg)
            raise ValueError(error_msg)

        # Use only quantile loss
        best_method = min(
            model_metrics.keys(),
            key=lambda m: (
                model_metrics[m]["quantile_loss"]
                if not np.isnan(model_metrics[m]["quantile_loss"])
                else np.inf
            ),
        )
        log.info(
            f"Selected {best_method} based on quantile loss: {model_metrics[best_method]['quantile_loss']:.6f}"
        )

    elif metric_priority == "categorical":
        # Check if any model has categorical variables
        has_categorical = any(
            model_metrics[m]["n_log_vars"] > 0 for m in model_metrics
        )
        if not has_categorical:
            error_msg = (
                "No categorical variables found for evaluation with 'categorical' metric priority. "
                "Please check your imputed_variables or use a different metric_priority."
            )
            log.error(error_msg)
            raise ValueError(error_msg)

        # Use only log loss
        best_method = min(
            model_metrics.keys(),
            key=lambda m: (
                model_metrics[m]["log_loss"]
                if not np.isnan(model_metrics[m]["log_loss"])
                else np.inf
            ),
        )
        log.info(
            f"Selected {best_method} based on log loss: {model_metrics[best_method]['log_loss']:.6f}"
        )

    elif metric_priority == "auto":
        # Rank-based selection
        models = list(model_metrics.keys())

        # Check if there are any variables to evaluate
        total_vars_across_models = sum(
            model_metrics[m]["n_quantile_vars"]
            + model_metrics[m]["n_log_vars"]
            for m in models
        )
        if total_vars_across_models == 0:
            error_msg = (
                "No variables compatible with any model for evaluation. "
                "Please check that your imputed_variables are compatible with the selected models. "
                "For example, QuantReg only supports numerical variables."
            )
            log.error(error_msg)
            raise ValueError(error_msg)

        # Calculate ranks for each metric
        quantile_scores = [model_metrics[m]["quantile_loss"] for m in models]
        log_scores = [model_metrics[m]["log_loss"] for m in models]

        # Replace NaN/inf with worst rank
        quantile_ranks = pd.Series(quantile_scores).rank(na_option="bottom")
        log_ranks = pd.Series(log_scores).rank(na_option="bottom")

        # Weight ranks by number of variables
        avg_ranks = []
        for i, m in enumerate(models):
            n_q = model_metrics[m]["n_quantile_vars"]
            n_l = model_metrics[m]["n_log_vars"]
            if (n_q + n_l) > 0:
                weighted_rank = (
                    n_q * quantile_ranks.iloc[i] + n_l * log_ranks.iloc[i]
                ) / (n_q + n_l)
            else:
                weighted_rank = float("inf")
            avg_ranks.append(weighted_rank)

        best_idx = np.argmin(avg_ranks)
        best_method = models[best_idx]
        log.info(
            f"Selected {best_method} based on weighted rank (quantile rank: {quantile_ranks.iloc[best_idx]:.1f}, "
            f"log rank: {log_ranks.iloc[best_idx]:.1f})"
        )

    else:  # combined or other
        # Check if there are any variables to evaluate
        total_vars = sum(
            model_metrics[m]["n_quantile_vars"]
            + model_metrics[m]["n_log_vars"]
            for m in model_metrics
        )
        if total_vars == 0:
            error_msg = (
                "No variables available for evaluation with 'combined' metric priority. "
                "No models have compatible variables with the imputed_variables provided."
            )
            log.error(error_msg)
            raise ValueError(error_msg)

        # Simple average of normalized metrics
        best_score = float("inf")
        best_method = None

        for model_name, metrics in model_metrics.items():
            q_loss = (
                metrics["quantile_loss"]
                if not np.isnan(metrics["quantile_loss"])
                else 0
            )
            l_loss = (
                metrics["log_loss"] if not np.isnan(metrics["log_loss"]) else 0
            )
            n_q = metrics["n_quantile_vars"]
            n_l = metrics["n_log_vars"]

            if (n_q + n_l) > 0:
                combined = (n_q * q_loss + n_l * l_loss) / (n_q + n_l)
                if combined < best_score:
                    best_score = combined
                    best_method = model_name

        if best_method is None:
            error_msg = "Failed to select a model - all models have infinite combined scores."
            log.error(error_msg)
            raise RuntimeError(error_msg)

        log.info(
            f"Selected {best_method} based on combined metric: {best_score:.6f}"
        )

    return best_method, model_metrics[best_method]
