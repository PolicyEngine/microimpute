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

import pandas as pd

from microimpute.comparisons.validation import (
    validate_imputation_inputs,
    validate_quantiles,
)
from microimpute.evaluations import cross_validate_model
from microimpute.models import Imputer
from microimpute.models.quantreg import QuantReg
from microimpute.utils.data import preprocess_data, unnormalize_predictions

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
    normalize_data: bool,
    train_size: float,
    test_size: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[dict]]:
    """Prepare training and imputing data, optionally with normalization.

    Args:
        donor_data: Original donor data.
        receiver_data: Original receiver data.
        predictors: Predictor columns.
        imputed_variables: Variables to impute.
        weight_col: Optional weight column.
        normalize_data: Whether to normalize.
        train_size: Training data proportion.
        test_size: Test data proportion.

    Returns:
        Tuple of (training_data, imputing_data, normalization_params or None)
    """
    # Remove imputed variables from receiver if present
    receiver_data = receiver_data.drop(
        columns=imputed_variables, errors="ignore"
    )

    training_data = donor_data.copy()
    imputing_data = receiver_data.copy()

    if normalize_data:
        # Normalize predictors and imputed variables together for consistency
        all_training_cols = predictors + imputed_variables
        normalized_training, norm_params = preprocess_data(
            training_data[all_training_cols],
            full_data=True,
            train_size=train_size,
            test_size=test_size,
            normalize=True,
        )

        # Normalize imputing data predictors using same parameters
        imputing_predictors, _ = preprocess_data(
            imputing_data[predictors],
            full_data=True,
            train_size=train_size,
            test_size=test_size,
            normalize=True,
        )

        # Reconstruct training data with normalized values
        training_data = normalized_training
        if weight_col:
            training_data[weight_col] = donor_data[weight_col]

        imputing_data = imputing_predictors

        # Extract normalization params only for imputed variables
        imputed_norm_params = {
            col: norm_params[col]
            for col in imputed_variables
            if col in norm_params
        }

        return training_data, imputing_data, imputed_norm_params
    else:
        # No normalization needed
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
    elif hyperparams and model_name in ["Matching", "QRF"]:
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

    # Generate predictions
    imputations = fitted_model.predict(imputing_data, quantiles=[quantile])

    # Handle case where predict returns a DataFrame directly
    if isinstance(imputations, pd.DataFrame):
        imputations = {quantile: imputations}

    return fitted_model, imputations


def select_best_model(
    method_results_df: pd.DataFrame,
) -> Tuple[str, pd.Series]:
    """Select the best model based on cross-validation results.

    Args:
        method_results_df: DataFrame with model performance metrics.

    Returns:
        Tuple of (best_method_name, best_method_row)
    """
    # Add mean_loss column if not present
    if "mean_loss" not in method_results_df.columns:
        method_results_df["mean_loss"] = method_results_df.mean(axis=1)

    best_method = method_results_df["mean_loss"].idxmin()
    best_row = method_results_df.loc[best_method]

    log.info(
        f"The method with the lowest average loss is {best_method}, "
        f"with an average loss across variables and quantiles of {best_row['mean_loss']}"
    )

    return best_method, best_row
