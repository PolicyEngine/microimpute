"""
Pipeline for autoimputation of missing values in a dataset.
This module integrates all steps necessary for method selection and imputation of missing values.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import joblib
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, validate_call
from tqdm.auto import tqdm

from microimpute.comparisons import *
from microimpute.comparisons.autoimpute_helpers import (
    evaluate_model,
    fit_and_predict_model,
    prepare_data_for_imputation,
    select_best_model,
    validate_autoimpute_inputs,
)
from microimpute.config import (
    QUANTILES,
    RANDOM_STATE,
    TRAIN_SIZE,
    VALIDATE_CONFIG,
)
from microimpute.models import OLS, QRF, Imputer, ImputerResults, QuantReg
from microimpute.utils.data import unnormalize_predictions

try:
    from microimpute.models import Matching

    HAS_MATCHING = True
except ImportError:
    HAS_MATCHING = False

log = logging.getLogger(__name__)


class AutoImputeResult(BaseModel):
    """
    Structured return value for `autoimpute`.

    Attributes
    ----------
    imputations : Dict[str, pd.DataFrame] or Dict[str, Dict[float, pd.DataFrame]]
        Mapping model name → {quantile → DataFrame of imputed cols}.
        By default this contains only the best model unless `impute_all=True`.
        By default, the quantile is 0.5 (median).
    receiver_data : pd.DataFrame
        Copy of the receiver data with the median-quantile imputations of the best performing model attached.
    fitted_models : Dict[str, Any]
        Mapping model name → fitted Imputer instance.
    cv_results : pd.DataFrame
        Cross-validation loss table (models as index, quantiles as columns)
        with an extra "mean_loss" column.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    imputations: Union[
        Dict[str, Dict[float, pd.DataFrame]], Dict[str, pd.DataFrame]
    ] = Field(...)
    receiver_data: pd.DataFrame = Field(...)
    fitted_models: Dict[str, Any] = Field(...)
    cv_results: pd.DataFrame = Field(...)


def _setup_logging(log_level: str) -> int:
    """Configure logging level.

    Args:
        log_level: String representation of log level.

    Returns:
        Numeric log level.
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    numeric_level = level_map[log_level]
    log.setLevel(numeric_level)
    warnings.filterwarnings("ignore")
    return numeric_level


def _evaluate_models_parallel(
    model_classes: List[Type[Imputer]],
    training_data: pd.DataFrame,
    predictors: List[str],
    imputed_variables: List[str],
    weight_col: Optional[str],
    quantiles: List[float],
    k_folds: int,
    random_state: int,
    tune_hyperparameters: bool,
    hyperparameters: Optional[Dict[str, Dict[str, Any]]],
    n_jobs: int = -1,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """Evaluate multiple models in parallel using cross-validation.

    Returns:
        Tuple of (results_dataframe, best_hyperparameters_dict or None)
    """
    # Check if Matching model is present (requires sequential processing)
    has_matching = any(model.__name__ == "Matching" for model in model_classes)
    if has_matching and n_jobs != 1:
        log.info(
            "Using sequential processing (n_jobs=1) because Matching model is present"
        )
        n_jobs = 1

    # Prepare tasks for parallel execution
    parallel_tasks = []
    for model in model_classes:
        model_hyperparams = None
        if hyperparameters and model.__name__ in hyperparameters:
            model_hyperparams = hyperparameters[model.__name__]

        parallel_tasks.append(
            (
                model,
                training_data,
                predictors,
                imputed_variables,
                weight_col,
                quantiles,
                k_folds,
                random_state,
                tune_hyperparameters,
                model_hyperparams,
            )
        )

    # Execute in parallel
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(lambda args: evaluate_model(*args))(task)
        for task in tqdm(parallel_tasks, desc="Evaluating models")
    )

    # Process results
    method_test_losses = {}
    best_hyperparams = {}

    if tune_hyperparameters:
        for result in results:
            if len(result) == 3:
                model_name, cv_result, best_params = result
                method_test_losses[model_name] = cv_result.loc["test"]
                if model_name in ["QRF", "Matching"]:
                    best_hyperparams[model_name] = best_params
            else:
                model_name, cv_result = result
                method_test_losses[model_name] = cv_result.loc["test"]
    else:
        for model_name, cv_result in results:
            method_test_losses[model_name] = cv_result.loc["test"]

    method_results_df = pd.DataFrame.from_dict(
        method_test_losses, orient="index"
    )

    return method_results_df, (
        best_hyperparams if tune_hyperparameters else None
    )


def _generate_imputations_for_all_models(
    model_classes: List[Type[Imputer]],
    best_method: str,
    training_data: pd.DataFrame,
    imputing_data: pd.DataFrame,
    predictors: List[str],
    imputed_variables: List[str],
    weight_col: Optional[str],
    imputation_q: float,
    normalize_data: bool,
    normalizing_params: Optional[dict],
    tune_hyperparameters: bool,
    hyperparams: Optional[Dict[str, Any]],
    log_level: str,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Generate imputations for all models when impute_all=True.

    Returns:
        Tuple of (imputations_dict, fitted_models_dict)
    """
    final_imputations_dict = {}
    fitted_models_dict = {}

    log.info("Generating imputations for all models using the full dataset.")

    for model_class in model_classes:
        model_name = model_class.__name__
        if model_name == best_method:
            continue  # Skip the best method as it's already done

        log.info(f"Generating imputations with {model_name}.")

        # Get model-specific hyperparameters if available
        model_hyperparams = None
        if tune_hyperparameters and hyperparams and model_name in hyperparams:
            model_hyperparams = hyperparams[model_name]

        # Fit and predict
        fitted_model, imputations = fit_and_predict_model(
            model_class,
            training_data,
            imputing_data,
            predictors,
            imputed_variables,
            weight_col,
            imputation_q,
            model_hyperparams,
            log_level,
        )

        # Unnormalize if needed
        if normalize_data and normalizing_params:
            final_imputations = unnormalize_predictions(
                imputations, normalizing_params
            )
        else:
            final_imputations = imputations

        final_imputations_dict[model_name] = final_imputations[imputation_q]
        fitted_models_dict[model_name] = fitted_model

    return final_imputations_dict, fitted_models_dict


@validate_call(config=VALIDATE_CONFIG)
def autoimpute(
    donor_data: pd.DataFrame,
    receiver_data: pd.DataFrame,
    predictors: List[str],
    imputed_variables: List[str],
    weight_col: Optional[str] = None,
    models: Optional[List[Type]] = None,
    imputation_quantiles: Optional[List[float]] = None,
    hyperparameters: Optional[Dict[str, Dict[str, Any]]] = None,
    tune_hyperparameters: Optional[bool] = False,
    normalize_data: Optional[bool] = False,
    impute_all: Optional[bool] = False,
    random_state: Optional[int] = RANDOM_STATE,
    train_size: Optional[float] = TRAIN_SIZE,
    k_folds: Optional[int] = 5,
    log_level: Optional[str] = "WARNING",
) -> AutoImputeResult:
    """Automatically select and apply the best imputation model.

    This function evaluates multiple imputation methods using cross-validation
    to determine which performs best on the provided donor data, then applies
    the winning method to impute values in the receiver data.

    Args:
        donor_data : Dataframe containing both predictor and target variables
            used  to train models
        receiver_data : Dataframe containing predictor variables where imputed
            values will be generated
        predictors : List of column names of predictor variables used to
            predict imputed variables
        imputed_variables : List of column names of variables to be imputed in
            the receiver data
        weight_col : Optional column name for sampling weights in donor data.
        models : List of imputer model classes to compare.
            If None, uses [QRF, OLS, QuantReg, Matching]
        imputation_quantiles : List of quantiles to predict for each imputed
            variable.Will use default QUANTILES if not passed.
        hyperparameters : Dictionary of hyperparameters for specific models,
            with model names as keys. Defaults to None and uses default model hyperparameters then.
        tune_hyperparameters : Whether to tune hyperparameters for the models.
            Defaults to False.
        normalize_data : If True, will normalize the data before imputation.
        impute_all : If True, will return final imputations for all models not
            just the best one.
        random_state : Random seed for reproducibility
        train_size : Proportion of data to use for training in preprocessing
        k_folds : Number of folds for cross-validation. Defaults to 5.
        log_level : Logging level for the function. Defaults to "WARNING".

    Returns:
        AutoImputeResult: A structured result containing:
            - imputations: Dict mapping model name(s) to quantile → DataFrame of imputed values
            - receiver_data: DataFrame with imputed values added
            - fitted_models: Dict mapping model name to ImputerResults instance(s)
            - cv_results: DataFrame of cross-validation losses for each model

    Raises:
        ValueError: If inputs are invalid (e.g., invalid quantiles, missing columns)
        RuntimeError: For unexpected errors during imputation
    """
    try:
        # Step 0: Setup and validation
        numeric_log_level = _setup_logging(log_level)

        # Create progress tracker if needed
        if numeric_log_level <= logging.INFO:
            main_progress = tqdm(total=5, desc="AutoImputation progress")
            main_progress.set_description("Input validation")

        # Use provided quantiles or defaults
        quantiles = imputation_quantiles if imputation_quantiles else QUANTILES

        # Validate all inputs
        validate_autoimpute_inputs(
            donor_data,
            receiver_data,
            predictors,
            imputed_variables,
            weight_col,
            quantiles,
            hyperparameters,
            tune_hyperparameters,
            log_level,
        )

        log.info(
            f"Generating imputations to impute from {len(donor_data)} donor data "
            f"to {len(receiver_data)} receiver data for variables {imputed_variables} "
            f"with predictors {predictors}."
        )

        # Step 1: Data preparation
        if numeric_log_level <= logging.INFO:
            log.info("Preprocessing data...")
            main_progress.update(1)
            main_progress.set_description("Data preparation")

        # Keep track of original imputed variable names
        original_imputed_variables = imputed_variables.copy()

        training_data, imputing_data, normalizing_params = (
            prepare_data_for_imputation(
                donor_data,
                receiver_data,
                predictors,
                imputed_variables,
                weight_col,
                normalize_data,
                train_size,
                1 - train_size,
            )
        )

        # Step 2: Model evaluation
        if numeric_log_level <= logging.INFO:
            main_progress.update(1)
            main_progress.set_description("Model evaluation")

        # Get model classes
        if not models:
            model_classes: List[Type[Imputer]] = [QRF, OLS, QuantReg]
            if HAS_MATCHING:
                model_classes.append(Matching)
        else:
            model_classes = models

        # Log hyperparameter usage
        if hyperparameters:
            model_names = [
                model_class.__name__ for model_class in model_classes
            ]
            for model_name, model_params in hyperparameters.items():
                if model_name in model_names:
                    log.info(
                        f"Using hyperparameters for {model_name}: {model_params}"
                    )
                else:
                    log.info(
                        f"Hyperparameters provided for {model_name} but model not in list: {model_names}"
                    )

        log.info(
            "Hyperparameter tuning and cross-validation for model comparison in progress..."
        )

        # Evaluate models in parallel
        method_results_df, best_hyperparams = _evaluate_models_parallel(
            model_classes,
            training_data,
            predictors,
            imputed_variables,
            weight_col,
            quantiles,
            k_folds,
            random_state,
            tune_hyperparameters,
            hyperparameters,
        )

        # Step 3: Model selection
        if numeric_log_level <= logging.INFO:
            main_progress.update(1)
            main_progress.set_description("Model selection")

        log.info(f"Comparing across {model_classes} methods.")
        best_method, best_row = select_best_model(method_results_df)

        # Step 4: Generate imputations with best method
        if numeric_log_level <= logging.INFO:
            main_progress.update(1)
            main_progress.set_description("Imputation")

        log.info(
            f"Generating imputations using the best method: {best_method} on the receiver data."
        )

        # Get the best model class
        models_dict = {model.__name__: model for model in model_classes}
        chosen_model = models_dict[best_method]

        # Default to median quantile for final imputation
        imputation_q = 0.5

        # Get hyperparameters for best model if tuned
        model_hyperparams = None
        if (
            tune_hyperparameters
            and best_hyperparams
            and best_method in best_hyperparams
        ):
            model_hyperparams = best_hyperparams[best_method]

        # Fit and predict with best model
        best_fitted_model, imputations = fit_and_predict_model(
            chosen_model,
            training_data,
            imputing_data,
            predictors,
            imputed_variables,
            weight_col,
            imputation_q,
            model_hyperparams,
            log_level,
        )

        # Unnormalize if needed
        if normalize_data and normalizing_params:
            final_imputations = unnormalize_predictions(
                imputations, normalizing_params
            )
        else:
            final_imputations = imputations

        log.info(
            f"Imputation generation completed for {len(receiver_data)} samples "
            f"using the best method: {best_method} and the median quantile."
        )

        # Add imputed values to receiver data
        median_imputations = final_imputations[imputation_q]
        for var in original_imputed_variables:
            if var in median_imputations.columns:
                receiver_data[var] = median_imputations[var]
            else:
                log.warning(
                    f"Imputed variable {var} not found in the imputations."
                )

        # Initialize results
        final_imputations_dict = {
            "best_method": (
                final_imputations[0.5]
                if imputation_quantiles is None
                else final_imputations
            )
        }
        fitted_models_dict = {"best_method": best_fitted_model}

        # Step 5: Generate imputations for all models if requested
        if impute_all:
            other_imputations, other_models = (
                _generate_imputations_for_all_models(
                    model_classes,
                    best_method,
                    training_data,
                    imputing_data,
                    predictors,
                    imputed_variables,
                    weight_col,
                    imputation_q,
                    normalize_data,
                    normalizing_params,
                    tune_hyperparameters,
                    best_hyperparams,
                    log_level,
                )
            )
            final_imputations_dict.update(other_imputations)
            fitted_models_dict.update(other_models)

        # Complete progress bar if used
        if numeric_log_level <= logging.INFO:
            main_progress.set_description("Complete")
            main_progress.close()

        return AutoImputeResult(
            imputations=final_imputations_dict,
            receiver_data=receiver_data,
            fitted_models=fitted_models_dict,
            cv_results=method_results_df,
        )

    except ValueError as e:
        # Re-raise validation errors directly
        raise e
    except (KeyError, TypeError, AttributeError) as e:
        log.error(f"Unexpected error during autoimputation: {str(e)}")
        raise RuntimeError(f"Failed to generate imputations: {str(e)}") from e
