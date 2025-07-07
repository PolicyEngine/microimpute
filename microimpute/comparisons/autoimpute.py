"""
Pipeline for autoimputation of missing values in a dataset.
This module integrates all steps necessary for method selection and imputation of missing values.
"""

import logging
import warnings
from functools import partial
from typing import Any, Dict, List, Optional, Type, Union

import joblib
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, validate_call
from tqdm.auto import tqdm

from microimpute.comparisons import *
from microimpute.comparisons.data import preprocess_data
from microimpute.config import (
    QUANTILES,
    RANDOM_STATE,
    TRAIN_SIZE,
    VALIDATE_CONFIG,
)
from microimpute.evaluations import cross_validate_model
from microimpute.models import *

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
        with an extra “mean_loss” column.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    imputations: Union[
        Dict[str, Dict[float, pd.DataFrame]], Dict[str, pd.DataFrame]
    ] = Field(...)
    receiver_data: pd.DataFrame = Field(...)
    fitted_models: Dict[str, Any] = Field(...)
    cv_results: pd.DataFrame = Field(...)


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
    # Set up logging level
    if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        error_msg = f"Invalid log_level: {log_level}. Must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL."
        log.error(error_msg)
        raise ValueError(error_msg)
    if log_level == "DEBUG":
        log_level = logging.DEBUG
    elif log_level == "INFO":
        log_level = logging.INFO
    elif log_level == "WARNING":
        log_level = logging.WARNING
    elif log_level == "ERROR":
        log_level = logging.ERROR
    elif log_level == "CRITICAL":
        log_level = logging.CRITICAL
    log.setLevel(log_level)
    warnings.filterwarnings("ignore")

    # Set up parallel processing
    n_jobs: Optional[int] = -1

    # Create a progress tracking system
    if log_level == "INFO" or log_level == "DEBUG":
        main_progress = tqdm(total=4, desc="AutoImputation progress")
        main_progress.set_description("Input validation")

    if imputation_quantiles is None:
        quantiles = QUANTILES  # Use default quantiles if not provided
    else:
        quantiles = imputation_quantiles

    # Step 0: Input validation
    try:
        # Validate quantiles if provided
        if quantiles:
            invalid_quantiles = [q for q in quantiles if not 0 <= q <= 1]
            if invalid_quantiles:
                error_msg = f"Invalid quantiles (must be between 0 and 1): {invalid_quantiles}"
                log.error(error_msg)
                raise ValueError(error_msg)

        # Validate that predictor and imputed variable columns exist in donor data
        missing_predictors_donor = [
            col for col in predictors if col not in donor_data.columns
        ]
        if missing_predictors_donor:
            error_msg = f"Missing predictor columns in donor data: {missing_predictors_donor}"
            log.error(error_msg)
            raise ValueError(error_msg)

        missing_predictors_receiver = [
            col for col in predictors if col not in receiver_data.columns
        ]
        if missing_predictors_receiver:
            error_msg = f"Missing predictor columns in reciver data: {missing_predictors_receiver}"
            log.error(error_msg)
            raise ValueError(error_msg)

        missing_imputed_donor = [
            col for col in imputed_variables if col not in donor_data.columns
        ]
        if missing_imputed_donor:
            error_msg = f"Missing imputed variable columns in donor data: {missing_imputed_donor}"
            log.error(error_msg)
            raise ValueError(error_msg)

        # Validate that predictor columns exist in receiver data (imputed variables may not be present in receiver data)
        missing_predictors_receiver = [
            col for col in predictors if col not in receiver_data.columns
        ]
        if missing_predictors_receiver:
            error_msg = f"Missing predictor columns in test data: {missing_predictors_receiver}"
            log.error(error_msg)
            raise ValueError(error_msg)

        log.info(
            f"Generating imputations to impute from {len(donor_data)} donor data to {len(receiver_data)} receiver data for variables {imputed_variables} with predictors {predictors}. "
        )

        if (hyperparameters is not None) and (tune_hyperparameters == True):
            error_msg = "Cannot specify both model_hyperparams and request to automatically tune hyperparameters, please select one or the other."
            log.error(error_msg)
            raise ValueError(error_msg)

        # Step 1: Data preparation
        if log_level == "INFO" or log_level == "DEBUG":
            log.info("Preprocessing data...")
            main_progress.update(1)
            main_progress.set_description("Data preparation")

        # If imputed variables are in receiver data, remove them
        receiver_data = receiver_data.drop(
            columns=imputed_variables, errors="ignore"
        )

        training_data = donor_data.copy()
        imputing_data = receiver_data.copy()

        # Keep track of original imputed variable names for final assignment
        original_imputed_variables = imputed_variables.copy()

        if normalize_data:
            training_data_predictors, _ = preprocess_data(
                training_data[predictors],
                full_data=True,
                train_size=train_size,
                test_size=(1 - train_size),
                normalize=normalize_data,
            )
            (
                training_data_imputed_variables,
                normalizing_params,
            ) = preprocess_data(
                training_data[imputed_variables],
                full_data=True,
                train_size=train_size,
                test_size=(1 - train_size),
                normalize=normalize_data,
            )
            imputing_data, _ = preprocess_data(
                imputing_data[predictors],
                full_data=True,
                train_size=train_size,
                test_size=(1 - train_size),
                normalize=normalize_data,
            )

            training_data = training_data_predictors.join(
                training_data_imputed_variables
            )
            if weight_col:
                training_data[weight_col] = donor_data[weight_col]
        else:
            training_data_predictors = preprocess_data(
                training_data[predictors],
                full_data=True,
                train_size=train_size,
                test_size=(1 - train_size),
                normalize=normalize_data,
            )
            training_data_imputed_variables = preprocess_data(
                training_data[imputed_variables],
                full_data=True,
                train_size=train_size,
                test_size=(1 - train_size),
                normalize=normalize_data,
            )
            imputing_data = preprocess_data(  # _
                imputing_data[predictors],
                full_data=True,
                train_size=train_size,
                test_size=(1 - train_size),
                normalize=normalize_data,
            )
            training_data = training_data_predictors.join(
                training_data_imputed_variables
            )
            if weight_col:
                training_data[weight_col] = donor_data[weight_col]

        # Step 2: Imputation with each method
        if log_level == "INFO" or log_level == "DEBUG":
            main_progress.update(1)
            main_progress.set_description("Model evaluation")

        if not models:
            # If no models are provided, use default models
            model_classes: List[Type[Imputer]] = [QRF, OLS, QuantReg, Matching]
        else:
            model_classes = models

        if hyperparameters:
            model_names = [
                model_class.__name__ for model_class in model_classes
            ]
            for model_name, model_params in hyperparameters.items():
                if model_name in model_names:
                    # Update the model class with the provided hyperparameters
                    if model_name == "QRF":
                        log.info(
                            f"Using hyperparameters for QRF: {model_params}"
                        )
                    elif model_name == "Matching":
                        log.info(
                            f"Using hyperparameters for Matching: {model_params}"
                        )
                else:
                    log.info(
                        f"None of the hyperparameters provided are relevant for the supported models: {model_names}. Using default hyperparameters."
                    )

        method_test_losses = {}
        log.info(
            "Hyperparameter tuning and cross-validation for model comparisson in progress... "
        )

        def evaluate_model(
            model: Type[Imputer],
            data: pd.DataFrame,
            predictors: List[str],
            imputed_variables: List[str],
            weight_col: Optional[str],
            quantiles: List[float],
            k_folds: Optional[int] = 5,
            random_state: Optional[bool] = RANDOM_STATE,
            tune_hyperparams: Optional[bool] = True,
            hyperparameters: Optional[Dict[str, Any]] = None,
        ) -> tuple[str, pd.DataFrame]:
            """Evaluate a single imputation model with cross-validation.

            Args:
                model: The imputation model class to evaluate
                data: The dataset to use for evaluation
                predictors: List of predictor column names
                imputed_variables: List of columns to impute
                quantiles: List of quantiles to evaluate
                k_folds: Number of cross-validation folds
                random_state: Random seed for reproducibility
                tune_hyperparams: Whether to tune hyperparameters
                hyperparameters: Optional model-specific hyperparameters

            Returns:
                Tuple containing model name and cross-validation results DataFrame
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

        # Special handling for models that use rpy2
        # Use sequential processing for Matching model to avoid thread context issues
        has_matching = any(
            model.__name__ == "Matching" for model in model_classes
        )
        if has_matching and n_jobs != 1:
            log.info(
                "Using sequential processing (n_jobs=1) because Matching model is present"
            )
            n_jobs = 1

        parallel_tasks = []
        for model in model_classes:
            parallel_tasks.append(
                (
                    model,
                    training_data,
                    predictors,
                    imputed_variables,
                    weight_col,
                    quantiles,
                    k_folds,
                    RANDOM_STATE,
                    tune_hyperparameters,
                    hyperparameters,
                )
            )

        # Execute in parallel
        results = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(lambda args: evaluate_model(*args))(task)
            for task in tqdm(parallel_tasks, desc="Evaluating models")
        )

        # Process results
        if tune_hyperparameters == True:
            hyperparams = {}
            for model_name, cv_result, best_tuned_hyperparams in results:
                method_test_losses[model_name] = cv_result.loc["test"]
                if model_name == "QRF" or model_name == "Matching":
                    hyperparams[model_name] = best_tuned_hyperparams
        else:
            for model_name, cv_result in results:
                method_test_losses[model_name] = cv_result.loc["test"]

        method_results_df = pd.DataFrame.from_dict(
            method_test_losses, orient="index"
        )

        # Step 3: Compare imputation methods
        log.info(f"Comparing across {model_classes} methods. ")

        if log_level == "INFO" or log_level == "DEBUG":
            main_progress.update(1)
            main_progress.set_description("Model selection")

        # add a column called "mean_loss" with the average loss across quantiles
        method_results_df["mean_loss"] = method_results_df.mean(axis=1)

        # Step 4: Select best method
        best_method = method_results_df["mean_loss"].idxmin()
        best_row = method_results_df.loc[best_method]

        log.info(
            f"The method with the lowest average loss is {best_method}, with an average loss across variables and quantiles of {best_row['mean_loss']}. "
        )

        # Step 5: Generate imputations with the best method on the receiver data
        log.info(
            f"Generating imputations using the best method: {best_method} on the receiver data. "
        )

        if log_level == "INFO" or log_level == "DEBUG":
            main_progress.update(1)
            main_progress.set_description("Imputation")

        models_dict = {model.__name__: model for model in model_classes}
        chosen_model = models_dict[best_method]

        # Initialize the model
        model = chosen_model(log_level=log_level)
        imputation_q = 0.5  # this can be an input parameter, or if unspecified will default to a random quantile
        # Fit the model
        if best_method == "QuantReg":
            # For QuantReg, we need to explicitly fit the quantile
            best_fitted_model = model.fit(
                training_data,
                predictors,
                imputed_variables,
                weight_col=weight_col,
                quantiles=[imputation_q],
            )
        else:
            if (
                chosen_model.__name__ == "Matching"
                or chosen_model.__name__ == "QRF"
            ) and tune_hyperparameters == True:
                # For Matching and QRF, we need to pass the best hyperparameters
                best_fitted_model = model.fit(
                    training_data,
                    predictors,
                    imputed_variables,
                    weight_col=weight_col,
                    **hyperparams[chosen_model.__name__],
                )
            else:
                best_fitted_model = model.fit(
                    training_data,
                    predictors,
                    imputed_variables,
                    weight_col=weight_col,
                )

        # Predict with explicit quantiles
        imputations = best_fitted_model.predict(
            imputing_data, quantiles=[imputation_q]
        )

        if normalize_data:
            # Unnormalize the imputations
            mean = pd.Series(
                {col: p["mean"] for col, p in normalizing_params.items()}
            )
            std = pd.Series(
                {col: p["std"] for col, p in normalizing_params.items()}
            )
            unnormalized_imputations = {}
            for q, df in imputations.items():
                cols = df.columns  # the imputed variables
                df_unnorm = df.mul(std[cols], axis=1)  # × std
                df_unnorm = df_unnorm.add(mean[cols], axis=1)  # + mean
                unnormalized_imputations[q] = df_unnorm
            final_imputations = unnormalized_imputations
        else:
            final_imputations = imputations

        log.info(
            f"Imputation generation completed for {len(receiver_data)} samples using the best method: {best_method} and the median quantile. "
        )

        if log_level == "INFO" or log_level == "DEBUG":
            main_progress.set_description("Complete")
            main_progress.close()

        median_imputations = final_imputations[
            0.5
        ]  # this may not work if we change the value of imputation_q
        # Add the imputed variables to the receiver data
        try:
            missing_imputed_vars = []
            for var in original_imputed_variables:
                if var in median_imputations.columns:
                    receiver_data[var] = median_imputations[var]
                else:
                    missing_imputed_vars.append(var)
                    log.warning(
                        f"Imputed variable {var} not found in the imputations. "
                    )
        except KeyError as e:
            error_msg = f"Missing imputed variable in the imputations: {e}"
            log.error(error_msg)
            raise ValueError(error_msg)

        final_imputations_dict = {}
        fitted_models_dict = {}
        final_imputations_dict["best_method"] = (
            final_imputations[0.5]
            if imputation_quantiles is None
            else final_imputations
        )
        fitted_models_dict["best_method"] = best_fitted_model

        # Step 6: If impute_all is True, impute using the full dataset with all the remaining models
        if impute_all:
            log.info(
                "Generating imputations for all the remaining models using the full dataset. "
            )
            for model_class in model_classes:
                model_name = model_class.__name__
                if model_name != best_method:
                    log.info(f"Generating imputations with {model_name}. ")
                    model = model_class(log_level=log_level)
                    if model_name == "QuantReg":
                        # For QuantReg, we need to explicitly fit the quantile
                        fitted_model = model.fit(
                            training_data,
                            predictors,
                            imputed_variables,
                            weight_col=weight_col,
                            quantiles=[imputation_q],
                        )
                    else:
                        if (
                            model_name == "Matching" or model_name == "QRF"
                        ) and tune_hyperparameters == True:
                            # For Matching and QRF, we need to pass the best hyperparameters
                            fitted_model = model.fit(
                                training_data,
                                predictors,
                                imputed_variables,
                                weight_col=weight_col,
                                **hyperparams[model_name],
                            )
                        else:
                            fitted_model = model.fit(
                                training_data,
                                predictors,
                                imputed_variables,
                                weight_col=weight_col,
                            )

                    # Predict with explicit quantiles
                    imputations = fitted_model.predict(
                        imputing_data, quantiles=[imputation_q]
                    )

                    if normalize_data:
                        unnormalized_imputations = {}
                        for q, df in imputations.items():
                            cols = df.columns  # the imputed variables
                            df_unnorm = df.mul(std[cols], axis=1)  # × std
                            df_unnorm = df_unnorm.add(
                                mean[cols], axis=1
                            )  # + mean
                            unnormalized_imputations[q] = df_unnorm
                        final_imputations = unnormalized_imputations
                    else:
                        final_imputations = imputations
                    final_imputations_dict[model_name] = (
                        final_imputations[0.5]
                        if imputation_quantiles is None
                        else final_imputations
                    )

                    log.warning(final_imputations)

                    fitted_models_dict[model_name] = fitted_model

        return AutoImputeResult(
            imputations=final_imputations_dict,
            receiver_data=receiver_data,
            fitted_models=fitted_models_dict,
            cv_results=method_results_df,
        )

    except ValueError as e:
        # Re-raise validation errors directly
        raise e
    except Exception as e:
        log.error(f"Unexpected error during autoimputation: {str(e)}")
        raise RuntimeError(f"Failed to generate imputations: {str(e)}") from e
