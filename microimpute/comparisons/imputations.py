"""Imputation generation utilities.

This module handles the generation of imputations using multiple model classes.
It provides functions to generate predictions at different quantiles,
and organize results in a consistent format for comparison.
"""

import logging
from typing import Any, Dict, List, Optional, Type

import pandas as pd
from pydantic import validate_call

from microimpute.comparisons.validation import (
    validate_columns_exist,
    validate_quantiles,
)
from microimpute.config import QUANTILES, VALIDATE_CONFIG
from microimpute.models.quantreg import QuantReg

log = logging.getLogger(__name__)


@validate_call(config=VALIDATE_CONFIG)
def get_imputations(
    model_classes: List[Type],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    predictors: List[str],
    imputed_variables: List[str],
    quantiles: Optional[List[float]] = QUANTILES,
) -> Dict[str, Dict[float, pd.DataFrame]]:
    """Generate imputations using multiple model classes for the specified variables.

    Args:
        model_classes: List of model classes to use (e.g., QRF, OLS, QuantReg, Matching).
        X_train: Training data containing predictors and variables to impute.
        X_test: Test data on which to make imputations.
        predictors: Names of columns to use as predictors.
        imputed_variables: Names of columns to impute.
        quantiles: List of quantiles to predict.

    Returns:
        Nested dictionary mapping method names to dictionaries mapping quantiles to imputations.

    Raises:
        ValueError: If input data is invalid or missing required columns.
        RuntimeError: If model fitting or prediction fails.
    """
    try:
        # Input validation
        if not model_classes:
            error_msg = "model_classes list is empty"
            log.error(error_msg)
            raise ValueError(error_msg)

        # Validate columns exist
        validate_columns_exist(X_train, predictors, "training data")
        validate_columns_exist(X_train, imputed_variables, "training data")
        validate_columns_exist(X_test, predictors, "test data")

        # Validate quantiles if provided
        if quantiles:
            validate_quantiles(quantiles)

        log.info(
            f"Generating imputations for {len(model_classes)} model classes"
        )
        log.info(
            f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}"
        )
        log.info(
            f"Using {len(predictors)} predictors and imputing {len(imputed_variables)} variables"
        )
        log.info(
            f"Evaluating at {len(quantiles) if quantiles else 'default'} quantiles"
        )

        method_imputations: Dict[str, Dict[float, Any]] = {}

        for model_class in model_classes:
            model_name = model_class.__name__
            log.info(f"Processing model: {model_name}")

            try:
                # Instantiate the model
                model = model_class()

                # Handle QuantReg which needs quantiles during fitting
                if model_class == QuantReg:
                    log.info(f"Fitting {model_name} with explicit quantiles")
                    fitted_model = model.fit(
                        X_train,
                        predictors,
                        imputed_variables,
                        quantiles=quantiles,
                    )
                else:
                    log.info(f"Fitting {model_name}")
                    fitted_model = model.fit(
                        X_train, predictors, imputed_variables
                    )

                # Get predictions
                log.info(f"Generating predictions with {model_name}")
                imputations = fitted_model.predict(X_test, quantiles)
                method_imputations[model_name] = imputations

            except (TypeError, AttributeError, ValueError) as model_error:
                log.error(
                    f"Error processing model {model_name}: {str(model_error)}"
                )
                raise RuntimeError(
                    f"Failed to process model {model_name}: {str(model_error)}"
                ) from model_error

        log.info(
            f"Successfully generated imputations for all {len(model_classes)} models"
        )
        return method_imputations

    except ValueError as e:
        # Re-raise validation errors directly
        raise e
    except (KeyError, TypeError, AttributeError) as e:
        log.error(f"Unexpected error during imputation generation: {str(e)}")
        raise RuntimeError(f"Failed to generate imputations: {str(e)}") from e
