"""Predictor analysis utilities for evaluating correlations and sensitivity in imputation models.

This module provides functions for analyzing predictor relationships and evaluating
the sensitivity of imputation models to predictor selection.
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

import joblib
import numpy as np
import pandas as pd
from pydantic import validate_call
from scipy.stats import spearmanr
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

from microimpute.comparisons.metrics import (
    compute_loss,
    get_metric_for_variable_type,
)
from microimpute.config import (
    QUANTILES,
    RANDOM_STATE,
    TRAIN_SIZE,
    VALIDATE_CONFIG,
)
from microimpute.models import Imputer, ImputerResults
from microimpute.utils.type_handling import (
    DummyVariableProcessor,
    VariableTypeDetector,
)

log = logging.getLogger(__name__)


@validate_call(config=VALIDATE_CONFIG)
def compute_predictor_correlations(
    data: pd.DataFrame,
    predictors: List[str],
    imputed_variables: Optional[List[str]] = None,
    method: str = "all",
) -> Dict[str, pd.DataFrame]:
    """Compute correlation matrices between predictors using multiple methods.

    This function analyzes relationships between predictor variables using
    Pearson correlation, Spearman correlation, and mutual information.
    Categorical variables are automatically encoded for correlation analysis.

    If imputed_variables are provided, also computes mutual information
    between each predictor and each target variable to identify which
    predictors are most informative for imputation.

    Args:
        data: DataFrame containing the data.
        predictors: List of predictor column names to analyze.
        imputed_variables: Optional list of target variables to compute
            predictor-target mutual information.
        method: Type of correlation to compute. Options are:
            - "all": Compute all correlation types (default)
            - "pearson": Only Pearson correlation
            - "spearman": Only Spearman correlation
            - "mutual_info": Only mutual information

    Returns:
        Dictionary containing correlation matrices:
            - "pearson": Pearson correlation matrix (if requested)
            - "spearman": Spearman correlation matrix (if requested)
            - "mutual_info": Normalized mutual information matrix (if requested)
            - "predictor_target_mi": DataFrame of MI between predictors and targets
                (only if imputed_variables provided and mutual_info requested)

    Raises:
        ValueError: If predictors are not found in data or method is invalid.

    Example:
        >>> data = pd.DataFrame({
        ...     'age': [25, 30, 35, 40],
        ...     'income': [30000, 45000, 55000, 70000],
        ...     'education': ['HS', 'BS', 'MS', 'PhD'],
        ...     'health_score': [7, 8, 6, 9]
        ... })
        >>> correlations = compute_predictor_correlations(
        ...     data,
        ...     predictors=['age', 'income', 'education'],
        ...     imputed_variables=['health_score']
        ... )
        >>> print(correlations['predictor_target_mi'])
    """
    # Validate inputs
    missing_predictors = set(predictors) - set(data.columns)
    if missing_predictors:
        raise ValueError(f"Predictors not found in data: {missing_predictors}")

    if imputed_variables:
        missing_targets = set(imputed_variables) - set(data.columns)
        if missing_targets:
            raise ValueError(
                f"Target variables not found in data: {missing_targets}"
            )

    valid_methods = ["all", "pearson", "spearman", "mutual_info"]
    if method not in valid_methods:
        raise ValueError(f"Invalid method. Choose from: {valid_methods}")

    # Prepare data - encode categorical variables
    detector = VariableTypeDetector()
    data_encoded = data[predictors].copy()

    # Track which variables are categorical for mutual information
    categorical_mask = {}

    for col in predictors:
        var_type, _ = detector.categorize_variable(data[col], col, log)
        categorical_mask[col] = var_type in [
            "categorical",
            "numeric_categorical",
            "bool",
        ]

        if categorical_mask[col]:
            # Use label encoding for correlation computation
            le = LabelEncoder()
            # Handle missing values by treating them as a separate category
            # Convert all to string first to avoid mixed type issues
            data_str = data[col].astype(str)
            data_encoded[col] = le.fit_transform(data_str)

    results = {}

    # Compute Pearson correlation
    if method in ["all", "pearson"]:
        results["pearson"] = data_encoded.corr(method="pearson")

    # Compute Spearman correlation
    if method in ["all", "spearman"]:
        results["spearman"] = data_encoded.corr(method="spearman")

    # Compute mutual information
    if method in ["all", "mutual_info"]:
        n_predictors = len(predictors)
        mi_matrix = pd.DataFrame(
            np.zeros((n_predictors, n_predictors)),
            index=predictors,
            columns=predictors,
        )

        for i, pred1 in enumerate(predictors):
            for j, pred2 in enumerate(predictors):
                if i == j:
                    mi_matrix.iloc[i, j] = 1.0
                elif j > i:
                    # Compute MI between pred1 and pred2
                    mi_value = _compute_mutual_information(
                        data_encoded[pred1].values,
                        data_encoded[pred2].values,
                        categorical_mask[pred2],
                    )
                    # Normalize by max possible MI (min of entropies)
                    # This makes it comparable to correlation coefficients
                    max_mi = min(
                        _compute_entropy(data_encoded[pred1].values),
                        _compute_entropy(data_encoded[pred2].values),
                    )
                    if max_mi > 0:
                        mi_normalized = mi_value / max_mi
                    else:
                        mi_normalized = 0.0

                    mi_matrix.iloc[i, j] = mi_normalized
                    mi_matrix.iloc[j, i] = mi_normalized

        results["mutual_info"] = mi_matrix

    # Compute predictor-target mutual information if requested
    if imputed_variables and method in ["all", "mutual_info"]:
        log.info(
            f"Computing mutual information between {len(predictors)} predictors and {len(imputed_variables)} targets"
        )

        # Initialize predictor-target MI matrix
        pred_target_mi = pd.DataFrame(
            index=predictors, columns=imputed_variables, dtype=float
        )

        # Prepare target variables - encode if categorical
        targets_encoded = {}
        target_is_categorical = {}

        for target in imputed_variables:
            var_type, _ = detector.categorize_variable(
                data[target], target, log
            )
            target_is_categorical[target] = var_type in [
                "categorical",
                "numeric_categorical",
                "bool",
            ]

            if target_is_categorical[target]:
                # Encode categorical targets
                le = LabelEncoder()
                targets_encoded[target] = le.fit_transform(
                    data[target].astype(str)
                )
            else:
                targets_encoded[target] = data[target].values

        # Compute MI between each predictor and each target
        for pred in predictors:
            for target in imputed_variables:
                # Use encoded predictor values
                pred_values = data_encoded[pred].values
                target_values = targets_encoded[target]

                # Compute mutual information
                mi_value = _compute_mutual_information(
                    pred_values, target_values, target_is_categorical[target]
                )

                # Optionally normalize by target entropy for comparability
                target_entropy = _compute_entropy(target_values)
                if target_entropy > 0:
                    mi_normalized = mi_value / target_entropy
                else:
                    mi_normalized = 0.0

                pred_target_mi.loc[pred, target] = mi_normalized

        results["predictor_target_mi"] = pred_target_mi

        log.info(f"Predictor-target MI computation complete")

    return results


@validate_call(config=VALIDATE_CONFIG)
def leave_one_out_analysis(
    data: pd.DataFrame,
    predictors: List[str],
    imputed_variables: List[str],
    model_class: Type[Imputer],
    weight_col: Optional[Union[str, np.ndarray, pd.Series]] = None,
    quantiles: List[float] = QUANTILES,
    train_size: float = TRAIN_SIZE,
    n_jobs: int = 1,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Evaluate the impact of removing each predictor one at a time.

    This function assesses the importance of each predictor by measuring
    the performance degradation when that predictor is removed from the model.

    Args:
        data: DataFrame containing the data.
        predictors: List of predictor column names.
        imputed_variables: List of variables to impute.
        model_class: The Imputer class to use for evaluation (e.g., OLS, QRF, QuantReg).
        weight_col: Optional column name or array of sampling weights.
        quantiles: List of quantiles for evaluation (default: [0.1, 0.5, 0.9]).
        train_size: Proportion of data to use for training (default: 0.8).
        n_jobs: Number of parallel jobs (-1 for all CPUs).
        random_state: Random state for reproducibility.

    Returns:
        DataFrame with columns:
            - predictor_removed: Name of the predictor that was removed
            - avg_quantile_loss: Average quantile loss without this predictor
            - avg_log_loss: Average log loss for categorical variables
            - loss_increase: Increase in loss compared to baseline
            - relative_impact: Percentage increase in loss

    Example:
        >>> from microimpute.models import QRF
        >>> results = leave_one_out_analysis(
        ...     data, predictors=['age', 'income'],
        ...     imputed_variables=['health_score'],
        ...     model_class=QRF
        ... )
        >>> print(results.sort_values('relative_impact', ascending=False))
    """
    # Split data
    train_data, test_data = train_test_split(
        data, train_size=train_size, random_state=random_state
    )

    # Compute baseline performance with all predictors
    log.info("Computing baseline performance with all predictors")
    baseline_losses = _evaluate_model_performance(
        train_data=train_data,
        test_data=test_data,
        predictors=predictors,
        imputed_variables=imputed_variables,
        model_class=model_class,
        weight_col=weight_col,
        quantiles=quantiles,
        random_state=random_state,
    )

    baseline_total = baseline_losses["quantile_loss"] + baseline_losses.get(
        "log_loss", 0
    )

    # Function to process each predictor removal
    def process_predictor(pred):
        reduced_predictors = [p for p in predictors if p != pred]

        if len(reduced_predictors) == 0:
            return {
                "predictor_removed": pred,
                "avg_quantile_loss": np.nan,
                "avg_log_loss": np.nan,
                "loss_increase": np.nan,
                "relative_impact": np.nan,
            }

        try:
            losses = _evaluate_model_performance(
                train_data=train_data,
                test_data=test_data,
                predictors=reduced_predictors,
                imputed_variables=imputed_variables,
                model_class=model_class,
                weight_col=weight_col,
                quantiles=quantiles,
                random_state=random_state,
            )

            total_loss = losses["quantile_loss"] + losses.get("log_loss", 0)
            loss_increase = total_loss - baseline_total

            return {
                "predictor_removed": pred,
                "avg_quantile_loss": losses["quantile_loss"],
                "avg_log_loss": losses.get("log_loss", 0),
                "loss_increase": loss_increase,
                "relative_impact": (
                    (loss_increase / baseline_total * 100)
                    if baseline_total > 0
                    else 0
                ),
            }

        except Exception as e:
            log.warning(
                f"Failed to evaluate model without predictor {pred}: {e}"
            )
            return {
                "predictor_removed": pred,
                "avg_quantile_loss": np.nan,
                "avg_log_loss": np.nan,
                "loss_increase": np.nan,
                "relative_impact": np.nan,
            }

    # Process all predictors
    if n_jobs == 1:
        results = [
            process_predictor(pred)
            for pred in tqdm(predictors, desc="Leave-one-out analysis")
        ]
    else:
        results = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(process_predictor)(pred)
            for pred in tqdm(predictors, desc="Leave-one-out analysis")
        )

    results_df = pd.DataFrame(results)
    results_df["baseline_quantile_loss"] = baseline_losses["quantile_loss"]
    results_df["baseline_log_loss"] = baseline_losses.get("log_loss", 0)

    # Sort by relative impact (most important predictors first)
    results_df = results_df.sort_values("relative_impact", ascending=False)

    return results_df


@validate_call(config=VALIDATE_CONFIG)
def progressive_predictor_inclusion(
    data: pd.DataFrame,
    predictors: List[str],
    imputed_variables: List[str],
    model_class: Type[Imputer],
    weight_col: Optional[Union[str, np.ndarray, pd.Series]] = None,
    quantiles: Optional[List[float]] = QUANTILES,
    train_size: Optional[float] = TRAIN_SIZE,
    max_predictors: Optional[int] = None,
    random_state: Optional[int] = RANDOM_STATE,
) -> Dict[str, Any]:
    """Add predictors one by one to find the optimal subset and ordering.

    This function starts with no predictors and progressively adds the predictor
    that provides the greatest performance improvement at each step.

    Args:
        data: DataFrame containing the data.
        predictors: List of candidate predictor column names.
        imputed_variables: List of variables to impute.
        model_class: The Imputer class to use for evaluation (e.g., OLS, QRF, QuantReg).
        weight_col: Optional column name or array of sampling weights.
        quantiles: List of quantiles for evaluation.
        train_size: Proportion of data to use for training.
        max_predictors: Maximum number of predictors to include (default: all).
        random_state: Random state for reproducibility.

    Returns:
        Dictionary containing:
            - results_df: DataFrame with columns ['step', 'predictor_added',
              'predictors_included', 'avg_quantile_loss', 'avg_log_loss',
              'cumulative_improvement', 'marginal_improvement']
            - optimal_subset: Predictors in the best performing subset
            - optimal_loss: Loss of the optimal subset

    Example:
        >>> from microimpute.models import QRF
        >>> results = progressive_predictor_inclusion(
        ...     data, predictors=['age', 'income', 'education'],
        ...     imputed_variables=['health_score'],
        ...     model_class=QRF
        ... )
        >>> print(f"Optimal subset: {results['optimal_subset']}")
        >>> print(results['results_df'])
    """
    if max_predictors is None:
        max_predictors = len(predictors)

    # Split data
    train_data, test_data = train_test_split(
        data, train_size=train_size, random_state=random_state
    )

    # Track results for DataFrame
    df_rows = []
    remaining_predictors = predictors.copy()
    selected_predictors = []
    previous_loss = float("inf")
    first_loss = None
    best_overall_loss = float("inf")
    best_subset = []

    for step in tqdm(
        range(min(max_predictors, len(predictors))),
        desc="Progressive inclusion",
    ):

        best_predictor = None
        best_step_loss = float("inf")
        best_losses_detail = {}

        # Try each remaining predictor
        for pred in remaining_predictors:
            test_predictors = selected_predictors + [pred]

            try:
                losses = _evaluate_model_performance(
                    train_data=train_data,
                    test_data=test_data,
                    predictors=test_predictors,
                    imputed_variables=imputed_variables,
                    model_class=model_class,
                    weight_col=weight_col,
                    quantiles=quantiles,
                    random_state=random_state,
                )

                total_loss = losses["quantile_loss"] + losses.get(
                    "log_loss", 0
                )

                if total_loss < best_step_loss:
                    best_step_loss = total_loss
                    best_predictor = pred
                    best_losses_detail = losses

            except Exception as e:
                log.warning(
                    f"Failed to evaluate model with predictors {test_predictors}: {e}"
                )
                continue

        # Add the best predictor if found
        if best_predictor is not None:
            selected_predictors.append(best_predictor)
            remaining_predictors.remove(best_predictor)

            # Track first loss for cumulative improvement calculation
            if first_loss is None:
                first_loss = best_step_loss

            # Calculate improvements
            marginal_improvement = (
                previous_loss - best_step_loss
                if previous_loss != float("inf")
                else 0.0
            )
            cumulative_improvement = first_loss - best_step_loss

            # Add row to DataFrame
            df_rows.append(
                {
                    "step": step + 1,
                    "predictor_added": best_predictor,
                    "predictors_included": selected_predictors.copy(),
                    "avg_quantile_loss": best_losses_detail["quantile_loss"],
                    "avg_log_loss": best_losses_detail.get("log_loss", 0),
                    "cumulative_improvement": cumulative_improvement,
                    "marginal_improvement": marginal_improvement,
                }
            )

            # Track best overall subset
            if best_step_loss < best_overall_loss:
                best_overall_loss = best_step_loss
                best_subset = selected_predictors.copy()

            previous_loss = best_step_loss

            log.info(
                f"Step {step + 1}: Added '{best_predictor}', "
                f"loss = {best_step_loss:.6f} (improvement: {marginal_improvement:.6f})"
            )
        else:
            log.warning(f"No valid predictor found at step {step + 1}")
            break

    # Create DataFrame from rows
    results_df = pd.DataFrame(df_rows)

    return {
        "results_df": results_df,
        "optimal_subset": best_subset,
        "optimal_loss": best_overall_loss,
    }


# Helper functions


def _compute_mutual_information(
    x: np.ndarray, y: np.ndarray, y_is_categorical: bool
) -> float:
    """Compute mutual information between two variables."""
    # Remove any rows where either variable is NaN
    mask = ~(pd.isna(x) | pd.isna(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) == 0:
        return 0.0

    # Reshape for sklearn
    x_clean = x_clean.reshape(-1, 1)

    # Use appropriate MI function based on target type
    if y_is_categorical:
        mi = mutual_info_classif(x_clean, y_clean, random_state=RANDOM_STATE)[
            0
        ]
    else:
        mi = mutual_info_regression(
            x_clean, y_clean, random_state=RANDOM_STATE
        )[0]

    return mi


def _compute_entropy(x: np.ndarray) -> float:
    """Compute entropy of a variable."""
    # Remove NaN values
    x_clean = x[~pd.isna(x)]

    if len(x_clean) == 0:
        return 0.0

    # Compute probability distribution
    _, counts = np.unique(x_clean, return_counts=True)
    probs = counts / counts.sum()

    # Compute entropy
    entropy = -np.sum(probs * np.log2(probs + 1e-10))

    return entropy


def _evaluate_model_performance(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    predictors: List[str],
    imputed_variables: List[str],
    model_class: Type[Imputer],
    weight_col: Optional[Union[str, np.ndarray, pd.Series]],
    quantiles: List[float],
    random_state: int,
) -> Dict[str, float]:
    """Train a model and evaluate its performance."""
    try:
        # Initialize and fit the model
        model = model_class()
        fitted_model = model.fit(
            X_train=train_data,
            predictors=predictors,
            imputed_variables=imputed_variables,
            weight_col=weight_col,
        )

        # Get predictions
        predictions = fitted_model.predict(test_data, quantiles)

        # Compute losses
        losses = _compute_losses_from_predictions(
            predictions=predictions,
            true_data=test_data,
            imputed_variables=imputed_variables,
            quantiles=quantiles,
        )

        return losses

    except Exception as e:
        log.error(f"Model evaluation failed: {e}")
        raise


def _compute_losses_from_predictions(
    predictions: Dict[float, pd.DataFrame],
    true_data: pd.DataFrame,
    imputed_variables: List[str],
    quantiles: List[float],
) -> Dict[str, float]:
    """Compute losses from model predictions."""
    quantile_losses = []
    log_losses = []

    for quantile in quantiles:
        if quantile not in predictions:
            continue

        for var in imputed_variables:
            if var not in predictions[quantile].columns:
                continue

            # Get true and predicted values
            true_values = true_data[var]
            pred_values = predictions[quantile][var]

            # Determine appropriate loss metric
            metric_type = get_metric_for_variable_type(true_values, var)

            # Compute loss (returns tuple of element-wise losses and mean)
            _, mean_loss = compute_loss(
                test_y=true_values.values,
                imputations=pred_values.values,
                metric=metric_type,
                q=quantile,
            )

            if metric_type == "quantile_loss":
                quantile_losses.append(mean_loss)
            else:  # log_loss for categorical
                log_losses.append(mean_loss)

    results = {}
    if quantile_losses:
        results["quantile_loss"] = np.mean(quantile_losses)
    else:
        results["quantile_loss"] = 0.0

    if log_losses:
        results["log_loss"] = np.mean(log_losses)

    return results
