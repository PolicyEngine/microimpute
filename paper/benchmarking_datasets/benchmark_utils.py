"""Benchmark utilities for CIA sensitivity analysis and cross-dataset comparison.

This module provides functions for:
1. CIA (Conditional Independence Assumption) sensitivity analysis via progressive
   predictor exclusion
2. Cross-dataset results visualization (summary tables and heatmaps)
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.integrate import trapezoid
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from microimpute.comparisons.metrics import quantile_loss
from microimpute.config import (
    PLOT_CONFIG,
    RANDOM_STATE,
    TRAIN_SIZE,
)
from microimpute.models import Imputer

log = logging.getLogger(__name__)

# Method colors from PLOT_CONFIG (Safe palette - colorblind-friendly)
METHOD_COLORS = {
    "QRF": "#88CCEE",  # Cyan
    "OLS": "#CC6677",  # Rose
    "QuantReg": "#DDCC77",  # Sand
    "Matching": "#117733",  # Green
    "MDN": "#332288",  # Indigo
}

# Use a simple set of quantiles for CIA analysis to avoid QuantReg issues
CIA_QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9]


def progressive_predictor_exclusion(
    data: pd.DataFrame,
    predictors: List[str],
    imputed_variables: List[str],
    model_class: Type[Imputer],
    ordering: str = "mutual_info",
    weight_col: Optional[Union[str, np.ndarray, pd.Series]] = None,
    quantiles: Optional[List[float]] = None,
    train_size: float = TRAIN_SIZE,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Progressively REMOVE predictors to measure CIA sensitivity.

    This function assesses sensitivity to the Conditional Independence Assumption
    by measuring how performance degrades when key predictors are unavailable
    (simulating incomplete linking variables between surveys).

    Args:
        data: DataFrame containing the data.
        predictors: List of predictor column names.
        imputed_variables: List of variables to impute.
        model_class: The Imputer class to use for evaluation.
        ordering: How to order predictor removal:
            - "mutual_info": Remove most informative first (default)
            - "correlation": Remove highest correlated first
            - "random": Remove in random order (control)
        weight_col: Optional column name or array of sampling weights.
        quantiles: List of quantiles for evaluation (default: [0.1, 0.3, 0.5, 0.7, 0.9]).
        train_size: Proportion of data to use for training.
        random_state: Random state for reproducibility.

    Returns:
        Dictionary containing:
            - results_df: DataFrame with columns ['step', 'predictor_removed',
              'remaining_predictors', 'quantile_loss', 'normalized_loss']
            - predictor_order: List of predictors in removal order
            - baseline_loss: Performance with all predictors
            - sensitivity_score: Area under degradation curve (higher = more sensitive)
    """
    if quantiles is None:
        quantiles = CIA_QUANTILES

    # Split data
    train_data, test_data = train_test_split(
        data, train_size=train_size, random_state=random_state
    )

    # Order predictors by importance
    predictor_order = _order_predictors(
        train_data, predictors, imputed_variables, ordering, random_state
    )

    # Compute baseline performance with all predictors
    log.info("Computing baseline performance with all predictors")
    baseline_loss = _evaluate_model(
        train_data=train_data,
        test_data=test_data,
        predictors=predictors,
        imputed_variables=imputed_variables,
        model_class=model_class,
        weight_col=weight_col,
        quantiles=quantiles,
    )

    # Track results
    results = []
    current_predictors = predictors.copy()

    # Add baseline (step 0, no predictors removed)
    results.append(
        {
            "step": 0,
            "predictor_removed": None,
            "remaining_predictors": current_predictors.copy(),
            "num_predictors": len(current_predictors),
            "quantile_loss": baseline_loss,
            "normalized_loss": 1.0,
        }
    )

    # Progressively remove predictors
    for step, pred_to_remove in enumerate(
        tqdm(predictor_order, desc="Progressive exclusion"), start=1
    ):
        current_predictors = [
            p for p in current_predictors if p != pred_to_remove
        ]

        if len(current_predictors) == 0:
            # No predictors left - record as maximum degradation
            results.append(
                {
                    "step": step,
                    "predictor_removed": pred_to_remove,
                    "remaining_predictors": [],
                    "num_predictors": 0,
                    "quantile_loss": np.nan,
                    "normalized_loss": np.nan,
                }
            )
            break

        try:
            loss = _evaluate_model(
                train_data=train_data,
                test_data=test_data,
                predictors=current_predictors,
                imputed_variables=imputed_variables,
                model_class=model_class,
                weight_col=weight_col,
                quantiles=quantiles,
            )

            normalized_loss = (
                loss / baseline_loss if baseline_loss > 0 else np.nan
            )

            results.append(
                {
                    "step": step,
                    "predictor_removed": pred_to_remove,
                    "remaining_predictors": current_predictors.copy(),
                    "num_predictors": len(current_predictors),
                    "quantile_loss": loss,
                    "normalized_loss": normalized_loss,
                }
            )

        except Exception as e:
            log.warning(
                f"Failed to evaluate after removing {pred_to_remove}: {e}"
            )
            results.append(
                {
                    "step": step,
                    "predictor_removed": pred_to_remove,
                    "remaining_predictors": current_predictors.copy(),
                    "num_predictors": len(current_predictors),
                    "quantile_loss": np.nan,
                    "normalized_loss": np.nan,
                }
            )

    results_df = pd.DataFrame(results)

    # Compute sensitivity score (AUC of normalized loss curve)
    # Higher = more sensitive to predictor removal
    valid_results = results_df[results_df["normalized_loss"].notna()]
    if len(valid_results) > 1:
        # Use trapezoidal rule for AUC calculation
        x = valid_results["step"].values
        y = valid_results["normalized_loss"].values
        # Normalize x to [0, 1] range
        x_norm = x / x.max() if x.max() > 0 else x
        sensitivity_score = trapezoid(y, x_norm)
    else:
        sensitivity_score = 1.0

    return {
        "results_df": results_df,
        "predictor_order": predictor_order,
        "baseline_loss": baseline_loss,
        "sensitivity_score": sensitivity_score,
    }


def _order_predictors(
    data: pd.DataFrame,
    predictors: List[str],
    imputed_variables: List[str],
    ordering: str,
    random_state: int,
) -> List[str]:
    """Order predictors by importance for removal.

    Args:
        data: Training data.
        predictors: List of predictor names.
        imputed_variables: Target variables.
        ordering: Ordering method ("mutual_info", "correlation", "random").
        random_state: Random state.

    Returns:
        List of predictors ordered by importance (most important first).
    """
    if ordering == "random":
        rng = np.random.RandomState(random_state)
        order = predictors.copy()
        rng.shuffle(order)
        return order

    # Compute importance scores
    importance_scores = {}

    for pred in predictors:
        scores = []
        for target in imputed_variables:
            # Get predictor values - encode categorical if needed
            X_series = data[pred]
            if X_series.dtype == "object" or str(X_series.dtype) == "category":
                # Encode categorical predictor as numeric codes
                X = pd.Categorical(X_series).codes.reshape(-1, 1)
            else:
                X = X_series.values.reshape(-1, 1)

            y = data[target].values

            # Handle missing values using pd.isna (works with all types)
            mask = ~(pd.isna(X.flatten()) | pd.isna(y))
            X_clean = X[mask].astype(float)
            y_clean = y[mask].astype(float)

            if len(X_clean) < 10:
                continue

            if ordering == "mutual_info":
                mi = mutual_info_regression(
                    X_clean, y_clean, random_state=random_state
                )[0]
                scores.append(mi)
            elif ordering == "correlation":
                corr = np.abs(np.corrcoef(X_clean.flatten(), y_clean)[0, 1])
                scores.append(corr if not np.isnan(corr) else 0)

        importance_scores[pred] = np.mean(scores) if scores else 0

    # Sort by importance (highest first - these will be removed first)
    sorted_preds = sorted(
        importance_scores.keys(),
        key=lambda x: importance_scores[x],
        reverse=True,
    )

    return sorted_preds


def _evaluate_model(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    predictors: List[str],
    imputed_variables: List[str],
    model_class: Type[Imputer],
    weight_col: Optional[Union[str, np.ndarray, pd.Series]],
    quantiles: List[float],
) -> float:
    """Train a model and evaluate its quantile loss.

    Returns:
        Mean quantile loss across all quantiles and variables.
    """
    model_name = model_class.__name__

    # Initialize and fit the model
    model = model_class()

    # QuantReg needs quantiles at fit time
    if model_name == "QuantReg":
        fitted_model = model.fit(
            X_train=train_data,
            predictors=predictors,
            imputed_variables=imputed_variables,
            weight_col=weight_col,
            quantiles=quantiles,
        )
    else:
        fitted_model = model.fit(
            X_train=train_data,
            predictors=predictors,
            imputed_variables=imputed_variables,
            weight_col=weight_col,
        )

    # Get predictions
    predictions = fitted_model.predict(test_data, quantiles)

    # Compute quantile loss using the existing function from metrics
    losses = []
    for q in quantiles:
        if q not in predictions:
            continue
        for var in imputed_variables:
            if var not in predictions[q].columns:
                continue

            true_values = test_data[var].values
            pred_values = predictions[q][var].values

            # Use existing quantile_loss function
            loss_array = quantile_loss(q, true_values, pred_values)
            losses.append(np.mean(loss_array))

    return np.mean(losses) if losses else np.nan


def plot_cia_degradation_curves(
    results: Dict[str, Dict[str, Any]],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (PLOT_CONFIG["width"], PLOT_CONFIG["height"]),
    use_absolute_loss: bool = True,
) -> go.Figure:
    """Plot CIA degradation curves for multiple methods.

    Args:
        results: Dict mapping method name to progressive_predictor_exclusion results.
        title: Plot title.
        save_path: Path to save the figure.
        figsize: Figure size as (width, height).
        use_absolute_loss: If True (default), plot absolute quantile loss values.
            If False, plot normalized loss (relative to each method's baseline).

    Returns:
        Plotly figure object.
    """
    fig = go.Figure()

    y_col = "quantile_loss" if use_absolute_loss else "normalized_loss"

    for method_name, method_results in results.items():
        df = method_results["results_df"]

        # Skip if results_df is empty or missing required column
        if df.empty or y_col not in df.columns:
            continue

        valid_df = df[df[y_col].notna()]

        if valid_df.empty:
            continue

        color = METHOD_COLORS.get(method_name, "#999999")
        baseline = method_results.get("baseline_loss", np.nan)

        # Build label with baseline info
        label = f"{method_name}"
        if not np.isnan(baseline):
            label += f" (baseline={baseline:.4f})"

        fig.add_trace(
            go.Scatter(
                x=valid_df["step"],
                y=valid_df[y_col],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                marker=dict(color=color, size=8),
            )
        )

    if title is None:
        title = "CIA Sensitivity: Performance Degradation as Predictors are Removed"

    y_axis_title = (
        "Quantile Loss"
        if use_absolute_loss
        else "Normalized Quantile Loss (1.0 = baseline)"
    )

    fig.update_layout(
        title=title,
        xaxis_title="Number of Predictors Removed",
        yaxis_title=y_axis_title,
        height=figsize[1],
        width=figsize[0],
        paper_bgcolor=PLOT_CONFIG["paper_bgcolor"],
        plot_bgcolor=PLOT_CONFIG["plot_bgcolor"],
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
        hovermode="x unified",
    )

    fig.update_xaxes(
        showgrid=PLOT_CONFIG["showgrid_x"],
        gridcolor=PLOT_CONFIG["gridcolor"],
        showline=PLOT_CONFIG["showline"],
        linecolor=PLOT_CONFIG["linecolor"],
    )
    fig.update_yaxes(
        showgrid=PLOT_CONFIG["showgrid_y"],
        gridcolor=PLOT_CONFIG["gridcolor"],
        showline=PLOT_CONFIG["showline"],
        linecolor=PLOT_CONFIG["linecolor"],
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def create_benchmark_summary_table(
    cv_results: Dict[str, Dict[str, Dict[str, Any]]],
    cia_results: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
) -> pd.DataFrame:
    """Create a summary table of benchmark results with rankings.

    Args:
        cv_results: Dict mapping dataset name to cv_results from autoimpute.
        cia_results: Optional dict mapping dataset name to CIA sensitivity results.

    Returns:
        DataFrame with columns: Dataset, Best Method, and rank columns for each method.
    """
    methods = ["QRF", "OLS", "QuantReg", "Matching", "MDN"]
    rows = []

    for dataset_name, dataset_cv_results in cv_results.items():
        row = {"Dataset": dataset_name}

        # Extract mean quantile loss for each method
        losses = {}
        for method in methods:
            if method in dataset_cv_results:
                method_data = dataset_cv_results[method]
                if "quantile_loss" in method_data:
                    ql = method_data["quantile_loss"]
                    losses[method] = ql.get("mean_test", np.nan)
                else:
                    losses[method] = np.nan
            else:
                losses[method] = np.nan

        # Compute ranks
        valid_losses = {k: v for k, v in losses.items() if not np.isnan(v)}
        if valid_losses:
            sorted_methods = sorted(
                valid_losses.keys(), key=lambda x: valid_losses[x]
            )
            ranks = {m: i + 1 for i, m in enumerate(sorted_methods)}

            # Fill in ranks for methods with NaN losses
            max_rank = len(valid_losses) + 1
            for method in methods:
                if method not in ranks:
                    ranks[method] = max_rank

            row["Best Method"] = sorted_methods[0]
        else:
            ranks = {m: np.nan for m in methods}
            row["Best Method"] = "N/A"

        # Add rank columns
        for method in methods:
            row[f"{method} Rank"] = ranks.get(method, np.nan)
            row[f"{method} Loss"] = losses.get(method, np.nan)

        # Add CIA sensitivity if provided
        if cia_results and dataset_name in cia_results:
            for method in methods:
                if method in cia_results[dataset_name]:
                    row[f"{method} CIA"] = cia_results[dataset_name][
                        method
                    ].get("sensitivity_score", np.nan)
                else:
                    row[f"{method} CIA"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)

    # Add mean rank row
    mean_row = {"Dataset": "Mean Rank", "Best Method": "-"}
    for method in methods:
        rank_col = f"{method} Rank"
        if rank_col in df.columns:
            mean_row[rank_col] = df[rank_col].mean()
        loss_col = f"{method} Loss"
        if loss_col in df.columns:
            mean_row[loss_col] = df[loss_col].mean()
        cia_col = f"{method} CIA"
        if cia_col in df.columns:
            mean_row[cia_col] = df[cia_col].mean()

    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    return df


def create_benchmark_heatmap(
    cv_results: Dict[str, Dict[str, Dict[str, Any]]],
    wasserstein_results: Optional[Dict[str, Dict[str, float]]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Optional[tuple] = None,
) -> go.Figure:
    """Create a heatmap showing quantile loss and Wasserstein distance.

    The heatmap uses:
    - Method-specific colors (consistent per column)
    - Row-type opacity: Q-Loss rows are darker (0.8), W-Dist rows lighter (0.5)
    - Font weight/size to indicate performance: best = bold + larger

    Args:
        cv_results: Dict mapping dataset name to cv_results from autoimpute.
        wasserstein_results: Optional dict mapping dataset to {method: distance}.
        title: Plot title.
        save_path: Path to save the figure.
        figsize: Figure size (width, height). Auto-calculated if None.

    Returns:
        Plotly figure object.
    """
    methods = ["QRF", "OLS", "QuantReg", "Matching", "MDN"]
    datasets = list(cv_results.keys())

    # Build data matrix with row type tracking
    row_labels = []
    row_types = []  # 'qloss' or 'wdist'
    data_matrix = []

    for dataset in datasets:
        dataset_cv = cv_results[dataset]

        # Quantile Loss row
        ql_row = []
        for method in methods:
            if method in dataset_cv:
                val = dataset_cv[method]
                # Handle both nested format and simple float format
                if isinstance(val, dict) and "quantile_loss" in val:
                    # Nested format: cv_results[dataset][method]["quantile_loss"]["mean_test"]
                    ql_row.append(
                        val["quantile_loss"].get("mean_test", np.nan)
                    )
                elif isinstance(val, (int, float)):
                    # Simple format: cv_results[dataset][method] = float
                    ql_row.append(val)
                else:
                    ql_row.append(np.nan)
            else:
                ql_row.append(np.nan)
        row_labels.append(f"{dataset} (Q-Loss)")
        row_types.append("qloss")
        data_matrix.append(ql_row)

        # Wasserstein Distance row
        if wasserstein_results and dataset in wasserstein_results:
            wd_row = []
            for method in methods:
                wd_row.append(wasserstein_results[dataset].get(method, np.nan))
            row_labels.append(f"{dataset} (W-Dist)")
            row_types.append("wdist")
            data_matrix.append(wd_row)

    data_array = np.array(data_matrix)
    n_rows = len(row_labels)
    n_cols = len(methods)

    # Compute ranks for each row (1 = best, higher = worse)
    rank_matrix = np.zeros_like(data_array)
    for i in range(len(data_array)):
        row = data_array[i]
        valid_mask = ~np.isnan(row)
        if valid_mask.sum() > 0:
            # Rank valid values (lower value = better = rank 1)
            valid_vals = row[valid_mask]
            ranks = np.argsort(np.argsort(valid_vals)) + 1
            rank_matrix[i, valid_mask] = ranks
            # Set NaN positions to max rank + 1
            rank_matrix[i, ~valid_mask] = valid_mask.sum() + 1

    # Create figure
    fig = go.Figure()

    # Create cells using shapes (rectangles)
    annotations = []

    for i in range(n_rows):
        row_type = row_types[i]
        # Set alpha based on row type: Q-Loss darker, W-Dist lighter
        alpha = 0.8 if row_type == "qloss" else 0.5

        for j, method in enumerate(methods):
            value = data_array[i, j]
            rank = rank_matrix[i, j]

            base_color = METHOD_COLORS.get(method, "#999999")
            r = int(base_color[1:3], 16)
            g = int(base_color[3:5], 16)
            b = int(base_color[5:7], 16)

            if np.isnan(value):
                cell_color = "rgba(200, 200, 200, 0.3)"
                text = "N/A"
                font_size = 10
                font_weight = "normal"
            else:
                cell_color = f"rgba({r}, {g}, {b}, {alpha:.2f})"
                text = f"{value:.4f}"
                # Best performer (rank 1) gets bold + larger font
                if rank == 1:
                    font_size = 12
                    font_weight = "bold"
                else:
                    font_size = 10
                    font_weight = "normal"

            # Add rectangle shape for cell background
            fig.add_shape(
                type="rect",
                x0=j - 0.5,
                x1=j + 0.5,
                y0=i - 0.5,
                y1=i + 0.5,
                fillcolor=cell_color,
                line=dict(color="white", width=1),
            )

            # Text color: white for MDN (dark indigo), black for others
            if method == "MDN" and not np.isnan(value):
                text_color = "white"
            else:
                text_color = "black"

            # Add text annotation with bold formatting for best
            if font_weight == "bold":
                text = f"<b>{text}</b>"

            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=text,
                    showarrow=False,
                    font=dict(color=text_color, size=font_size),
                    xanchor="center",
                    yanchor="middle",
                )
            )

    # Add method color legend
    for method in methods:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=12, color=METHOD_COLORS.get(method, "#999999")
                ),
                name=method,
            )
        )

    if title is None:
        title = "Cross-Dataset Benchmark Results"

    # Calculate compact figure size if not provided
    if figsize is None:
        width = max(500, n_cols * 100 + 200)
        height = max(300, n_rows * 40 + 100)
        figsize = (width, height)

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(n_cols)),
            ticktext=methods,
            side="top",
            showgrid=False,
            zeroline=False,
            range=[-0.5, n_cols - 0.5],
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(n_rows)),
            ticktext=row_labels,
            showgrid=False,
            zeroline=False,
            autorange="reversed",
            range=[-0.5, n_rows - 0.5],
        ),
        height=figsize[1],
        width=figsize[0],
        paper_bgcolor=PLOT_CONFIG["paper_bgcolor"],
        plot_bgcolor=PLOT_CONFIG["plot_bgcolor"],
        annotations=annotations,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=180, r=20, t=60, b=40),
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def run_cia_analysis_for_dataset(
    data: pd.DataFrame,
    predictors: List[str],
    imputed_variables: List[str],
    model_classes: Optional[List[Type[Imputer]]] = None,
    ordering: str = "mutual_info",
    train_size: float = TRAIN_SIZE,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Dict[str, Any]]:
    """Run CIA sensitivity analysis for all models on a dataset.

    Args:
        data: DataFrame containing the data.
        predictors: List of predictor column names.
        imputed_variables: List of variables to impute.
        model_classes: List of Imputer classes to evaluate.
        ordering: Predictor ordering method.
        train_size: Proportion for training.
        random_state: Random state.

    Returns:
        Dict mapping method name to progressive_predictor_exclusion results.
    """
    if model_classes is None:
        from microimpute.models import OLS, QRF, QuantReg

        model_classes = [QRF, OLS, QuantReg]
        try:
            from microimpute.models import Matching

            model_classes.append(Matching)
        except ImportError:
            pass
        try:
            from microimpute.models import MDN

            model_classes.append(MDN)
        except ImportError:
            pass

    results = {}

    for model_class in model_classes:
        method_name = model_class.__name__

        try:
            method_results = progressive_predictor_exclusion(
                data=data,
                predictors=predictors,
                imputed_variables=imputed_variables,
                model_class=model_class,
                ordering=ordering,
                train_size=train_size,
                random_state=random_state,
            )
            results[method_name] = method_results
            print(
                f"  {method_name} sensitivity score: "
                f"{method_results['sensitivity_score']:.3f}"
            )
        except Exception as e:
            log.warning(f"CIA analysis failed for {method_name}: {e}")
            print(f"  CIA analysis failed for {method_name}: {e}")
            results[method_name] = {
                "results_df": pd.DataFrame(),
                "predictor_order": [],
                "baseline_loss": np.nan,
                "sensitivity_score": np.nan,
            }

    return results
