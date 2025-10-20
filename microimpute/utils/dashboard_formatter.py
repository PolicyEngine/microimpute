"""
Utility function to format various imputation outputs into a unified CSV format for dashboard visualization.
"""

import json
from typing import Any, Dict, List, Optional, Union

import pandas as pd


def format_csv(
    output_path: Optional[str] = None,
    autoimpute_result: Optional[Dict] = None,
    comparison_metrics_df: Optional[pd.DataFrame] = None,
    distribution_comparison_df: Optional[pd.DataFrame] = None,
    predictor_correlations: Optional[Dict[str, pd.DataFrame]] = None,
    predictor_importance_df: Optional[pd.DataFrame] = None,
    progressive_inclusion_df: Optional[pd.DataFrame] = None,
    best_method_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Format various imputation outputs into a unified long-format CSV for dashboard visualization.

    Parameters
    ----------
    output_path : str
        Path to save the formatted CSV file.

    autoimpute_result : Dict, optional
        Result from autoimpute containing cv_results with benchmark losses.
        Expected structure: {'cv_results': {method: {'quantile_loss': {...}, 'log_loss': {...}}}}

    comparison_metrics_df : pd.DataFrame, optional
        DataFrame from compare_metrics() with columns:
        ['Method', 'Imputed Variable', 'Percentile', 'Loss', 'Metric']

    distribution_comparison_df : pd.DataFrame, optional
        DataFrame from compare_distributions() with columns:
        ['Variable', 'Metric', 'Distance']

    predictor_correlations : Dict[str, pd.DataFrame], optional
        Dict from compute_predictor_correlations() with keys like 'pearson', 'spearman', 'mutual_info'
        and correlation matrices as values. Also can include 'predictor_target_mi'.

    predictor_importance_df : pd.DataFrame, optional
        DataFrame from leave_one_out_analysis() with columns:
        ['predictor_removed', 'avg_quantile_loss', 'avg_log_loss', 'loss_increase', 'relative_impact']

    progressive_inclusion_df : pd.DataFrame, optional
        DataFrame from progressive_predictor_inclusion()['results_df'] with columns:
        ['step', 'predictor_added', 'predictors_included', 'avg_quantile_loss',
         'avg_log_loss', 'cumulative_improvement', 'marginal_improvement']

    best_method_name : str, optional
        Name of the best method to append "_best_method" suffix to.

    Returns
    -------
    pd.DataFrame
        Unified long-format DataFrame with columns:
        ['type', 'method', 'variable', 'quantile', 'metric_name', 'metric_value', 'split', 'additional_info']
    """
    rows = []

    # 1. Process autoimpute benchmark losses from cv_results
    if autoimpute_result and "cv_results" in autoimpute_result:
        for method, cv_result in autoimpute_result["cv_results"].items():
            # Append "_best_method" if this is the best method
            method_label = (
                f"{method}_best_method"
                if method == best_method_name
                else method
            )

            for metric_type in ["quantile_loss", "log_loss"]:
                if metric_type in cv_result:
                    data = cv_result[metric_type]
                    results_df = data.get("results")
                    variables = data.get("variables", [])

                    if results_df is not None:
                        # Add individual quantile results
                        for split in ["train", "test"]:
                            if split in results_df.index:
                                for quantile in results_df.columns:
                                    # Add mean_all row for this quantile
                                    rows.append(
                                        {
                                            "type": "benchmark_loss",
                                            "method": method_label,
                                            "variable": f"{metric_type}_mean_all",
                                            "quantile": float(quantile),
                                            "metric_name": metric_type,
                                            "metric_value": results_df.loc[
                                                split, quantile
                                            ],
                                            "split": split,
                                            "additional_info": json.dumps(
                                                {"n_variables": len(variables)}
                                            ),
                                        }
                                    )

                        # Add mean across all quantiles
                        if "mean_train" in data:
                            rows.append(
                                {
                                    "type": "benchmark_loss",
                                    "method": method_label,
                                    "variable": f"{metric_type}_mean_all",
                                    "quantile": "mean",
                                    "metric_name": metric_type,
                                    "metric_value": data["mean_train"],
                                    "split": "train",
                                    "additional_info": json.dumps(
                                        {"n_variables": len(variables)}
                                    ),
                                }
                            )

                        if "mean_test" in data:
                            rows.append(
                                {
                                    "type": "benchmark_loss",
                                    "method": method_label,
                                    "variable": f"{metric_type}_mean_all",
                                    "quantile": "mean",
                                    "metric_name": metric_type,
                                    "metric_value": data["mean_test"],
                                    "split": "test",
                                    "additional_info": json.dumps(
                                        {"n_variables": len(variables)}
                                    ),
                                }
                            )

    # 2. Process comparison metrics (per-variable benchmark losses)
    if comparison_metrics_df is not None and not comparison_metrics_df.empty:
        for _, row in comparison_metrics_df.iterrows():
            method = row["Method"]
            method_label = (
                f"{method}_best_method"
                if method == best_method_name
                else method
            )

            # Handle variable naming - check if it's an aggregate
            variable = row["Imputed Variable"]
            if "mean_quantile_loss" in variable:
                variable = "quantile_loss_mean_all"
            elif "mean_log_loss" in variable:
                variable = "log_loss_mean_all"

            # Handle percentile - can be a number or "mean_loss"
            percentile = row["Percentile"]
            if percentile == "mean_loss":
                quantile = "mean"
            else:
                quantile = float(percentile) if pd.notna(percentile) else "N/A"

            rows.append(
                {
                    "type": "benchmark_loss",
                    "method": method_label,
                    "variable": variable,
                    "quantile": quantile,
                    "metric_name": row["Metric"],
                    "metric_value": row["Loss"],
                    "split": "test",  # Comparison metrics are typically on test set
                    "additional_info": "{}",
                }
            )

    # 3. Process distribution comparison metrics
    if (
        distribution_comparison_df is not None
        and not distribution_comparison_df.empty
    ):
        for _, row in distribution_comparison_df.iterrows():
            rows.append(
                {
                    "type": "distribution_distance",
                    "method": best_method_name if best_method_name else "N/A",
                    "variable": row["Variable"],
                    "quantile": "N/A",
                    "metric_name": row["Metric"]
                    .lower()
                    .replace(" ", "_"),  # e.g., "wasserstein_distance"
                    "metric_value": row["Distance"],
                    "split": "full",
                    "additional_info": "{}",
                }
            )

    # 4. Process predictor correlations
    if predictor_correlations:
        # Handle correlation matrices (pearson, spearman, mutual_info between predictors)
        for corr_type in ["pearson", "spearman", "mutual_info"]:
            if corr_type in predictor_correlations:
                corr_matrix = predictor_correlations[corr_type]
                # Extract upper triangle (excluding diagonal)
                for i in range(len(corr_matrix.index)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        pred1 = corr_matrix.index[i]
                        pred2 = corr_matrix.columns[j]
                        rows.append(
                            {
                                "type": "predictor_correlation",
                                "method": "N/A",
                                "variable": pred1,
                                "quantile": "N/A",
                                "metric_name": corr_type,
                                "metric_value": corr_matrix.iloc[i, j],
                                "split": "full",
                                "additional_info": json.dumps(
                                    {"predictor2": pred2}
                                ),
                            }
                        )

        # Handle predictor-target MI
        if "predictor_target_mi" in predictor_correlations:
            mi_df = predictor_correlations["predictor_target_mi"]
            for predictor in mi_df.index:
                for target in mi_df.columns:
                    rows.append(
                        {
                            "type": "predictor_target_mi",
                            "method": "N/A",
                            "variable": predictor,
                            "quantile": "N/A",
                            "metric_name": "mutual_info",
                            "metric_value": mi_df.loc[predictor, target],
                            "split": "full",
                            "additional_info": json.dumps({"target": target}),
                        }
                    )

    # 5. Process predictor importance (leave-one-out)
    if (
        predictor_importance_df is not None
        and not predictor_importance_df.empty
    ):
        for _, row in predictor_importance_df.iterrows():
            predictor = row["predictor_removed"]

            # Add relative impact
            rows.append(
                {
                    "type": "predictor_importance",
                    "method": best_method_name if best_method_name else "N/A",
                    "variable": predictor,
                    "quantile": "N/A",
                    "metric_name": "relative_impact",
                    "metric_value": row["relative_impact"],
                    "split": "test",
                    "additional_info": json.dumps(
                        {"removed_predictor": predictor}
                    ),
                }
            )

            # Add loss increase
            rows.append(
                {
                    "type": "predictor_importance",
                    "method": best_method_name if best_method_name else "N/A",
                    "variable": predictor,
                    "quantile": "N/A",
                    "metric_name": "loss_increase",
                    "metric_value": row["loss_increase"],
                    "split": "test",
                    "additional_info": json.dumps(
                        {"removed_predictor": predictor}
                    ),
                }
            )

    # 6. Process progressive predictor inclusion
    if (
        progressive_inclusion_df is not None
        and not progressive_inclusion_df.empty
    ):
        for _, row in progressive_inclusion_df.iterrows():
            step = row["step"]
            predictor_added = row["predictor_added"]
            predictors_included = row["predictors_included"]

            # Add cumulative improvement
            if "cumulative_improvement" in row and pd.notna(
                row["cumulative_improvement"]
            ):
                rows.append(
                    {
                        "type": "progressive_inclusion",
                        "method": (
                            best_method_name if best_method_name else "N/A"
                        ),
                        "variable": "N/A",
                        "quantile": "N/A",
                        "metric_name": "cumulative_improvement",
                        "metric_value": row["cumulative_improvement"],
                        "split": "test",
                        "additional_info": json.dumps(
                            {
                                "step": int(step),
                                "predictor_added": predictor_added,
                                "predictors": (
                                    predictors_included
                                    if isinstance(predictors_included, list)
                                    else [predictors_included]
                                ),
                            }
                        ),
                    }
                )

            # Add marginal improvement
            if "marginal_improvement" in row and pd.notna(
                row["marginal_improvement"]
            ):
                rows.append(
                    {
                        "type": "progressive_inclusion",
                        "method": (
                            best_method_name if best_method_name else "N/A"
                        ),
                        "variable": "N/A",
                        "quantile": "N/A",
                        "metric_name": "marginal_improvement",
                        "metric_value": row["marginal_improvement"],
                        "split": "test",
                        "additional_info": json.dumps(
                            {
                                "step": int(step),
                                "predictor_added": predictor_added,
                            }
                        ),
                    }
                )

    # Create DataFrame from rows
    if not rows:
        # Return empty DataFrame with correct columns if no data
        return pd.DataFrame(
            columns=[
                "type",
                "method",
                "variable",
                "quantile",
                "metric_name",
                "metric_value",
                "split",
                "additional_info",
            ]
        )

    df = pd.DataFrame(rows)

    # Ensure correct data types
    df["metric_value"] = pd.to_numeric(df["metric_value"], errors="coerce")

    # Convert quantile to numeric where possible (keep 'mean' and 'N/A' as strings)
    def convert_quantile(q):
        if isinstance(q, (int, float)):
            return float(q)
        elif q in ["mean", "N/A"]:
            return q
        else:
            try:
                return float(q)
            except:
                return q

    df["quantile"] = df["quantile"].apply(convert_quantile)

    if output_path:
        df.to_csv(output_path, index=False)

    return df
