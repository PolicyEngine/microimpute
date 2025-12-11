"""
Utility function to format various imputation outputs into a unified CSV format for dashboard visualization.
"""

import json
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


def _compute_histogram_data(
    donor_values: np.ndarray,
    receiver_values: np.ndarray,
    variable_name: str,
    n_bins: int = 30,
) -> Dict[str, Union[List[float], str, int]]:
    """
    Compute histogram bin data for numerical variables.

    Parameters
    ----------
    donor_values : np.ndarray
        Original donor dataset values
    receiver_values : np.ndarray
        Imputed receiver dataset values
    variable_name : str
        Name of the variable being analyzed
    n_bins : int
        Number of histogram bins (default: 30)

    Returns
    -------
    Dict containing bin edges and heights for both distributions
    """
    # Remove NaN values
    donor_clean = donor_values[~np.isnan(donor_values)]
    receiver_clean = receiver_values[~np.isnan(receiver_values)]

    # Determine bin edges based on combined data range using numpy's auto algorithm
    combined = np.concatenate([donor_clean, receiver_clean])
    _, bin_edges = np.histogram(combined, bins=n_bins)

    # Compute histogram heights (normalized as densities)
    donor_heights, _ = np.histogram(donor_clean, bins=bin_edges, density=True)
    receiver_heights, _ = np.histogram(
        receiver_clean, bins=bin_edges, density=True
    )

    # Convert to percentages for easier interpretation
    # Multiply by bin width to get probability mass per bin
    bin_widths = np.diff(bin_edges)
    donor_heights = (donor_heights * bin_widths * 100).tolist()
    receiver_heights = (receiver_heights * bin_widths * 100).tolist()

    return {
        "variable": variable_name,
        "bin_edges": bin_edges.tolist(),
        "donor_heights": donor_heights,
        "receiver_heights": receiver_heights,
        "n_samples_donor": len(donor_clean),
        "n_samples_receiver": len(receiver_clean),
        "n_bins": n_bins,
    }


def _compute_categorical_distribution(
    donor_values: pd.Series,
    receiver_values: pd.Series,
    variable_name: str,
    max_categories: int = 20,
) -> Dict[str, Union[List, str, bool]]:
    """
    Compute distribution data for categorical variables.

    Parameters
    ----------
    donor_values : pd.Series
        Original donor dataset values
    receiver_values : pd.Series
        Imputed receiver dataset values
    variable_name : str
        Name of the variable
    max_categories : int
        Maximum number of categories to include (others grouped as "Other")

    Returns
    -------
    Dict containing category labels and proportions
    """
    # Get value counts
    donor_counts = donor_values.value_counts()
    receiver_counts = receiver_values.value_counts()

    # Get all unique categories
    all_categories = list(set(donor_counts.index) | set(receiver_counts.index))

    # If too many categories, keep top ones and group rest as "Other"
    if len(all_categories) > max_categories:
        # Get top categories by combined frequency
        combined_counts = donor_counts.add(receiver_counts, fill_value=0)
        top_categories = combined_counts.nlargest(
            max_categories - 1
        ).index.tolist()

        # Calculate "Other" category
        donor_other = donor_counts[
            ~donor_counts.index.isin(top_categories)
        ].sum()
        receiver_other = receiver_counts[
            ~receiver_counts.index.isin(top_categories)
        ].sum()

        categories = top_categories + ["Other"]

        # Get proportions
        donor_props = [donor_counts.get(cat, 0) for cat in top_categories]
        donor_props.append(donor_other)
        donor_props = (
            pd.Series(donor_props) / donor_values.count() * 100
        ).tolist()

        receiver_props = [
            receiver_counts.get(cat, 0) for cat in top_categories
        ]
        receiver_props.append(receiver_other)
        receiver_props = (
            pd.Series(receiver_props) / receiver_values.count() * 100
        ).tolist()
    else:
        categories = sorted(all_categories)
        donor_props = [
            (donor_counts.get(cat, 0) / donor_values.count() * 100)
            for cat in categories
        ]
        receiver_props = [
            (receiver_counts.get(cat, 0) / receiver_values.count() * 100)
            for cat in categories
        ]

    return {
        "variable": variable_name,
        "categories": categories,
        "donor_proportions": donor_props,
        "receiver_proportions": receiver_props,
        "n_samples_donor": int(donor_values.count()),
        "n_samples_receiver": int(receiver_values.count()),
        "is_categorical": True,
    }


def _format_histogram_rows(
    histogram_data: Dict[str, Union[List, str, int, bool]], method: str
) -> List[Dict]:
    """
    Convert histogram data to CSV row format.

    Parameters
    ----------
    histogram_data : Dict
        Output from _compute_histogram_data or _compute_categorical_distribution
    method : str
        Imputation method name

    Returns
    -------
    List of dictionaries ready for CSV formatting
    """
    rows = []

    if histogram_data.get("is_categorical", False):
        # Categorical variable - store as distribution_bins type
        for i, category in enumerate(histogram_data["categories"]):
            rows.append(
                {
                    "type": "distribution_bins",
                    "method": method,
                    "variable": histogram_data["variable"],
                    "quantile": "N/A",
                    "metric_name": "categorical_distribution",
                    "metric_value": None,  # Not used for histograms
                    "split": "full",
                    "additional_info": json.dumps(
                        {
                            "category": str(category),
                            "donor_proportion": float(
                                histogram_data["donor_proportions"][i]
                            ),
                            "receiver_proportion": float(
                                histogram_data["receiver_proportions"][i]
                            ),
                            "n_samples_donor": int(
                                histogram_data["n_samples_donor"]
                            ),
                            "n_samples_receiver": int(
                                histogram_data["n_samples_receiver"]
                            ),
                        }
                    ),
                }
            )
    else:
        # Numerical variable - store bin data
        n_bins = len(histogram_data["donor_heights"])
        for i in range(n_bins):
            rows.append(
                {
                    "type": "distribution_bins",
                    "method": method,
                    "variable": histogram_data["variable"],
                    "quantile": "N/A",
                    "metric_name": "histogram_distribution",
                    "metric_value": None,  # Not used for histograms
                    "split": "full",
                    "additional_info": json.dumps(
                        {
                            "bin_index": int(i),
                            "bin_start": float(histogram_data["bin_edges"][i]),
                            "bin_end": float(
                                histogram_data["bin_edges"][i + 1]
                            ),
                            "donor_height": float(
                                histogram_data["donor_heights"][i]
                            ),
                            "receiver_height": float(
                                histogram_data["receiver_heights"][i]
                            ),
                            "n_samples_donor": int(
                                histogram_data["n_samples_donor"]
                            ),
                            "n_samples_receiver": int(
                                histogram_data["n_samples_receiver"]
                            ),
                            "total_bins": int(n_bins),
                        }
                    ),
                }
            )

    return rows


def _validate_imputed_variables(
    donor_data: pd.DataFrame,
    receiver_data: pd.DataFrame,
    imputed_variables: List[str],
) -> None:
    """
    Validate that all imputed variables exist in both datasets.

    Parameters
    ----------
    donor_data : pd.DataFrame
        Original donor dataset
    receiver_data : pd.DataFrame
        Imputed receiver dataset
    imputed_variables : List[str]
        List of variable names that were imputed

    Raises
    ------
    ValueError
        If any imputed variable is missing from either dataset
    """
    missing_in_donor = [
        var for var in imputed_variables if var not in donor_data.columns
    ]
    missing_in_receiver = [
        var for var in imputed_variables if var not in receiver_data.columns
    ]

    if missing_in_donor:
        raise ValueError(
            f"The following imputed variables are missing from donor_data: {missing_in_donor}"
        )

    if missing_in_receiver:
        raise ValueError(
            f"The following imputed variables are missing from receiver_data: {missing_in_receiver}"
        )


def format_csv(
    output_path: Optional[str] = None,
    autoimpute_result: Optional[Dict] = None,
    comparison_metrics_df: Optional[pd.DataFrame] = None,
    distribution_comparison_df: Optional[pd.DataFrame] = None,
    predictor_correlations: Optional[Dict[str, pd.DataFrame]] = None,
    predictor_importance_df: Optional[pd.DataFrame] = None,
    progressive_inclusion_df: Optional[pd.DataFrame] = None,
    best_method_name: Optional[str] = None,
    donor_data: Optional[pd.DataFrame] = None,
    receiver_data: Optional[pd.DataFrame] = None,
    imputed_variables: Optional[List[str]] = None,
    n_histogram_bins: int = 30,
) -> pd.DataFrame:
    """
    Format various imputation outputs into a unified long-format CSV for dashboard visualization.

    Parameters
    ----------
    output_path : str
        Path to save the formatted CSV file.

    autoimpute_result : Dict, optional
        Result from autoimpute containing cv_results with benchmark losses.
        Expected structure: {method: {'quantile_loss': {...}, 'log_loss': {...}}}

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

    donor_data : pd.DataFrame, optional
        Original donor dataset for histogram generation. Required if imputed_variables is provided.

    receiver_data : pd.DataFrame, optional
        Imputed receiver dataset for histogram generation. Required if imputed_variables is provided.

    imputed_variables : List[str], optional
        List of variable names that were imputed. When provided with donor_data and receiver_data,
        histogram bin data will be included in the CSV for distribution visualization.

    n_histogram_bins : int, default 30
        Number of bins to use for numerical variable histograms.

    Returns
    -------
    pd.DataFrame
        Unified long-format DataFrame with columns:
        ['type', 'method', 'variable', 'quantile', 'metric_name', 'metric_value', 'split', 'additional_info']

    Raises
    ------
    ValueError
        If imputed_variables is provided but donor_data or receiver_data is missing.
        If any imputed variable is not present in both donor_data and receiver_data.
    """
    rows = []

    # 1. Process autoimpute benchmark losses from cv_results
    if autoimpute_result and isinstance(autoimpute_result, dict):
        first_value = next(iter(autoimpute_result.values()), None)
        if isinstance(first_value, dict) and (
            "quantile_loss" in first_value or "log_loss" in first_value
        ):
            for method, cv_result in autoimpute_result.items():
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
                                                    {
                                                        "n_variables": len(
                                                            variables
                                                        )
                                                    }
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

    # 7. Process histogram distribution data for imputed variables
    if imputed_variables is not None:
        # Validate inputs
        if donor_data is None or receiver_data is None:
            raise ValueError(
                "donor_data and receiver_data are required when imputed_variables is provided"
            )

        # Validate that all imputed variables exist in both datasets
        _validate_imputed_variables(
            donor_data, receiver_data, imputed_variables
        )

        # Generate histogram data for each imputed variable
        for var in imputed_variables:
            # Check if variable is categorical or numerical
            if donor_data[
                var
            ].dtype == "object" or pd.api.types.is_categorical_dtype(
                donor_data[var]
            ):
                # Categorical variable
                hist_data = _compute_categorical_distribution(
                    donor_data[var], receiver_data[var], var
                )
            else:
                # Numerical variable
                hist_data = _compute_histogram_data(
                    donor_data[var].values,
                    receiver_data[var].values,
                    var,
                    n_bins=n_histogram_bins,
                )

            # Format histogram rows and add to main rows list
            histogram_rows = _format_histogram_rows(
                hist_data, best_method_name if best_method_name else "N/A"
            )
            rows.extend(histogram_rows)

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
