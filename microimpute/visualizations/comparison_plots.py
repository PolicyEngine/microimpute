"""Multi-method comparison visualization with dual metric support

This module provides comprehensive visualization tools for comparing the performance
of multiple imputation methods. It supports both quantile loss and log loss metrics,
creating appropriate visualizations for each type.

Key components:
    - MethodComparisonResults: container class for comparison data with plotting methods
    - method_comparison_results: factory function to create comparison visualizations
    - Support for quantile loss, log loss, and combined metric comparisons
    - Stacked bar plots showing contribution to total loss
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from microimpute.config import PLOT_CONFIG
from microimpute.visualizations.performance_plots import _save_figure

logger = logging.getLogger(__name__)


class MethodComparisonResults:
    """Class to store and visualize comparison results across methods with dual metric support."""

    def __init__(
        self,
        comparison_data: Union[pd.DataFrame, Dict[str, Dict[str, Dict]]],
        metric: str = "quantile_loss",
        imputed_variables: Optional[List[str]] = None,
        data_format: str = "wide",
    ):
        """Initialize MethodComparisonResults with comparison data.

        Args:
            comparison_data: Either:
                - DataFrame with comparison data (backward compat)
                - Dict with method names as keys, containing dual metric results
            metric: Which metric to visualize: 'quantile_loss', 'log_loss', or 'combined'
            imputed_variables: List of variable names that were imputed
            data_format: Input data format - 'wide', 'long', or 'dual_metrics'
        """
        self.metric = metric
        self.imputed_variables = imputed_variables or []
        self.data_format = data_format

        # Set display name based on metric
        if metric == "quantile_loss":
            self.metric_name = "Quantile loss"
        elif metric == "log_loss":
            self.metric_name = "Log loss"
        else:
            self.metric_name = "Loss"

        # Process data based on input format
        if isinstance(comparison_data, dict) and "quantile_loss" in next(
            iter(comparison_data.values()), {}
        ):
            # New dual metrics format
            self._process_dual_metrics_input(comparison_data)
        elif data_format == "wide":
            # Convert wide format to long format for internal use
            self._process_wide_input(comparison_data)
        else:
            # Data is already in long format
            self.comparison_data = comparison_data.copy()

            # Validate required columns for long format
            required_cols = [
                "Method",
                "Imputed Variable",
                "Percentile",
                "Loss",
            ]
            missing_cols = [
                col
                for col in required_cols
                if col not in self.comparison_data.columns
            ]
            if missing_cols:
                error_msg = f"Missing required columns: {missing_cols}"
                logger.error(error_msg)
                raise ValueError(error_msg)

        # Get unique methods and variables
        if hasattr(self, "comparison_data"):
            self.methods = self.comparison_data["Method"].unique().tolist()
            self.variables = (
                self.comparison_data["Imputed Variable"].unique().tolist()
            )
        else:
            # For dual metrics format
            self.methods = list(self.dual_metrics_data.keys())
            self.variables = []
            for method_data in self.dual_metrics_data.values():
                if "quantile_loss" in method_data:
                    self.variables.extend(
                        method_data["quantile_loss"].get("variables", [])
                    )
                if "log_loss" in method_data:
                    self.variables.extend(
                        method_data["log_loss"].get("variables", [])
                    )
            self.variables = list(set(self.variables))

        logger.debug(
            f"Initialized MethodComparisonResults with {len(self.methods)} methods "
            f"and {len(self.variables)} variables"
        )

    def _process_wide_input(self, wide_data: pd.DataFrame):
        """Convert wide format data to long format for internal use.

        Args:
            wide_data: DataFrame with methods as index and quantiles as columns
        """
        logger.debug("Converting wide format input to long format")

        # Reset index to get methods as a column
        data = wide_data.reset_index()
        if "index" in data.columns:
            data = data.rename(columns={"index": "Method"})

        # Convert to long format
        long_format_data = []

        for _, row in data.iterrows():
            method = row["Method"]

            for col in wide_data.columns:
                if col == "mean_loss":
                    # Add mean_loss as special case
                    long_format_data.append(
                        {
                            "Method": method,
                            "Imputed Variable": "mean_loss",
                            "Percentile": "mean_loss",
                            "Loss": row[col],
                        }
                    )
                else:
                    # Regular quantile columns
                    # Use first imputed variable if specified, otherwise "y"
                    var_name = (
                        self.imputed_variables[0]
                        if self.imputed_variables
                        else "y"
                    )
                    long_format_data.append(
                        {
                            "Method": method,
                            "Imputed Variable": var_name,
                            "Percentile": col,
                            "Loss": row[col],
                        }
                    )

        self.comparison_data = pd.DataFrame(long_format_data)

    def _process_dual_metrics_input(
        self, dual_data: Dict[str, Dict[str, Dict]]
    ):
        """Process dual metrics format from cross-validation results.

        Args:
            dual_data: Dict with method names as keys, containing 'quantile_loss' and 'log_loss' dicts
        """
        logger.debug("Processing dual metrics input")

        self.dual_metrics_data = dual_data

        # Convert to long format for compatibility
        long_format_data = []

        for method_name, method_results in dual_data.items():
            # Process quantile loss if available
            if (
                self.metric in ["quantile_loss", "combined"]
                and "quantile_loss" in method_results
            ):
                ql_data = method_results["quantile_loss"]
                if (
                    ql_data.get("results") is not None
                    and not ql_data["results"].empty
                ):
                    # Get test results (single row)
                    if "test" in ql_data["results"].index:
                        test_results = ql_data["results"].loc["test"]
                        for quantile in test_results.index:
                            for var in ql_data.get("variables", ["y"]):
                                long_format_data.append(
                                    {
                                        "Method": method_name,
                                        "Imputed Variable": var,
                                        "Percentile": quantile,
                                        "Loss": test_results[quantile],
                                        "Metric": "quantile_loss",
                                    }
                                )

                    # Add mean loss
                    if "mean_test" in ql_data:
                        for var in ql_data.get("variables", ["y"]):
                            long_format_data.append(
                                {
                                    "Method": method_name,
                                    "Imputed Variable": var,
                                    "Percentile": "mean_quantile_loss",
                                    "Loss": ql_data["mean_test"],
                                    "Metric": "quantile_loss",
                                }
                            )

            # Process log loss if available
            if (
                self.metric in ["log_loss", "combined"]
                and "log_loss" in method_results
            ):
                ll_data = method_results["log_loss"]
                if (
                    ll_data.get("results") is not None
                    and not ll_data["results"].empty
                ):
                    # Log loss is constant across quantiles
                    if "test" in ll_data["results"].index:
                        test_loss = ll_data["results"].loc["test"].mean()
                        for var in ll_data.get("variables", []):
                            long_format_data.append(
                                {
                                    "Method": method_name,
                                    "Imputed Variable": var,
                                    "Percentile": "log_loss",
                                    "Loss": test_loss,
                                    "Metric": "log_loss",
                                }
                            )

                    # Add mean loss
                    if "mean_test" in ll_data:
                        for var in ll_data.get("variables", []):
                            long_format_data.append(
                                {
                                    "Method": method_name,
                                    "Imputed Variable": var,
                                    "Percentile": "mean_log_loss",
                                    "Loss": ll_data["mean_test"],
                                    "Metric": "log_loss",
                                }
                            )

        self.comparison_data = pd.DataFrame(long_format_data)

    def plot(
        self,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_mean: bool = True,
        figsize: Tuple[int, int] = (
            PLOT_CONFIG["width"],
            PLOT_CONFIG["height"],
        ),
        plot_type: str = "bar",
    ) -> go.Figure:
        """Plot a comparison of performance across different imputation methods.

        Args:
            title: Custom title for the plot. If None, a default title is used.
            save_path: Path to save the plot. If None, the plot is displayed.
            show_mean: Whether to show horizontal lines for mean loss values.
            figsize: Figure size as (width, height) in pixels.
            plot_type: Type of plot: 'bar' (default) or 'stacked' (for contribution analysis)

        Returns:
            Plotly figure object

        Raises:
            ValueError: If data_subset is invalid or not available
            RuntimeError: If plot creation or saving fails
        """
        logger.debug(
            f"Creating method comparison plot with {len(self.methods)} methods"
        )

        if plot_type == "stacked":
            return self._plot_stacked_contribution(title, save_path, figsize)
        elif self.metric == "log_loss":
            return self._plot_log_loss_comparison(title, save_path, figsize)
        elif self.metric == "combined":
            return self._plot_combined_metrics(title, save_path, figsize)
        else:
            return self._plot_quantile_loss_comparison(
                title, save_path, show_mean, figsize
            )

    def _plot_quantile_loss_comparison(
        self,
        title: Optional[str],
        save_path: Optional[str],
        show_mean: bool,
        figsize: Tuple[int, int],
    ) -> go.Figure:
        """Plot quantile loss comparison across methods."""

        try:
            # Filter data for quantile loss only - include only numeric quantiles between 0 and 1
            if "Metric" in self.comparison_data.columns:
                # First filter for quantile_loss metric
                quantile_data = self.comparison_data[
                    self.comparison_data["Metric"] == "quantile_loss"
                ].copy()

                # Then filter for numeric quantiles only (between 0 and 1)
                numeric_mask = pd.to_numeric(
                    quantile_data["Percentile"], errors="coerce"
                ).notna()
                quantile_data = quantile_data[numeric_mask]

                # Convert to numeric and filter to valid quantile range
                quantile_data["Percentile"] = pd.to_numeric(
                    quantile_data["Percentile"]
                )
                melted_df = quantile_data[
                    (quantile_data["Percentile"] >= 0)
                    & (quantile_data["Percentile"] <= 1)
                ].copy()
            else:
                # Backward compatibility - filter for numeric quantiles
                melted_df = self.comparison_data.copy()

                # Filter for numeric percentiles only
                numeric_mask = pd.to_numeric(
                    melted_df["Percentile"], errors="coerce"
                ).notna()
                melted_df = melted_df[numeric_mask]

                # Convert to numeric and filter to valid quantile range
                melted_df["Percentile"] = pd.to_numeric(
                    melted_df["Percentile"]
                )
                melted_df = melted_df[
                    (melted_df["Percentile"] >= 0)
                    & (melted_df["Percentile"] <= 1)
                ].copy()

            melted_df = melted_df.rename(columns={"Loss": self.metric_name})
            melted_df["Percentile"] = melted_df["Percentile"].astype(str)

            if title is None:
                title = f"Test {self.metric_name} Across Quantiles for Different Imputation Methods"

            # Create the bar chart
            logger.debug("Creating bar chart with plotly express")
            fig = px.bar(
                melted_df,
                x="Percentile",
                y=self.metric_name,
                color="Method",
                color_discrete_sequence=px.colors.qualitative.Plotly,
                barmode="group",
                title=title,
                labels={
                    "Percentile": "Quantiles",
                    self.metric_name: f"Test {self.metric_name}",
                },
            )

            # Add horizontal lines for mean loss if requested
            if show_mean:
                logger.debug("Adding mean loss markers to plot")
                for i, method in enumerate(self.methods):
                    method_data = melted_df[melted_df["Method"] == method]
                    if not method_data.empty:
                        mean_loss = method_data[self.metric_name].mean()
                        n_percentiles = melted_df["Percentile"].nunique()
                        fig.add_shape(
                            type="line",
                            x0=-0.5,
                            y0=mean_loss,
                            x1=n_percentiles - 0.5,
                            y1=mean_loss,
                            line=dict(
                                color=px.colors.qualitative.Plotly[
                                    i % len(px.colors.qualitative.Plotly)
                                ],
                                width=2,
                                dash="dot",
                            ),
                            name=f"{method} Mean",
                        )

            fig.update_layout(
                title_font_size=14,
                xaxis_title_font_size=12,
                yaxis_title_font_size=12,
                paper_bgcolor="#F0F0F0",
                plot_bgcolor="#F0F0F0",
                legend_title="Method",
                height=figsize[1],
                width=figsize[0],
            )

            fig.update_xaxes(showgrid=False, zeroline=False)
            fig.update_yaxes(showgrid=False, zeroline=False)

            # Save or show the plot
            if save_path:
                _save_figure(fig, save_path)

            logger.debug("Plot creation completed successfully")
            return fig

        except Exception as e:
            logger.error(f"Error creating method comparison plot: {str(e)}")
            raise RuntimeError(
                f"Failed to create method comparison plot: {str(e)}"
            ) from e

    def _plot_log_loss_comparison(
        self,
        title: Optional[str],
        save_path: Optional[str],
        figsize: Tuple[int, int],
    ) -> go.Figure:
        """Plot log loss comparison across methods."""
        try:
            # Filter data for log loss only
            if "Metric" in self.comparison_data.columns:
                log_loss_df = self.comparison_data[
                    self.comparison_data["Metric"] == "log_loss"
                ].copy()
            else:
                # No log loss data available
                logger.warning("No log loss data available")
                return go.Figure()

            # Get mean log loss per method
            method_means = (
                log_loss_df.groupby("Method")["Loss"].mean().reset_index()
            )

            if title is None:
                title = f"Log Loss Comparison Across Methods"

            # Create bar chart
            fig = px.bar(
                method_means,
                x="Method",
                y="Loss",
                color="Method",
                title=title,
                labels={"Loss": "Log Loss"},
                color_discrete_sequence=px.colors.qualitative.Plotly,
            )

            fig.update_layout(
                title_font_size=14,
                xaxis_title_font_size=12,
                yaxis_title_font_size=12,
                paper_bgcolor="#F0F0F0",
                plot_bgcolor="#F0F0F0",
                height=figsize[1],
                width=figsize[0],
                showlegend=False,
            )

            fig.update_xaxes(showgrid=False, zeroline=False)
            fig.update_yaxes(showgrid=False, zeroline=False)

            if save_path:
                _save_figure(fig, save_path)

            return fig

        except Exception as e:
            logger.error(f"Error creating log loss comparison plot: {str(e)}")
            raise RuntimeError(
                f"Failed to create log loss comparison plot: {str(e)}"
            ) from e

    def _plot_combined_metrics(
        self,
        title: Optional[str],
        save_path: Optional[str],
        figsize: Tuple[int, int],
    ) -> go.Figure:
        """Plot combined view of both metrics."""
        try:
            # Create subplots
            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=["Quantile Loss", "Log Loss"],
                vertical_spacing=0.15,
            )

            # Plot quantile loss - filter for numeric quantiles only
            if "Metric" in self.comparison_data.columns:
                # Filter for quantile_loss metric
                ql_df = self.comparison_data[
                    self.comparison_data["Metric"] == "quantile_loss"
                ].copy()

                # Filter for numeric quantiles only (between 0 and 1)
                numeric_mask = pd.to_numeric(
                    ql_df["Percentile"], errors="coerce"
                ).notna()
                ql_df = ql_df[numeric_mask]

                # Convert to numeric and filter to valid quantile range
                ql_df["Percentile"] = pd.to_numeric(ql_df["Percentile"])
                ql_df = ql_df[
                    (ql_df["Percentile"] >= 0) & (ql_df["Percentile"] <= 1)
                ]
            else:
                # Backward compatibility
                ql_df = self.comparison_data.copy()

                # Filter for numeric percentiles only
                numeric_mask = pd.to_numeric(
                    ql_df["Percentile"], errors="coerce"
                ).notna()
                ql_df = ql_df[numeric_mask]

                # Convert to numeric and filter to valid quantile range
                ql_df["Percentile"] = pd.to_numeric(ql_df["Percentile"])
                ql_df = ql_df[
                    (ql_df["Percentile"] >= 0) & (ql_df["Percentile"] <= 1)
                ]

            if not ql_df.empty:
                for i, method in enumerate(self.methods):
                    method_data = ql_df[ql_df["Method"] == method]
                    if not method_data.empty:
                        fig.add_trace(
                            go.Bar(
                                x=method_data["Percentile"].astype(str),
                                y=method_data["Loss"],
                                name=method,
                                legendgroup=method,
                                marker_color=px.colors.qualitative.Plotly[
                                    i % len(px.colors.qualitative.Plotly)
                                ],
                            ),
                            row=1,
                            col=1,
                        )

            # Plot log loss
            if "Metric" in self.comparison_data.columns:
                ll_df = self.comparison_data[
                    self.comparison_data["Metric"] == "log_loss"
                ]

                if not ll_df.empty:
                    method_means = ll_df.groupby("Method")["Loss"].mean()
                    fig.add_trace(
                        go.Bar(
                            x=list(method_means.index),
                            y=list(method_means.values),
                            marker_color=[
                                px.colors.qualitative.Plotly[
                                    i % len(px.colors.qualitative.Plotly)
                                ]
                                for i in range(len(method_means))
                            ],
                            showlegend=False,
                        ),
                        row=2,
                        col=1,
                    )

            if title is None:
                title = "Method comparison - combined metrics"

            fig.update_layout(
                title=title,
                barmode="group",
                height=figsize[1] * 1.5,
                width=figsize[0],
                paper_bgcolor="#F0F0F0",
                plot_bgcolor="#F0F0F0",
                showlegend=True,
            )

            fig.update_xaxes(
                title_text="Quantile", row=1, col=1, showgrid=False
            )
            fig.update_xaxes(title_text="Method", row=2, col=1, showgrid=False)
            fig.update_yaxes(title_text="Loss", row=1, col=1, showgrid=False)
            fig.update_yaxes(
                title_text="Log loss", row=2, col=1, showgrid=False
            )

            if save_path:
                _save_figure(fig, save_path)

            return fig

        except Exception as e:
            logger.error(f"Error creating combined metrics plot: {str(e)}")
            raise RuntimeError(
                f"Failed to create combined metrics plot: {str(e)}"
            ) from e

    def _plot_stacked_contribution(
        self,
        title: Optional[str],
        save_path: Optional[str],
        figsize: Tuple[int, int],
    ) -> go.Figure:
        """Plot stacked bar chart showing rank-based contribution scores.

        Uses the same rank-based methodology as autoimpute's model selection:
        1. Rank models for each variable based on their loss
        2. Stack the ranks to show total rank score
        3. Lower total rank indicates better overall performance
        """
        try:
            # Calculate rank-based contributions for each method and variable
            contribution_data = []

            # First, collect all losses by variable
            losses_by_variable = {}
            for var in self.variables:
                var_data = self.comparison_data[
                    self.comparison_data["Imputed Variable"] == var
                ]
                if not var_data.empty:
                    # Get metric type for this variable
                    if "Metric" in var_data.columns:
                        metric_type = var_data["Metric"].iloc[0]
                    else:
                        metric_type = "quantile_loss"

                    # Get losses for each method for this variable
                    method_losses = {}
                    for method in self.methods:
                        method_var_data = var_data[
                            var_data["Method"] == method
                        ]
                        if not method_var_data.empty:
                            method_losses[method] = method_var_data[
                                "Loss"
                            ].mean()
                        else:
                            method_losses[method] = np.inf

                    losses_by_variable[var] = {
                        "losses": method_losses,
                        "metric_type": metric_type,
                    }

            # Calculate ranks for each variable
            for var, var_info in losses_by_variable.items():
                method_losses = var_info["losses"]
                metric_type = var_info["metric_type"]

                # Convert to pandas Series and rank (lower loss = better rank = 1)
                losses_series = pd.Series(method_losses)
                ranks = losses_series.rank(na_option="bottom", method="min")

                # Add rank data for each method
                for method in self.methods:
                    contribution_data.append(
                        {
                            "Method": method,
                            "Variable": var,
                            "Rank": (
                                ranks[method]
                                if method in ranks
                                else len(self.methods)
                            ),
                            "Metric": metric_type,
                        }
                    )

            if not contribution_data:
                logger.warning(
                    "No data available for stacked contribution plot"
                )
                return go.Figure()

            contrib_df = pd.DataFrame(contribution_data)

            # Create stacked bar chart
            fig = go.Figure()

            # Add traces for each variable
            for var in self.variables:
                var_data = contrib_df[contrib_df["Variable"] == var]
                if not var_data.empty:
                    # Determine color based on metric type
                    metric_type = (
                        var_data["Metric"].iloc[0]
                        if "Metric" in var_data.columns
                        else "quantile_loss"
                    )
                    color_idx = 0 if metric_type == "quantile_loss" else 1

                    fig.add_trace(
                        go.Bar(
                            x=var_data["Method"],
                            y=var_data["Rank"],
                            name=f"{var} ({metric_type.replace('_', ' ')})",
                            marker_color=px.colors.qualitative.Set2[
                                color_idx % len(px.colors.qualitative.Set2)
                            ],
                            text=var_data["Rank"].round(1),
                            textposition="inside",
                        )
                    )

            if title is None:
                title = "Rank-based mmodel performance by variable (lower is better)"

            fig.update_layout(
                title=title,
                barmode="stack",
                xaxis_title="Method",
                yaxis_title="Total rank score",
                height=figsize[1],
                width=figsize[0],
                paper_bgcolor="#F0F0F0",
                plot_bgcolor="#F0F0F0",
                legend_title="Variable (Metric)",
            )

            fig.update_xaxes(showgrid=False, zeroline=False)
            fig.update_yaxes(showgrid=False, zeroline=False)

            if save_path:
                _save_figure(fig, save_path)

            return fig

        except Exception as e:
            logger.error(f"Error creating stacked contribution plot: {str(e)}")
            raise RuntimeError(
                f"Failed to create stacked contribution plot: {str(e)}"
            ) from e

    def summary(self, format: str = "wide") -> pd.DataFrame:
        """Generate a summary table of the comparison results.

        Args:
            format: 'wide' for methods as columns, 'long' for stacked format

        Returns:
            Summary DataFrame
        """
        logger.debug(f"Generating {format} format summary")

        if format == "wide":
            # Pivot table with methods as columns
            summary = self.comparison_data.pivot_table(
                index=["Imputed Variable", "Percentile"],
                columns="Method",
                values="Loss",
                aggfunc="mean",
            )
            # Add a row for average across all quantiles
            overall_mean = summary.mean()
            overall_mean.name = ("Overall", "Mean")
            summary = pd.concat([summary, overall_mean.to_frame().T])

        else:  # long format
            # Group by method and calculate statistics
            summary = (
                self.comparison_data.groupby("Method")["Loss"]
                .agg(["mean", "std", "min", "max"])
                .round(6)
            )

        logger.debug(f"Summary generated with shape {summary.shape}")
        return summary

    def get_best_method(self, criterion: str = "mean") -> str:
        """Identify the best performing method.

        Args:
            criterion: 'mean' for average loss, 'median' for median loss

        Returns:
            Name of the best performing method
        """
        logger.debug(f"Finding best method using {criterion} criterion")

        if criterion == "mean":
            method_scores = self.comparison_data.groupby("Method")[
                "Loss"
            ].mean()
        elif criterion == "median":
            method_scores = self.comparison_data.groupby("Method")[
                "Loss"
            ].median()
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        best_method = method_scores.idxmin()
        logger.info(
            f"Best method: {best_method} with {criterion} loss = {method_scores[best_method]:.6f}"
        )
        return best_method

    def __repr__(self) -> str:
        """String representation of the MethodComparisonResults object."""
        return (
            f"MethodComparisonResults(methods={self.methods}, "
            f"variables={len(self.variables)}, "
            f"shape={self.comparison_data.shape})"
        )


def method_comparison_results(
    data: Union[pd.DataFrame, Dict[str, Dict[str, Dict]]],
    metric_name: Optional[str] = None,
    metric: str = "quantile_loss",
    data_format: str = "wide",
) -> MethodComparisonResults:
    """Create a MethodComparisonResults object from comparison data.

    This unified factory function supports multiple input formats:
    - "wide": DataFrame with methods as index and quantiles as columns
    - "long": DataFrame with columns ["Method", "Imputed Variable", "Percentile", "Loss"]
    - Dict: Dual metrics format from cross-validation results

    Args:
        data: Either DataFrame or Dict containing performance data.
        metric_name: Name of the metric being compared (deprecated, use metric).
        metric: Which metric to visualize: 'quantile_loss', 'log_loss', or 'combined'.
        data_format: Format of the input data.

    Returns:
        MethodComparisonResults object for visualization
    """
    # Note: quantiles parameter is kept for backward compatibility but not used

    return MethodComparisonResults(
        comparison_data=data,
        metric=metric,
        data_format=data_format,
    )
