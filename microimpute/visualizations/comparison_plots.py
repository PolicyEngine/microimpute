"""Multi-method comparison visualization

This module provides comprehensive visualization tools for comparing the performance
of multiple imputation methods. It creates interactive plots and heatmaps that help
identify the best performing method for different variables and quantiles.

Key components:
    - MethodComparisonResults: container class for comparison data with plotting methods
    - method_comparison_results: factory function to create comparison visualizations
    - Support for variable-specific and aggregate performance comparisons
    - Interactive Plotly-based visualizations with customizable layouts
"""

import logging
from typing import List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from microimpute.config import PLOT_CONFIG
from microimpute.visualizations.performance_plots import _save_figure

logger = logging.getLogger(__name__)


class MethodComparisonResults:
    """Class to store and visualize comparison results across methods."""

    def __init__(
        self,
        comparison_data: pd.DataFrame,
        metric_name: str = "Quantile Loss",
        imputed_variables: Optional[List[str]] = None,
        data_format: str = "wide",
    ):
        """Initialize MethodComparisonResults with comparison data.

        Args:
            comparison_data: DataFrame with comparison data in one of two formats:
                - "wide": DataFrame with methods as index and quantiles as columns
                - "long": DataFrame with columns 'Method', 'Imputed Variable', 'Percentile', 'Loss'
            metric_name: Name of the metric being compared (e.g., "Quantile Loss", "MAE", "RMSE")
            imputed_variables: List of variable names that were imputed
            data_format: Input data format - 'wide' or 'long'
        """
        self.metric_name = metric_name
        self.imputed_variables = imputed_variables or []
        self.data_format = data_format

        # Process data based on input format
        if data_format == "wide":
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
        self.methods = self.comparison_data["Method"].unique().tolist()
        self.variables = (
            self.comparison_data["Imputed Variable"].unique().tolist()
        )

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

    def plot(
        self,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_mean: bool = True,
        figsize: Tuple[int, int] = (
            PLOT_CONFIG["width"],
            PLOT_CONFIG["height"],
        ),
    ) -> go.Figure:
        """Plot a bar chart comparing performance across different imputation methods.

        Args:
            title: Custom title for the plot. If None, a default title is used.
            save_path: Path to save the plot. If None, the plot is displayed.
            show_mean: Whether to show horizontal lines for mean loss values.
            figsize: Figure size as (width, height) in pixels.

        Returns:
            Plotly figure object

        Raises:
            ValueError: If data_subset is invalid or not available
            RuntimeError: If plot creation or saving fails
        """
        logger.debug(
            f"Creating method comparison plot with {len(self.methods)} methods"
        )

        try:
            # Prepare data for plotting - we need it in a specific format
            # regardless of how it was input
            if hasattr(self, "method_results_df"):
                # Data came in wide format, convert to long for plotting
                plot_df = self.method_results_df.reset_index().rename(
                    columns={"index": "Method"}
                )

                id_vars = ["Method"]
                value_vars = [
                    col
                    for col in plot_df.columns
                    if col not in id_vars and col != "mean_loss"
                ]

                melted_df = pd.melt(
                    plot_df,
                    id_vars=id_vars,
                    value_vars=value_vars,
                    var_name="Percentile",
                    value_name=self.metric_name,
                )

                melted_df["Percentile"] = melted_df["Percentile"].astype(str)

            else:
                # Data is already in long format (comparison_data)
                # Filter out mean_loss entries for the bar chart
                melted_df = self.comparison_data[
                    (self.comparison_data["Percentile"] != "mean_loss")
                    & (self.comparison_data["Imputed Variable"] != "mean_loss")
                ].copy()
                melted_df = melted_df.rename(
                    columns={"Loss": self.metric_name}
                )
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

            # Add horizontal lines for mean loss if present and requested
            if show_mean:
                logger.debug("Adding mean loss markers to plot")

                if (
                    hasattr(self, "method_results_df")
                    and "mean_loss" in self.method_results_df.columns
                ):
                    # Wide format data has mean_loss column
                    for i, method in enumerate(self.method_results_df.index):
                        mean_loss = self.method_results_df.loc[
                            method, "mean_loss"
                        ]
                        fig.add_shape(
                            type="line",
                            x0=-0.5,
                            y0=mean_loss,
                            x1=len(value_vars) - 0.5,
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
                else:
                    # Calculate means from the data
                    for i, method in enumerate(self.methods):
                        method_data = melted_df[melted_df["Method"] == method]
                        if not method_data.empty:
                            mean_loss = method_data[self.metric_name].mean()
                            # Get number of unique percentiles for x1 position
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
    data: pd.DataFrame,
    metric_name: str = "Quantile Loss",
    quantiles: List[float] = None,
    data_format: str = "wide",
) -> MethodComparisonResults:
    """Create a MethodComparisonResults object from comparison data.

    This unified factory function supports multiple input formats:
    - "wide": DataFrame with methods as index and quantiles as columns (and
             optional 'mean_loss' column)
    - "long": DataFrame with columns ["Method", "Imputed Variable", "Percentile", "Loss"]

    Args:
        data: DataFrame containing performance data in one of the supported formats.
        metric_name: Name of the metric being compared (default: "Quantile Loss").
        quantiles: List of quantile values (e.g., [0.05, 0.1, ...]).
        data_format: Format of the input data ("wide" or "long").

    Returns:
        MethodComparisonResults object for visualization
    """
    # Note: quantiles parameter is kept for backward compatibility but not used
    # The quantiles are inferred from the data itself

    return MethodComparisonResults(
        comparison_data=data,
        metric_name=metric_name,
        imputed_variables=None,  # Will be inferred from data
        data_format=data_format,
    )
