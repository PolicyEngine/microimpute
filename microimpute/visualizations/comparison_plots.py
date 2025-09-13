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
from plotly.subplots import make_subplots

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

    def _plot_wide_format(
        self,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (
            PLOT_CONFIG["width"],
            PLOT_CONFIG["height"],
        ),
    ) -> go.Figure:
        """Internal method to create wide format plot with methods on x-axis.

        Args:
            title: Custom title for the plot
            save_path: Path to save the plot
            figsize: Figure size as (width, height) in pixels

        Returns:
            Plotly figure object
        """
        logger.debug("Creating wide format comparison plot")

        # Filter out mean_loss from variables for individual plots
        plot_variables = [v for v in self.variables if v != "mean_loss"]

        if not plot_variables:
            logger.warning("No individual variables to plot")
            return self._create_single_mean_plot(title, save_path, figsize)

        # Create subplots
        n_vars = len(plot_variables)
        n_cols = min(3, n_vars)  # Max 3 columns
        n_rows = (n_vars + n_cols - 1) // n_cols

        subplot_titles = plot_variables
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
        )

        # Color palette
        colors = px.colors.qualitative.Plotly

        # Add traces for each variable
        for idx, var in enumerate(plot_variables):
            row = (idx // n_cols) + 1
            col = (idx % n_cols) + 1

            var_data = self.comparison_data[
                self.comparison_data["Imputed Variable"] == var
            ]

            # Get unique percentiles for this variable
            percentiles = sorted(var_data["Percentile"].unique())

            for pidx, percentile in enumerate(percentiles):
                if isinstance(percentile, str):
                    continue  # Skip mean_loss percentile

                perc_data = var_data[var_data["Percentile"] == percentile]

                # Sort by method name for consistent ordering
                perc_data = perc_data.sort_values("Method")

                fig.add_trace(
                    go.Bar(
                        x=perc_data["Method"],
                        y=perc_data["Loss"],
                        name=f"q={percentile:.2f}",
                        marker_color=colors[pidx % len(colors)],
                        showlegend=(
                            idx == 0
                        ),  # Only show legend for first subplot
                    ),
                    row=row,
                    col=col,
                )

        # Update layout
        fig.update_layout(
            title=title or "Method Comparison Across Variables",
            barmode="group",
            width=figsize[0],
            height=figsize[1],
            paper_bgcolor="#F0F0F0",
            plot_bgcolor="#F0F0F0",
            margin=dict(l=50, r=50, t=100, b=50),
        )

        # Update axes
        fig.update_xaxes(title_text="Method", showgrid=False)
        fig.update_yaxes(title_text=self.metric_name, showgrid=False)

        if save_path:
            _save_figure(fig, save_path)

        return fig

    def _plot_long_format(
        self,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (
            PLOT_CONFIG["width"],
            PLOT_CONFIG["height"],
        ),
    ) -> go.Figure:
        """Internal method to create long format plot with quantiles on x-axis.

        Args:
            title: Custom title for the plot
            save_path: Path to save the plot
            figsize: Figure size as (width, height) in pixels

        Returns:
            Plotly figure object
        """
        logger.debug("Creating long format comparison plot")

        # Filter out mean_loss from variables for individual plots
        plot_variables = [v for v in self.variables if v != "mean_loss"]

        if not plot_variables:
            logger.warning("No individual variables to plot")
            return self._create_single_mean_plot(title, save_path, figsize)

        # Create subplots
        n_vars = len(plot_variables)
        n_cols = min(2, n_vars)  # Max 2 columns for long format
        n_rows = (n_vars + n_cols - 1) // n_cols

        subplot_titles = plot_variables
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.15,
            horizontal_spacing=0.12,
        )

        # Color palette
        colors = px.colors.qualitative.Plotly

        # Add traces for each variable
        for idx, var in enumerate(plot_variables):
            row = (idx // n_cols) + 1
            col = (idx % n_cols) + 1

            var_data = self.comparison_data[
                self.comparison_data["Imputed Variable"] == var
            ]

            # Filter out non-numeric percentiles
            var_data = var_data[var_data["Percentile"] != "mean_loss"]
            var_data = var_data.copy()
            var_data["Percentile"] = pd.to_numeric(
                var_data["Percentile"], errors="coerce"
            )
            var_data = var_data.dropna(subset=["Percentile"])

            for midx, method in enumerate(self.methods):
                method_data = var_data[
                    var_data["Method"] == method
                ].sort_values("Percentile")

                fig.add_trace(
                    go.Scatter(
                        x=method_data["Percentile"],
                        y=method_data["Loss"],
                        mode="lines+markers",
                        name=method,
                        line=dict(color=colors[midx % len(colors)]),
                        marker=dict(size=6),
                        showlegend=(
                            idx == 0
                        ),  # Only show legend for first subplot
                    ),
                    row=row,
                    col=col,
                )

        # Update layout
        fig.update_layout(
            title=title or f"{self.metric_name} Comparison Across Variables",
            width=figsize[0],
            height=figsize[1],
            paper_bgcolor="#F0F0F0",
            plot_bgcolor="#F0F0F0",
            margin=dict(l=50, r=50, t=100, b=50),
        )

        # Update axes
        fig.update_xaxes(
            title_text="Quantile", showgrid=True, gridcolor="white"
        )
        fig.update_yaxes(
            title_text=self.metric_name, showgrid=True, gridcolor="white"
        )

        if save_path:
            _save_figure(fig, save_path)

        return fig

    def _create_single_mean_plot(
        self,
        title: Optional[str],
        save_path: Optional[str],
        figsize: Tuple[int, int],
    ) -> go.Figure:
        """Create a simple bar plot for mean loss comparison.

        Args:
            title: Plot title
            save_path: Path to save the plot
            figsize: Figure size

        Returns:
            Plotly figure object
        """
        logger.debug("Creating single mean loss plot")

        # Filter for mean_loss data
        mean_data = self.comparison_data[
            (self.comparison_data["Imputed Variable"] == "mean_loss")
            & (self.comparison_data["Percentile"] == "mean_loss")
        ]

        if mean_data.empty:
            logger.warning("No mean loss data available")
            mean_data = (
                self.comparison_data.groupby("Method")["Loss"]
                .mean()
                .reset_index()
            )

        fig = go.Figure(
            data=[
                go.Bar(
                    x=mean_data["Method"],
                    y=mean_data["Loss"],
                    marker_color=px.colors.qualitative.Plotly[
                        : len(mean_data)
                    ],
                )
            ]
        )

        fig.update_layout(
            title=title
            or f"Average {self.metric_name} Comparison Across Methods",
            xaxis_title="Method",
            yaxis_title=f"Average {self.metric_name}",
            width=figsize[0],
            height=figsize[1],
            paper_bgcolor="#F0F0F0",
            plot_bgcolor="#F0F0F0",
        )

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        if save_path:
            _save_figure(fig, save_path)

        return fig

    def plot(
        self,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_mean: bool = False,
        figsize: Tuple[int, int] = (
            PLOT_CONFIG["width"],
            PLOT_CONFIG["height"],
        ),
    ) -> go.Figure:
        """Create a plot of the comparison results using the initialized format.

        Args:
            title: Custom title for the plot
            save_path: Path to save the plot
            show_mean: If True, shows only mean comparison, otherwise shows all quantiles
            figsize: Figure size as (width, height) in pixels

        Returns:
            Plotly figure object
        """
        if show_mean:
            return self._create_single_mean_plot(title, save_path, figsize)
        elif self.data_format == "long":
            return self._plot_long_format(title, save_path, figsize)
        else:  # default to wide format
            return self._plot_wide_format(title, save_path, figsize)

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
