"""Visualization interfaces for imputation model performance.

This module provides classes and functions for visualizing the performance
of imputation models, following a statsmodels-like interface, built on plotly.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pydantic import validate_call

from microimpute.config import PLOT_CONFIG, QUANTILES, VALIDATE_CONFIG

logger = logging.getLogger(__name__)


class PerformanceResults:
    """Class to store and visualize model performance results.

    This class provides an interface for storing and visualizing
    performance metrics, with methods like plot() and summary().
    """

    def __init__(
        self,
        results: pd.DataFrame,
        model_name: Optional[str] = None,
        method_name: Optional[str] = None,
    ):
        """Initialize PerformanceResults with train/test performance data.

        Args:
            results: DataFrame with train and test rows, quantiles
                as columns, and loss values.
            model_name: Name of the model used for imputation.
            method_name: Name of the imputation method.
        """
        self.results = results.copy()
        self.model_name = model_name or "Unknown Model"
        self.method_name = method_name or "Unknown Method"

        # Validate inputs
        required_indices = ["train", "test"]
        available_indices = self.results.index.tolist()
        missing_indices = [
            idx for idx in required_indices if idx not in available_indices
        ]

        if missing_indices:
            logger.warning(
                f"Missing indices in results DataFrame: {missing_indices}"
            )
            logger.info(f"Available indices: {available_indices}")

        # Convert column names to strings if they are not already
        self.results.columns = [str(col) for col in self.results.columns]

    def plot(
        self,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (
            PLOT_CONFIG["width"],
            PLOT_CONFIG["height"],
        ),
    ) -> go.Figure:
        """Plot the performance comparison between training and testing
        sets across quantiles.

        Args:
            title: Custom title for the plot. If None, a default title is used.
            save_path: Path to save the plot. If None, the plot is displayed.
            figsize: Figure size as (width, height) in pixels.

        Returns:
            Plotly figure object

        Raises:
            RuntimeError: If plot creation or saving fails
        """
        logger.debug(
            f"Creating train-test performance plot from results shape {self.results.shape}"
        )
        palette = px.colors.qualitative.Plotly
        train_color = palette[2]
        test_color = palette[3]

        try:
            logger.debug("Creating Plotly figure")
            fig = go.Figure()

            # Add bars for training data if present
            if "train" in self.results.index:
                logger.debug("Adding training data bars")
                fig.add_trace(
                    go.Bar(
                        x=self.results.columns,
                        y=self.results.loc["train"],
                        name="Train",
                        marker_color=train_color,
                    )
                )

            # Add bars for test data if present
            if "test" in self.results.index:
                logger.debug("Adding test data bars")
                fig.add_trace(
                    go.Bar(
                        x=self.results.columns,
                        y=self.results.loc["test"],
                        name="Test",
                        marker_color=test_color,
                    )
                )

            logger.debug("Updating plot layout")
            fig.update_layout(
                title=title,
                xaxis_title="Quantile",
                yaxis_title="Average Quantile Loss",
                barmode="group",
                width=figsize[0],
                height=figsize[1],
                paper_bgcolor="#F0F0F0",
                plot_bgcolor="#F0F0F0",
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                margin=dict(l=50, r=50, t=80, b=50),
            )

            fig.update_xaxes(showgrid=False, zeroline=False)
            fig.update_yaxes(showgrid=False, zeroline=False)

            if save_path:
                try:
                    logger.info(f"Saving plot to {save_path}")

                    # Ensure directory exists
                    save_dir = os.path.dirname(save_path)
                    if save_dir and not os.path.exists(save_dir):
                        logger.debug(f"Creating directory: {save_dir}")
                        os.makedirs(save_dir, exist_ok=True)

                    # Try to save as image if kaleido is available
                    try:
                        fig.write_image(save_path)
                        logger.info(f"Plot saved as image to {save_path}")
                    except Exception as img_error:
                        logger.warning(
                            f"Could not save image to {save_path}: {str(img_error)}. "
                            "Install kaleido to enable image export."
                        )

                    # Always save HTML version for interactive viewing
                    html_path = save_path.replace(".jpg", ".html")
                    fig.write_html(html_path)

                    logger.info(f"Plot saved as HTML to {html_path}")
                except Exception as e:
                    logger.error(f"Error saving train-test plot: {str(e)}")
                    raise RuntimeError(
                        f"Failed to save plot to {save_path}"
                    ) from e

            logger.debug("Train-test performance plot created successfully")
            return fig

        except Exception as e:
            logger.error(
                f"Error creating train-test performance plot: {str(e)}"
            )
            raise RuntimeError(
                f"Failed to create train-test performance plot: {str(e)}"
            ) from e

    def summary(self) -> pd.DataFrame:
        """Provide a summary of model performance statistics.

        Returns:
            DataFrame with performance statistics including:
                - Mean loss per dataset (train/test)
                - Loss by quantile
                - Difference between train and test
                - Overfitting ratio (test/train)
        """
        logger.debug("Generating performance summary statistics")

        try:
            summary_data = {}

            # Overall mean loss
            mean_loss = self.results.mean(axis=1)
            summary_data["mean_loss"] = mean_loss.to_dict()

            # Calculate train/test difference if both are available
            if all(idx in self.results.index for idx in ["train", "test"]):
                diff = self.results.loc["test"] - self.results.loc["train"]
                summary_data["difference"] = {
                    "mean": diff.mean(),
                    "max": diff.max(),
                    "min": diff.min(),
                }

                # Calculate overfitting ratio (test/train)
                ratio = self.results.loc["test"] / self.results.loc["train"]
                summary_data["ratio"] = {
                    "mean": ratio.mean(),
                    "max": ratio.max(),
                    "min": ratio.min(),
                }

            summary_df = pd.DataFrame(
                {
                    "Model": self.model_name,
                    "Method": self.method_name,
                    "Mean Train Loss": summary_data.get("mean_loss", {}).get(
                        "train", np.nan
                    ),
                    "Mean Test Loss": summary_data.get("mean_loss", {}).get(
                        "test", np.nan
                    ),
                    "Mean Difference": summary_data.get("difference", {}).get(
                        "mean", np.nan
                    ),
                    "Overfitting Ratio": summary_data.get("ratio", {}).get(
                        "mean", np.nan
                    ),
                },
                index=[0],
            )

            return summary_df

        except Exception as e:
            logger.error(f"Error generating performance summary: {str(e)}")
            raise RuntimeError(
                f"Failed to generate performance summary: {str(e)}"
            ) from e


class MethodComparisonResults:
    """Class to store and visualize performance comparison across different methods.

    This unified comparison class provides an interface for comparing and visualizing performance metrics across different imputation methods, with support for diverse dataset shapes, different quantiles, and various metrics.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        metric_name: str = "Quantile Loss",
        quantiles: List[float] = QUANTILES,
        data_format: str = "wide",
    ):
        """Initialize MethodComparison with performance data.

        This class supports multiple input formats through the data_format parameter:
        - "wide": DataFrame with methods as index and quantiles as columns (and
                optional 'mean_loss' column)
        - "long": DataFrame with columns ["Method", "Imputed Variable", "Percentile", "Loss"]

        Args:
            data: DataFrame containing performance data in one of the supported formats.
            metric_name: Name of the metric being compared (default: "Quantile Loss").
            quantiles: List of quantile values (e.g., [0.05, 0.1, ...]).
            data_format: Format of the input data ("wide" or "long").

        Raises:
            ValueError: If input DataFrame is invalid or in unsupported format
        """
        self.quantiles = quantiles or QUANTILES
        self.metric_name = metric_name
        self.data_format = data_format

        # Process data based on format
        if data_format == "wide":
            self._process_wide_format(data)
        elif data_format == "long":
            self._process_long_format(data)
        else:
            raise ValueError(
                f"Unsupported data_format: {data_format}. "
                "Must be 'wide' or 'long'."
            )

    def _process_wide_format(self, data: pd.DataFrame) -> None:
        """Process data in wide format (methods as index, quantiles as columns)."""
        # Validate inputs
        if data.empty:
            logger.error("Empty DataFrame provided for plotting")
            raise ValueError("DataFrame cannot be empty")

        self.method_results_df = data.copy()

        expected_columns = [str(q) for q in self.quantiles]
        if not all(
            str(q) in self.method_results_df.columns
            or q in self.method_results_df.columns
            for q in self.quantiles
        ):
            logger.warning(
                f"Some quantiles not found in DataFrame columns. "
                f"Expected: {expected_columns}, Found: {list(self.method_results_df.columns)}"
            )

        self.methods = self.method_results_df.index.tolist()
        self.data_subset = "test"  # Default to test data for wide format

        # Compute mean loss if not already present
        if "mean_loss" not in self.method_results_df.columns:
            quantile_cols = [
                col
                for col in self.method_results_df.columns
                if col != "mean_loss"
                and (
                    col in map(str, self.quantiles)
                    or col in map(float, self.quantiles)
                )
            ]
            if quantile_cols:
                self.method_results_df["mean_loss"] = self.method_results_df[
                    quantile_cols
                ].mean(axis=1)

    def _process_long_format(self, data: pd.DataFrame) -> None:
        """Process data in long format (Method, Imputed Variable, Percentile, Loss)."""
        # Validate inputs
        required_columns = ["Method", "Imputed Variable", "Percentile", "Loss"]
        missing_columns = [
            col for col in required_columns if col not in data.columns
        ]
        if missing_columns:
            logger.error(
                f"Missing required columns for long format: {missing_columns}"
            )
            raise ValueError(
                f"Long format DataFrame must contain columns: {required_columns}"
            )

        self.loss_comparison_df = data.copy()

        # Convert to wide format for internal storage
        df_avg = self.loss_comparison_df[
            self.loss_comparison_df["Imputed Variable"] == "mean_loss"
        ]
        df_regular = df_avg[df_avg["Percentile"] != "mean_loss"]

        wide_df = df_regular.pivot(
            index="Method", columns="Percentile", values="Loss"
        )

        # Add mean loss column
        wide_df["mean_loss"] = wide_df.mean(axis=1)

        self.method_results_df = wide_df
        self.methods = wide_df.index.tolist()
        self.data_subset = "test"  # Default to test data for long format

    def plot(
        self,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (
            PLOT_CONFIG["width"],
            PLOT_CONFIG["height"],
        ),
        show_mean: bool = True,
    ) -> go.Figure:
        """Plot a bar chart comparing performance across different imputation methods.

        Args:
            title: Custom title for the plot. If None, a default title is used.
            save_path: Path to save the plot. If None, the plot is displayed.
            figsize: Figure size as (width, height) in pixels.
            show_mean: Whether to show horizontal lines for mean loss values.

        Returns:
            Plotly figure object

        Raises:
            ValueError: If data_subset is invalid or not available
            RuntimeError: If plot creation or saving fails
        """
        logger.debug(
            f"Creating method comparison plot with DataFrame of shape {self.method_results_df.shape}"
        )

        try:
            # Convert DataFrame to long format for plotting
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

            # Add a horizontal line for the mean loss if present and requested
            if show_mean and "mean_loss" in self.method_results_df.columns:
                logger.debug("Adding mean loss markers to plot")
                for i, method in enumerate(self.method_results_df.index):
                    mean_loss = self.method_results_df.loc[method, "mean_loss"]
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
                try:
                    logger.info(f"Saving plot to {save_path}")

                    # Ensure directory exists
                    save_dir = os.path.dirname(save_path)
                    if save_dir and not os.path.exists(save_dir):
                        logger.debug(f"Creating directory: {save_dir}")
                        os.makedirs(save_dir, exist_ok=True)

                    # Try to save as image if kaleido is available
                    try:
                        fig.write_image(save_path)
                        logger.info(f"Plot saved as image to {save_path}")
                    except Exception as img_error:
                        logger.warning(
                            f"Could not save image to {save_path}: {str(img_error)}. "
                            "Install kaleido to enable image export."
                        )

                    # Always save as HTML for interactive viewing
                    html_path = save_path.replace(".jpg", ".html").replace(
                        ".png", ".html"
                    )
                    if html_path == save_path:
                        html_path = save_path + ".html"
                    fig.write_html(html_path)

                    logger.info(f"Plot saved as HTML to {html_path}")
                except Exception as e:
                    logger.error(f"Error saving plot: {str(e)}")
                    raise RuntimeError(
                        f"Failed to save plot to {save_path}"
                    ) from e

            logger.debug("Plot creation completed successfully")
            return fig

        except Exception as e:
            logger.error(f"Error creating method comparison plot: {str(e)}")
            raise RuntimeError(
                f"Failed to create method comparison plot: {str(e)}"
            ) from e

    def summary(
        self,
        data_subset: Optional[str] = None,
    ) -> pd.DataFrame:
        """Provide a summary of method comparison statistics.

        Args:
            data_subset: Which data subset to summarize ("train" or "test").
                Only applicable for train_test format data.

        Returns:
            DataFrame with summary statistics by method including:
                - Mean loss across quantiles
                - Best/worst quantiles and their corresponding losses

        Raises:
            ValueError: If data_subset is invalid or not available
        """
        logger.debug("Generating method comparison summary statistics")

        try:
            methods = self.method_results_df.index.tolist()
            summary_data = []

            for method in methods:
                method_data = self.method_results_df.loc[method]

                quantile_cols = [
                    col for col in method_data.index if col != "mean_loss"
                ]

                if "mean_loss" in method_data.index:
                    mean_loss = method_data["mean_loss"]
                else:
                    mean_loss = (
                        method_data[quantile_cols].mean()
                        if quantile_cols
                        else np.nan
                    )

                if quantile_cols:
                    best_quantile = method_data[quantile_cols].idxmin()
                    best_loss = method_data[quantile_cols].min()
                    worst_quantile = method_data[quantile_cols].idxmax()
                    worst_loss = method_data[quantile_cols].max()
                else:
                    best_quantile = worst_quantile = "N/A"
                    best_loss = worst_loss = np.nan

                summary_data.append(
                    {
                        "Method": method,
                        f"Mean {self.metric_name}": mean_loss,
                        "Best Quantile": best_quantile,
                        f"Best {self.metric_name}": best_loss,
                        "Worst Quantile": worst_quantile,
                        f"Worst {self.metric_name}": worst_loss,
                    }
                )

            # Create summary DataFrame
            summary_df = pd.DataFrame(summary_data)

            # Sort by mean loss
            if (
                not summary_df.empty
                and f"Mean {self.metric_name}" in summary_df.columns
            ):
                summary_df = summary_df.sort_values(
                    f"Mean {self.metric_name}", ascending=True
                )

            return summary_df

        except Exception as e:
            logger.error(
                f"Error generating method comparison summary: {str(e)}"
            )
            raise RuntimeError(
                f"Failed to generate method comparison summary: {str(e)}"
            ) from e


# Functions to create visualization objects
@validate_call(config=VALIDATE_CONFIG)
def model_performance_results(
    results: pd.DataFrame,
    model_name: Optional[str] = None,
    method_name: Optional[str] = None,
) -> PerformanceResults:
    """Create a PerformanceResults object from train/test results.

    Args:
        results: DataFrame with train and test rows, quantiles
            as columns, and loss values.
        model_name: Name of the model used for imputation.
        method_name: Name of the imputation method.

    Returns:
        PerformanceResults object for visualization
    """
    return PerformanceResults(
        results=results,
        model_name=model_name,
        method_name=method_name,
    )


@validate_call(config=VALIDATE_CONFIG)
def method_comparison_results(
    data: pd.DataFrame,
    metric_name: str = "Quantile Loss",
    quantiles: List[float] = QUANTILES,
    data_format: str = "wide",
) -> MethodComparisonResults:
    """Create a MethodComparison object for visualizing performance comparisons.

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
        MethodComparison object for visualization
    """
    return MethodComparisonResults(
        data=data,
        metric_name=metric_name,
        quantiles=quantiles,
        data_format=data_format,
    )
