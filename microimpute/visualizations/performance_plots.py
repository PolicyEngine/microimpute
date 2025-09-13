"""Individual model performance visualization

This module provides comprehensive visualization tools for analyzing the performance
of individual imputation models. It creates interactive plots showing train/test
performance across different quantiles, helping identify overfitting and understand
model behavior at different points of the distribution.

Key components:
    - PerformanceResults: container class for model performance data with plotting methods
    - model_performance_results: factory function to create performance visualizations
    - Interactive Plotly-based visualizations with customizable styling
"""

import logging
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from microimpute.config import PLOT_CONFIG

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
                _save_figure(fig, save_path)

            logger.debug("Plot creation successful")
            return fig

        except KeyError as e:
            error_msg = f"Missing required data in results: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except ValueError as e:
            error_msg = f"Invalid data format for plotting: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def summary(self) -> pd.DataFrame:
        """Generate a summary of the performance metrics.

        Returns:
            Summary DataFrame with metrics
        """
        logger.debug("Generating performance summary")

        # Calculate summary statistics
        train_mean = (
            self.results.loc["train"].mean()
            if "train" in self.results.index
            else np.nan
        )
        test_mean = (
            self.results.loc["test"].mean()
            if "test" in self.results.index
            else np.nan
        )

        train_std = (
            self.results.loc["train"].std()
            if "train" in self.results.index
            else np.nan
        )
        test_std = (
            self.results.loc["test"].std()
            if "test" in self.results.index
            else np.nan
        )

        train_min = (
            self.results.loc["train"].min()
            if "train" in self.results.index
            else np.nan
        )
        test_min = (
            self.results.loc["test"].min()
            if "test" in self.results.index
            else np.nan
        )

        train_max = (
            self.results.loc["train"].max()
            if "train" in self.results.index
            else np.nan
        )
        test_max = (
            self.results.loc["test"].max()
            if "test" in self.results.index
            else np.nan
        )

        # Create summary DataFrame
        summary_data = {
            "Model": [self.model_name],
            "Method": [self.method_name],
            "Train Mean": [train_mean],
            "Test Mean": [test_mean],
            "Train Std": [train_std],
            "Test Std": [test_std],
            "Train Min": [train_min],
            "Test Min": [test_min],
            "Train Max": [train_max],
            "Test Max": [test_max],
            "Train/Test Ratio": [
                train_mean / test_mean if test_mean != 0 else np.nan
            ],
        }

        summary_df = pd.DataFrame(summary_data)
        logger.debug(f"Summary generated with shape {summary_df.shape}")
        return summary_df

    def __repr__(self) -> str:
        """String representation of the PerformanceResults object."""
        return (
            f"PerformanceResults(model='{self.model_name}', "
            f"method='{self.method_name}', "
            f"shape={self.results.shape})"
        )


def _save_figure(fig: go.Figure, save_path: str) -> None:
    """Save a plotly figure to file.

    Args:
        fig: Plotly figure to save
        save_path: Path where to save the figure

    Raises:
        RuntimeError: If saving fails
    """
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
        except ImportError:
            # Fall back to HTML if kaleido is not available
            html_path = save_path.rsplit(".", 1)[0] + ".html"
            fig.write_html(html_path)
            logger.warning(
                f"kaleido not available for image export. "
                f"Saved as HTML to {html_path}"
            )
    except OSError as e:
        error_msg = f"Failed to save plot to {save_path}: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


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
