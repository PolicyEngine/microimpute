"""Individual model performance visualization with dual metric support

This module provides comprehensive visualization tools for analyzing the performance
of individual imputation models. It supports both quantile loss (for numerical variables)
and log loss (for categorical variables), creating appropriate visualizations for each
metric type.

Key components:
    - PerformanceResults: container class for model performance data with plotting methods
    - model_performance_results: factory function to create performance visualizations
    - Support for quantile loss, log loss, and combined metric visualizations
    - Confusion matrix and probability distribution plots for categorical variables
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from microimpute.config import PLOT_CONFIG

logger = logging.getLogger(__name__)


class PerformanceResults:
    """Class to store and visualize model performance results with dual metric support.

    This class provides an interface for storing and visualizing
    performance metrics for both quantile loss and log loss.
    """

    def __init__(
        self,
        results: Union[pd.DataFrame, Dict[str, Dict[str, any]]],
        model_name: Optional[str] = None,
        method_name: Optional[str] = None,
        metric: str = "quantile_loss",
        class_probabilities: Optional[Dict[str, pd.DataFrame]] = None,
        y_true: Optional[Dict[str, np.ndarray]] = None,
        y_pred: Optional[Dict[str, np.ndarray]] = None,
    ):
        """Initialize PerformanceResults with train/test performance data.

        Args:
            results: Either:
                - DataFrame with train and test rows, quantiles as columns (backward compat)
                - Dict with 'quantile_loss' and/or 'log_loss' keys containing metrics
            model_name: Name of the model used for imputation.
            method_name: Name of the imputation method.
            metric: Which metric to visualize: 'quantile_loss', 'log_loss', or 'combined'
            class_probabilities: Optional dict of class probability DataFrames for categorical vars
            y_true: Optional dict of true values for confusion matrix
            y_pred: Optional dict of predicted values for confusion matrix
        """
        self.model_name = model_name or "Unknown Model"
        self.method_name = method_name or "Unknown Method"
        self.metric = metric
        self.class_probabilities = class_probabilities or {}
        self.y_true = y_true or {}
        self.y_pred = y_pred or {}

        # Handle different input formats
        if isinstance(results, pd.DataFrame):
            # Backward compatibility: single metric DataFrame
            self.results = {"quantile_loss": {"results": results.copy()}}
            self.has_quantile_loss = True
            self.has_log_loss = False
        else:
            # New format: dual metric dict
            self.results = results
            self.has_quantile_loss = (
                "quantile_loss" in results
                and results["quantile_loss"].get("results") is not None
                and not results["quantile_loss"]["results"].empty
            )
            self.has_log_loss = (
                "log_loss" in results
                and results["log_loss"].get("results") is not None
                and not results["log_loss"]["results"].empty
            )

        # Validate metric parameter
        if metric not in ["quantile_loss", "log_loss", "combined"]:
            raise ValueError(
                f"Invalid metric: {metric}. Must be 'quantile_loss', 'log_loss', or 'combined'"
            )

    def plot(
        self,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (
            PLOT_CONFIG["width"],
            PLOT_CONFIG["height"],
        ),
    ) -> go.Figure:
        """Plot the performance based on the specified metric.

        Args:
            title: Custom title for the plot. If None, a default title is used.
            save_path: Path to save the plot. If None, the plot is displayed.
            figsize: Figure size as (width, height) in pixels.

        Returns:
            Plotly figure object

        Raises:
            RuntimeError: If plot creation or saving fails
        """
        logger.debug(f"Creating performance plot for metric: {self.metric}")

        if self.metric == "quantile_loss":
            return self._plot_quantile_loss(title, save_path, figsize)
        elif self.metric == "log_loss":
            return self._plot_log_loss(title, save_path, figsize)
        elif self.metric == "combined":
            return self._plot_combined(title, save_path, figsize)
        else:
            raise ValueError(f"Invalid metric: {self.metric}")

    def _plot_quantile_loss(
        self,
        title: Optional[str],
        save_path: Optional[str],
        figsize: Tuple[int, int],
    ) -> go.Figure:
        """Plot quantile loss performance across quantiles."""
        if not self.has_quantile_loss:
            logger.warning("No quantile loss data available")
            return go.Figure()

        palette = px.colors.qualitative.Plotly
        train_color = palette[2]
        test_color = palette[3]

        try:
            fig = go.Figure()

            # Get the DataFrame for quantile loss
            ql_data = self.results["quantile_loss"]["results"]

            # Add bars for training data
            if "train" in ql_data.index:
                fig.add_trace(
                    go.Bar(
                        x=[str(x) for x in ql_data.columns],
                        y=ql_data.loc["train"].values,
                        name="Train",
                        marker_color=train_color,
                    )
                )

            # Add bars for test data
            if "test" in ql_data.index:
                fig.add_trace(
                    go.Bar(
                        x=[str(x) for x in ql_data.columns],
                        y=ql_data.loc["test"].values,
                        name="Test",
                        marker_color=test_color,
                    )
                )

            if title is None:
                title = f"Quantile Loss Performance - {self.model_name}"

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

            return fig

        except Exception as e:
            error_msg = f"Error creating quantile loss plot: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _plot_log_loss(
        self,
        title: Optional[str],
        save_path: Optional[str],
        figsize: Tuple[int, int],
    ) -> go.Figure:
        """Plot log loss performance and additional categorical metrics."""
        if not self.has_log_loss:
            logger.warning("No log loss data available")
            return go.Figure()

        ll_data = self.results["log_loss"]
        num_subplots = 1  # Base subplot for log loss bars

        # Check if we have confusion matrix data
        has_confusion = bool(self.y_true and self.y_pred)
        # Check if we have probability distributions
        has_probs = bool(self.class_probabilities)

        if has_confusion:
            num_subplots += 1
        if has_probs:
            num_subplots += 1

        # Create subplots
        subplot_titles = ["Log loss performance"]
        if has_confusion:
            subplot_titles.append("Confusion matrix")
        if has_probs:
            subplot_titles.append("Class probability distribution")

        fig = make_subplots(
            rows=num_subplots,
            cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=0.15,
            row_heights=[1] * num_subplots,
        )

        # Plot 1: Log Loss bars
        palette = px.colors.qualitative.Plotly
        train_color = palette[2]
        test_color = palette[3]

        # Get log loss values from the results DataFrame
        ll_results_df = ll_data["results"]

        if "train" in ll_results_df.index:
            # Log loss should be constant across quantiles, so take the mean
            train_loss = ll_results_df.loc["train"].mean()
            fig.add_trace(
                go.Bar(
                    x=["Train"],
                    y=[train_loss],
                    name="Train",
                    marker_color=train_color,
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        if "test" in ll_results_df.index:
            test_loss = ll_results_df.loc["test"].mean()
            fig.add_trace(
                go.Bar(
                    x=["Test"],
                    y=[test_loss],
                    name="Test",
                    marker_color=test_color,
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        current_row = 2

        # Plot 2: Confusion Matrix (if available)
        if has_confusion:
            # Use first categorical variable for confusion matrix
            var_name = list(self.y_true.keys())[0]
            y_true = self.y_true[var_name]
            y_pred = self.y_pred[var_name]

            # Create confusion matrix
            from sklearn.metrics import confusion_matrix

            labels = np.unique(np.concatenate([y_true, y_pred]))
            cm = confusion_matrix(y_true, y_pred, labels=labels)

            # Create heatmap
            fig.add_trace(
                go.Heatmap(
                    z=cm,
                    x=[str(l) for l in labels],
                    y=[str(l) for l in labels],
                    colorscale="Blues",
                    showscale=True,
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
                ),
                row=current_row,
                col=1,
            )

            fig.update_xaxes(title_text="Predicted", row=current_row, col=1)
            fig.update_yaxes(title_text="True", row=current_row, col=1)
            current_row += 1

        # Plot 3: Class Probability Distribution (if available)
        if has_probs:
            var_name = list(self.class_probabilities.keys())[0]
            probs_df = self.class_probabilities[var_name]

            # Create box plots for each class
            for col in probs_df.columns:
                fig.add_trace(
                    go.Box(
                        y=probs_df[col],
                        name=str(col),
                        boxmean=True,
                    ),
                    row=current_row,
                    col=1,
                )

            fig.update_xaxes(title_text="Class", row=current_row, col=1)
            fig.update_yaxes(
                title_text="Predicted Probability", row=current_row, col=1
            )

        if title is None:
            title = f"Log loss performance - {self.model_name}"

        fig.update_layout(
            title=title,
            height=figsize[1] * num_subplots * 0.7,
            width=figsize[0],
            paper_bgcolor="#F0F0F0",
            plot_bgcolor="#F0F0F0",
            showlegend=True,
        )

        if save_path:
            _save_figure(fig, save_path)

        return fig

    def _plot_combined(
        self,
        title: Optional[str],
        save_path: Optional[str],
        figsize: Tuple[int, int],
    ) -> go.Figure:
        """Plot combined view of both metrics."""
        if not self.has_quantile_loss and not self.has_log_loss:
            logger.warning("No metric data available")
            return go.Figure()

        # Count number of subplots needed
        num_subplots = 0
        subplot_titles = []

        if self.has_quantile_loss:
            num_subplots += 1
            subplot_titles.append("Quantile loss")
        if self.has_log_loss:
            num_subplots += 1
            subplot_titles.append("Log loss")

        fig = make_subplots(
            rows=num_subplots,
            cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=0.2,
        )

        palette = px.colors.qualitative.Plotly
        train_color = palette[2]
        test_color = palette[3]
        current_row = 1

        # Add quantile loss plot
        if self.has_quantile_loss:
            ql_data = self.results["quantile_loss"]["results"]

            if "train" in ql_data.index:
                fig.add_trace(
                    go.Bar(
                        x=[str(x) for x in ql_data.columns],
                        y=ql_data.loc["train"].values,
                        name="QL Train",
                        marker_color=train_color,
                        legendgroup="train",
                    ),
                    row=current_row,
                    col=1,
                )

            if "test" in ql_data.index:
                fig.add_trace(
                    go.Bar(
                        x=[str(x) for x in ql_data.columns],
                        y=ql_data.loc["test"].values,
                        name="QL Test",
                        marker_color=test_color,
                        legendgroup="test",
                    ),
                    row=current_row,
                    col=1,
                )

            fig.update_xaxes(title_text="Quantile", row=current_row, col=1)
            fig.update_yaxes(title_text="Loss", row=current_row, col=1)
            current_row += 1

        # Add log loss plot
        if self.has_log_loss:
            ll_data = self.results["log_loss"]["results"]

            if "train" in ll_data.index:
                train_loss = ll_data.loc["train"].mean()
                fig.add_trace(
                    go.Bar(
                        x=["Log loss"],
                        y=[train_loss],
                        name="Log loss train",
                        marker_color=train_color,
                        legendgroup="train",
                        showlegend=self.has_quantile_loss == False,
                    ),
                    row=current_row,
                    col=1,
                )

            if "test" in ll_data.index:
                test_loss = ll_data.loc["test"].mean()
                fig.add_trace(
                    go.Bar(
                        x=["Log loss"],
                        y=[test_loss],
                        name="Log loss test",
                        marker_color=test_color,
                        legendgroup="test",
                        showlegend=self.has_quantile_loss == False,
                    ),
                    row=current_row,
                    col=1,
                )

            fig.update_yaxes(title_text="Loss", row=current_row, col=1)

        if title is None:
            title = f"Combined Metric Performance - {self.model_name}"

        fig.update_layout(
            title=title,
            barmode="group",
            height=figsize[1] * num_subplots * 0.6,
            width=figsize[0],
            paper_bgcolor="#F0F0F0",
            plot_bgcolor="#F0F0F0",
            showlegend=True,
        )

        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=False, zeroline=False)

        if save_path:
            _save_figure(fig, save_path)

        return fig

    def summary(self) -> pd.DataFrame:
        """Generate a summary of the performance metrics.

        Returns:
            Summary DataFrame with metrics for available metric types
        """
        logger.debug("Generating performance summary")

        summary_data = {
            "Model": [self.model_name],
            "Method": [self.method_name],
        }

        # Add quantile loss statistics if available
        if self.has_quantile_loss:
            ql_data = self.results["quantile_loss"]["results"]

            if "train" in ql_data.index:
                train_data = ql_data.loc["train"]
                summary_data["Quantile loss train mean"] = [train_data.mean()]
                summary_data["Quantile loss train std"] = [train_data.std()]
            else:
                summary_data["Quantile loss train mean"] = [np.nan]
                summary_data["Quantile loss train std"] = [np.nan]

            if "test" in ql_data.index:
                test_data = ql_data.loc["test"]
                summary_data["Quantile loss test mean"] = [test_data.mean()]
                summary_data["Quantile loss test std"] = [test_data.std()]
            else:
                summary_data["Quantile loss test mean"] = [np.nan]
                summary_data["Quantile loss test std"] = [np.nan]

            # Add ratio
            if "train" in ql_data.index and "test" in ql_data.index:
                train_mean = ql_data.loc["train"].mean()
                test_mean = ql_data.loc["test"].mean()
                summary_data["Quantile loss train/test ratio"] = [
                    train_mean / test_mean if test_mean != 0 else np.nan
                ]
            else:
                summary_data["Quantile loss train/test ratio"] = [np.nan]

        # Add log loss statistics if available
        if self.has_log_loss:
            ll_data = self.results["log_loss"]
            ll_results_df = ll_data["results"]

            if "train" in ll_results_df.index:
                train_loss = ll_results_df.loc["train"].mean()
                summary_data["Log loss train mean"] = [train_loss]
            else:
                summary_data["Log loss train mean"] = [np.nan]

            if "test" in ll_results_df.index:
                test_loss = ll_results_df.loc["test"].mean()
                summary_data["Log loss test mean"] = [test_loss]
            else:
                summary_data["Log loss test mean"] = [np.nan]

            if (
                "train" in ll_results_df.index
                and "test" in ll_results_df.index
            ):
                train_loss = ll_results_df.loc["train"].mean()
                test_loss = ll_results_df.loc["test"].mean()
                summary_data["Log loss train/test ratio"] = [
                    train_loss / test_loss if test_loss != 0 else np.nan
                ]
            else:
                summary_data["Log loss train/test ratio"] = [np.nan]

            # Add variable info
            if "variables" in ll_data:
                summary_data["Log loss variables"] = [
                    ", ".join(ll_data["variables"])
                ]

        summary_df = pd.DataFrame(summary_data)
        logger.debug(f"Summary generated with shape {summary_df.shape}")
        return summary_df

    def __repr__(self) -> str:
        """String representation of the PerformanceResults object."""
        metrics = []
        if self.has_quantile_loss:
            metrics.append("quantile_loss")
        if self.has_log_loss:
            metrics.append("log_loss")
        return (
            f"PerformanceResults(model='{self.model_name}', "
            f"method='{self.method_name}', "
            f"metrics={metrics})"
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
    results: Union[pd.DataFrame, Dict[str, Dict[str, any]]],
    model_name: Optional[str] = None,
    method_name: Optional[str] = None,
    metric: str = "quantile_loss",
    class_probabilities: Optional[Dict[str, pd.DataFrame]] = None,
    y_true: Optional[Dict[str, np.ndarray]] = None,
    y_pred: Optional[Dict[str, np.ndarray]] = None,
) -> PerformanceResults:
    """Create a PerformanceResults object from train/test results.

    Args:
        results: Either:
            - DataFrame with train and test rows, quantiles as columns (backward compat)
            - Dict with 'quantile_loss' and/or 'log_loss' keys containing metrics
        model_name: Name of the model used for imputation.
        method_name: Name of the imputation method.
        metric: Which metric to visualize: 'quantile_loss', 'log_loss', or 'combined'
        class_probabilities: Optional dict of class probability DataFrames for categorical vars
        y_true: Optional dict of true values for confusion matrix
        y_pred: Optional dict of predicted values for confusion matrix

    Returns:
        PerformanceResults object for visualization
    """
    return PerformanceResults(
        results=results,
        model_name=model_name,
        method_name=method_name,
        metric=metric,
        class_probabilities=class_probabilities,
        y_true=y_true,
        y_pred=y_pred,
    )
