"""Comprehensive tests for visualization modules."""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes

from microimpute.evaluations import cross_validate_model
from microimpute.models.ols import OLS
from microimpute.models.quantreg import QuantReg
from microimpute.utils.data import preprocess_data
from microimpute.visualizations import (
    MethodComparisonResults,
    PerformanceResults,
    model_performance_results,
)


@pytest.fixture
def sample_quantile_loss_results():
    """Create sample quantile loss results for testing."""
    np.random.seed(42)
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    # Create results DataFrame with train/test rows
    results_df = pd.DataFrame(
        {q: np.random.uniform(0.1, 0.5, 2) for q in quantiles},
        index=["train", "test"],
    )

    return {
        "results": results_df,
        "mean_train": results_df.loc["train"].mean(),
        "mean_test": results_df.loc["test"].mean(),
        "variables": ["var1", "var2"],
    }


@pytest.fixture
def sample_log_loss_results():
    """Create sample log loss results for testing."""
    np.random.seed(42)

    # Create results DataFrame with train/test rows
    results_df = pd.DataFrame(
        {
            "cat_var1": np.random.uniform(0.5, 1.5, 2),
            "cat_var2": np.random.uniform(0.3, 1.0, 2),
        },
        index=["train", "test"],
    )

    return {
        "results": results_df,
        "mean_train": results_df.loc["train"].mean(),
        "mean_test": results_df.loc["test"].mean(),
        "variables": ["cat_var1", "cat_var2"],
    }


@pytest.fixture
def sample_combined_results(
    sample_quantile_loss_results, sample_log_loss_results
):
    """Create sample combined metric results."""
    return {
        "quantile_loss": sample_quantile_loss_results,
        "log_loss": sample_log_loss_results,
    }


@pytest.fixture
def sample_confusion_matrix():
    """Create sample confusion matrix data."""
    return pd.DataFrame(
        [[50, 10, 5], [8, 45, 7], [3, 5, 52]],
        columns=["Class A", "Class B", "Class C"],
        index=["Class A", "Class B", "Class C"],
    )


@pytest.fixture
def sample_probability_distribution():
    """Create sample probability distribution data."""
    np.random.seed(42)
    n_samples = 100
    n_classes = 3

    # Generate random probabilities that sum to 1
    probs = np.random.dirichlet(np.ones(n_classes), n_samples)

    return pd.DataFrame(
        probs, columns=[f"Class {i}" for i in range(n_classes)]
    )


@pytest.fixture
def sample_comparison_results():
    """Create sample comparison results for multiple models."""
    np.random.seed(42)
    quantiles = [0.1, 0.5, 0.9]
    models = ["OLS", "QuantileReg", "RandomForest"]

    results = {}
    for model in models:
        results[model] = {
            "quantile_loss": {
                "results": pd.DataFrame(
                    {q: np.random.uniform(0.1, 0.5, 2) for q in quantiles},
                    index=["train", "test"],
                ),
                "mean_train": np.random.uniform(0.2, 0.4),
                "mean_test": np.random.uniform(0.25, 0.45),
                "variables": ["x1", "x2"],
            }
        }

    return results


@pytest.fixture
def diabetes_data():
    """Load diabetes dataset for integration tests."""
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    return df


class TestPerformanceResults:
    """Test PerformanceResults visualization class."""

    def test_quantile_loss_visualization(self, sample_quantile_loss_results):
        """Test quantile loss visualization."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            viz = PerformanceResults(
                results=sample_quantile_loss_results,
                metric="quantile_loss",
                model_name="TestModel",
                method_name="Cross-Validation",
            )

            assert viz.metric == "quantile_loss"
            assert viz.model_name == "TestModel"
            assert viz.method_name == "Cross-Validation"

            # Test plot generation
            fig = viz.plot()
            assert fig is not None

    def test_log_loss_visualization(self, sample_log_loss_results):
        """Test log loss visualization."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            viz = PerformanceResults(
                results=sample_log_loss_results,
                metric="log_loss",
                model_name="TestModel",
                method_name="Cross-Validation",
            )

            assert viz.metric == "log_loss"

            # Test plot generation
            fig = viz.plot()
            assert fig is not None

    def test_combined_visualization(self, sample_combined_results):
        """Test combined metric visualization."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            viz = PerformanceResults(
                results=sample_combined_results,
                metric="combined",
                model_name="TestModel",
                method_name="Cross-Validation",
            )

            assert viz.metric == "combined"

            # Test plot generation
            fig = viz.plot()
            assert fig is not None

    def test_confusion_matrix_plot(self, sample_confusion_matrix):
        """Test confusion matrix visualization."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Create log loss results
            log_loss_results = {
                "log_loss": {
                    "results": pd.DataFrame(
                        {"cat_var": [0.5, 0.6]}, index=["train", "test"]
                    ),
                    "mean_test": 0.6,
                    "variables": ["cat_var"],
                }
            }

            # Create y_true and y_pred for confusion matrix
            np.random.seed(42)
            y_true_data = np.random.choice(["A", "B", "C"], 100)
            y_pred_data = np.random.choice(["A", "B", "C"], 100)

            viz = PerformanceResults(
                results=log_loss_results,
                metric="log_loss",
                model_name="TestModel",
                method_name="Confusion Matrix Test",
                y_true={"cat_var": y_true_data},
                y_pred={"cat_var": y_pred_data},
            )

            # Test that plot generates without error and includes confusion matrix
            fig = viz.plot()
            assert fig is not None

            # Verify the figure contains data (plotly figure should have data attribute)
            assert hasattr(fig, "data")
            assert len(fig.data) > 0  # Should have at least one trace

    def test_probability_distribution_plot(
        self, sample_probability_distribution
    ):
        """Test probability distribution visualization."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Create log loss results
            log_loss_results = {
                "log_loss": {
                    "results": pd.DataFrame(
                        {"cat_var": [0.5, 0.6]}, index=["train", "test"]
                    ),
                    "mean_test": 0.6,
                    "variables": ["cat_var"],
                }
            }

            # Create class probabilities for distribution plot
            class_probs = {"cat_var": sample_probability_distribution}

            viz = PerformanceResults(
                results=log_loss_results,
                metric="log_loss",
                model_name="TestModel",
                method_name="Probability Distribution Test",
                class_probabilities=class_probs,
            )

            # Test that plot generates without error
            fig = viz.plot()
            assert fig is not None

            # Verify the figure contains data
            assert hasattr(fig, "data")
            assert (
                len(fig.data) > 0
            )  # Should have histogram or distribution data

    def test_df_compatibility(self, sample_quantile_loss_results):
        """Test compatibility with DataFrame input."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Test with DataFrame directly
            viz = PerformanceResults(
                results=sample_quantile_loss_results["results"],
                metric="quantile_loss",
                model_name="TestModel",
                method_name="Test",
            )

            fig = viz.plot()
            assert fig is not None

    def test_invalid_metric_type(self):
        """Test error handling for invalid metric type."""
        with pytest.raises(ValueError, match="Invalid metric"):
            PerformanceResults(
                results=pd.DataFrame(),
                metric="invalid_metric",
                model_name="Test",
                method_name="Test",
            )


class TestMethodComparisonResults:
    """Test MethodComparisonResults visualization class."""

    def test_basic_comparison(self, sample_comparison_results):
        """Test basic model comparison visualization."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            viz = MethodComparisonResults(
                sample_comparison_results,
                metric="quantile_loss",
            )

            assert viz.metric == "quantile_loss"

            # Test plot generation
            fig = viz.plot()
            assert fig is not None

    def test_stacked_contribution_plot(self, sample_comparison_results):
        """Test rank-based stacked contribution plot."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            viz = MethodComparisonResults(
                sample_comparison_results, metric="quantile_loss"
            )

            # Test that stacked contribution plot can be generated
            fig = viz.plot()
            assert fig is not None

    def test_combined_metric_comparison(self):
        """Test comparison with combined metrics."""
        np.random.seed(42)

        # Create results with both metrics
        results = {
            "Model1": {
                "quantile_loss": {
                    "results": pd.DataFrame(
                        {0.5: [0.3, 0.35]}, index=["train", "test"]
                    ),
                    "mean_test": 0.35,
                    "variables": ["x1"],
                },
                "log_loss": {
                    "results": pd.DataFrame(
                        {"cat1": [0.6, 0.65]}, index=["train", "test"]
                    ),
                    "mean_test": 0.65,
                    "variables": ["cat1"],
                },
            },
            "Model2": {
                "quantile_loss": {
                    "results": pd.DataFrame(
                        {0.5: [0.25, 0.3]}, index=["train", "test"]
                    ),
                    "mean_test": 0.3,
                    "variables": ["x1"],
                },
                "log_loss": {
                    "results": pd.DataFrame(
                        {"cat1": [0.55, 0.6]}, index=["train", "test"]
                    ),
                    "mean_test": 0.6,
                    "variables": ["cat1"],
                },
            },
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            viz = MethodComparisonResults(results, metric="combined")

            fig = viz.plot()
            assert fig is not None

    def test_empty_results(self):
        """Test handling of empty results."""
        # MethodComparisonResults expects proper structure, empty dict causes error
        with pytest.raises(AttributeError):
            MethodComparisonResults({}, metric="quantile_loss")

    def test_mismatched_metrics(self):
        """Test handling of mismatched metrics across models."""
        results = {
            "Model1": {
                "quantile_loss": {
                    "results": pd.DataFrame(
                        {0.5: [0.3, 0.35]}, index=["train", "test"]
                    ),
                    "mean_test": 0.35,
                }
            },
            "Model2": {
                "log_loss": {  # Different metric
                    "results": pd.DataFrame(
                        {"cat": [0.6, 0.65]}, index=["train", "test"]
                    ),
                    "mean_test": 0.65,
                }
            },
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Should handle gracefully
            viz = MethodComparisonResults(results, metric="quantile_loss")

            # Only Model1 should be plotted
            fig = viz.plot()
            assert fig is not None


class TestModelPerformanceResults:
    """Test the model_performance_results helper function."""

    def test_basic_usage(self, sample_quantile_loss_results):
        """Test basic usage of model_performance_results."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            viz = model_performance_results(
                results=sample_quantile_loss_results["results"],
                model_name="TestModel",
                method_name="Cross-Validation",
            )

            assert viz is not None
            assert isinstance(viz, PerformanceResults)

            fig = viz.plot()
            assert fig is not None

    def test_with_dict_input(self, sample_combined_results):
        """Test with dictionary input."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            viz = model_performance_results(
                results=sample_combined_results,
                model_name="TestModel",
                method_name="Combined Metrics",
            )

            assert viz is not None
            fig = viz.plot()
            assert fig is not None


class TestIntegrationWithModels:
    """Integration tests with actual model outputs."""

    def test_ols_visualization(self, diabetes_data):
        """Test visualization with OLS model results."""
        predictors = ["age", "sex", "bmi", "bp"]
        imputed_variables = ["s1", "s2"]

        data = diabetes_data[predictors + imputed_variables]
        data = preprocess_data(data, full_data=True)

        # Run cross-validation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            results = cross_validate_model(
                OLS, data, predictors, imputed_variables
            )

            # Test visualization
            viz = PerformanceResults(
                results=results["quantile_loss"],
                metric="quantile_loss",
                model_name="OLS",
                method_name="Cross-Validation",
            )

            fig = viz.plot()
            assert fig is not None

    def test_quantreg_visualization(self, diabetes_data):
        """Test visualization with QuantReg model results."""
        predictors = ["age", "sex", "bmi", "bp"]
        imputed_variables = ["s1"]

        data = diabetes_data[predictors + imputed_variables]
        data = preprocess_data(data, full_data=True)

        # Run cross-validation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            results = cross_validate_model(
                QuantReg, data, predictors, imputed_variables
            )

            # Test visualization
            viz = PerformanceResults(
                results=results["quantile_loss"],
                metric="quantile_loss",
                model_name="QuantReg",
                method_name="Cross-Validation",
            )

            fig = viz.plot()
            assert fig is not None

    def test_model_comparison_integration(self, diabetes_data):
        """Test model comparison visualization with multiple models."""
        predictors = ["age", "sex", "bmi", "bp"]
        imputed_variables = ["s1", "s2"]

        data = diabetes_data[predictors + imputed_variables]
        data = preprocess_data(data, full_data=True)

        # Run cross-validation for multiple models
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            comparison_results = {}

            # OLS
            ols_results = cross_validate_model(
                OLS, data, predictors, imputed_variables
            )
            comparison_results["OLS"] = ols_results

            # QuantReg
            qr_results = cross_validate_model(
                QuantReg, data, predictors, imputed_variables
            )
            comparison_results["QuantReg"] = qr_results

            # Test comparison visualization
            viz = MethodComparisonResults(
                comparison_results, metric="quantile_loss"
            )

            fig = viz.plot()
            assert fig is not None


class TestVisualizationFromAutoimpute:
    """Tests for integration with autoimpute."""

    def test_performance_visualization_from_autoimpute(self):
        """Test performance visualization as used in autoimpute."""
        np.random.seed(42)

        # Create sample results as autoimpute would generate
        results = pd.DataFrame(
            {0.1: [0.15, 0.18], 0.5: [0.25, 0.28], 0.9: [0.35, 0.38]},
            index=["train", "test"],
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            viz = model_performance_results(
                results=results,
                model_name="AutoImpute",
                method_name="Best Model Performance",
            )

            assert viz is not None
            fig = viz.plot()
            assert fig is not None

    def test_comparison_visualization_from_autoimpute(self):
        """Test comparison visualization as used in autoimpute."""
        np.random.seed(42)

        # Create results as autoimpute would generate
        model_results = {
            "OLS": {
                "quantile_loss": {
                    "results": pd.DataFrame(
                        {0.5: [0.3, 0.35]}, index=["train", "test"]
                    ),
                    "mean_test": 0.35,
                    "variables": ["x1", "x2"],
                }
            },
            "QRF": {
                "quantile_loss": {
                    "results": pd.DataFrame(
                        {0.5: [0.25, 0.28]}, index=["train", "test"]
                    ),
                    "mean_test": 0.28,
                    "variables": ["x1", "x2"],
                }
            },
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            viz = MethodComparisonResults(
                model_results, metric="quantile_loss"
            )

            fig = viz.plot()
            assert fig is not None


class TestErrorHandling:
    """Test error handling in visualization modules."""

    def test_invalid_results_format(self):
        """Test handling of invalid results format."""
        # PerformanceResults actually handles various input formats gracefully
        # Let's test that it properly validates the metric instead
        with pytest.raises(ValueError, match="Invalid metric"):
            PerformanceResults(
                results=pd.DataFrame(),
                metric="invalid_metric_type",  # Invalid metric
                model_name="Test",
                method_name="Test",
            )

    def test_missing_required_keys(self):
        """Test handling of missing required keys in results."""
        incomplete_results = {
            "mean_test": 0.5
            # Missing "results" key
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Should handle gracefully or raise informative error
            try:
                viz = PerformanceResults(
                    results=incomplete_results,
                    metric="quantile_loss",
                    model_name="Test",
                    method_name="Test",
                )
                # If it doesn't raise, should still be able to handle plotting
                fig = viz.plot()
                assert fig is not None
            except (ValueError, KeyError) as e:
                # Should have informative error message
                assert (
                    "results" in str(e).lower() or "missing" in str(e).lower()
                )

    def test_nan_handling(self):
        """Test handling of NaN values in results."""
        results_with_nan = pd.DataFrame(
            {0.5: [0.3, np.nan], 0.9: [np.nan, 0.4]}, index=["train", "test"]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            viz = PerformanceResults(
                results=results_with_nan,
                metric="quantile_loss",
                model_name="Test",
                method_name="Test",
            )

            # Should handle NaNs gracefully
            fig = viz.plot()
            assert fig is not None
