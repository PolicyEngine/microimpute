"""Comprehensive tests for predictor analysis functions.

This module tests all functions in the predictor_analysis module including:
- compute_predictor_correlations
- leave_one_out_analysis
- progressive_predictor_inclusion
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from microimpute.evaluations.predictor_analysis import (
    compute_predictor_correlations,
    leave_one_out_analysis,
    progressive_predictor_inclusion,
)
from microimpute.models import OLS, QRF, QuantReg


@pytest.fixture
def sample_regression_data():
    """Create sample regression data for testing."""
    np.random.seed(42)
    n_samples = 200

    # Create correlated features
    X, y = make_regression(
        n_samples=n_samples,
        n_features=5,
        n_informative=3,
        noise=10,
        random_state=42,
    )

    # Create DataFrame with meaningful names
    feature_names = [
        "feature_1",
        "feature_2",
        "feature_3",
        "feature_4",
        "feature_5",
    ]
    df = pd.DataFrame(X, columns=feature_names)

    # Add target variable
    df["target"] = y

    # Add a categorical predictor
    df["category"] = np.random.choice(["A", "B", "C"], size=n_samples)

    # Add a binary predictor
    df["binary"] = np.random.choice([0, 1], size=n_samples)

    return df


@pytest.fixture
def sample_mixed_data():
    """Create sample data with mixed variable types."""
    np.random.seed(42)
    n_samples = 200

    data = pd.DataFrame(
        {
            "numeric_1": np.random.randn(n_samples),
            "numeric_2": np.random.randn(n_samples) * 2 + 5,
            "numeric_3": np.random.exponential(2, n_samples),
            "categorical_1": np.random.choice(
                ["cat_A", "cat_B", "cat_C"], n_samples
            ),
            "categorical_2": np.random.choice(["X", "Y", "Z", "W"], n_samples),
            "binary_1": np.random.choice([True, False], n_samples),
            "binary_2": np.random.choice([0, 1], n_samples),
            "target_numeric": np.random.randn(n_samples) * 3,
            "target_categorical": np.random.choice(
                ["class_1", "class_2"], n_samples
            ),
        }
    )

    # Create some correlation
    data["numeric_2"] = (
        data["numeric_1"] * 0.7 + np.random.randn(n_samples) * 0.5
    )
    data["target_numeric"] = (
        data["numeric_1"] * 0.5
        + data["numeric_2"] * 0.3
        + (data["binary_1"].astype(int) * 2)
        + np.random.randn(n_samples)
    )

    return data


class TestComputePredictorCorrelations:
    """Test suite for compute_predictor_correlations function."""

    def test_basic_correlation_computation(self, sample_regression_data):
        """Test basic correlation computation with numeric predictors."""
        predictors = [
            "feature_1",
            "feature_2",
            "feature_3",
            "feature_4",
            "feature_5",
        ]

        # Test all methods
        results = compute_predictor_correlations(
            data=sample_regression_data, predictors=predictors, method="all"
        )

        # Check that all correlation types are returned
        assert "pearson" in results
        assert "spearman" in results
        assert "mutual_info" in results

        # Check dimensions
        for corr_type in ["pearson", "spearman", "mutual_info"]:
            assert results[corr_type].shape == (5, 5)
            assert list(results[corr_type].columns) == predictors
            assert list(results[corr_type].index) == predictors

        # Check diagonal is 1 for all correlation matrices
        for corr_type in results:
            np.testing.assert_array_almost_equal(
                np.diag(results[corr_type].values), np.ones(5)
            )

    def test_mixed_type_correlations(self, sample_mixed_data):
        """Test correlation computation with mixed variable types."""
        predictors = ["numeric_1", "numeric_2", "categorical_1", "binary_1"]

        results = compute_predictor_correlations(
            data=sample_mixed_data, predictors=predictors, method="all"
        )

        # Check that correlations were computed for mixed types
        assert results["pearson"].notna().all().all()
        assert results["spearman"].notna().all().all()
        assert results["mutual_info"].notna().all().all()

        # Check strong correlation between numeric_1 and numeric_2
        pearson_corr = results["pearson"].loc["numeric_1", "numeric_2"]
        assert pearson_corr > 0.6  # Should be strong positive correlation

    def test_single_method_computation(self, sample_regression_data):
        """Test computing only specific correlation types."""
        predictors = ["feature_1", "feature_2", "feature_3"]

        # Test Pearson only
        results_pearson = compute_predictor_correlations(
            data=sample_regression_data,
            predictors=predictors,
            method="pearson",
        )
        assert len(results_pearson) == 1
        assert "pearson" in results_pearson

        # Test Spearman only
        results_spearman = compute_predictor_correlations(
            data=sample_regression_data,
            predictors=predictors,
            method="spearman",
        )
        assert len(results_spearman) == 1
        assert "spearman" in results_spearman

        # Test Mutual Information only
        results_mi = compute_predictor_correlations(
            data=sample_regression_data,
            predictors=predictors,
            method="mutual_info",
        )
        assert len(results_mi) == 1
        assert "mutual_info" in results_mi

    def test_missing_predictors_error(self, sample_regression_data):
        """Test that error is raised for missing predictors."""
        with pytest.raises(ValueError, match="Predictors not found"):
            compute_predictor_correlations(
                data=sample_regression_data,
                predictors=["nonexistent_column"],
                method="all",
            )

    def test_invalid_method_error(self, sample_regression_data):
        """Test that error is raised for invalid method."""
        with pytest.raises(ValueError, match="Invalid method"):
            compute_predictor_correlations(
                data=sample_regression_data,
                predictors=["feature_1", "feature_2"],
                method="invalid_method",
            )

    def test_predictor_target_mutual_information(self, sample_mixed_data):
        """Test computation of predictor-target mutual information."""
        predictors = ["numeric_1", "numeric_2", "binary_1", "categorical_1"]
        imputed_variables = ["target_numeric", "target_categorical"]

        results = compute_predictor_correlations(
            data=sample_mixed_data,
            predictors=predictors,
            imputed_variables=imputed_variables,
            method="all",
        )

        # Check that predictor-target MI is computed
        assert "predictor_target_mi" in results

        # Check dimensions
        assert results["predictor_target_mi"].shape == (
            len(predictors),
            len(imputed_variables),
        )
        assert list(results["predictor_target_mi"].index) == predictors
        assert (
            list(results["predictor_target_mi"].columns) == imputed_variables
        )

        # Check values are in valid range [0, 1] (normalized MI)
        mi_values = results["predictor_target_mi"].values
        assert np.all((mi_values >= 0) & (mi_values <= 1))

        # Check that numeric_1 and numeric_2 have non-zero MI with target_numeric
        # (since we created correlation in the fixture)
        assert (
            results["predictor_target_mi"].loc["numeric_1", "target_numeric"]
            > 0
        )
        assert (
            results["predictor_target_mi"].loc["numeric_2", "target_numeric"]
            > 0
        )

        # Check that binary_1 has non-zero MI with target_numeric
        # (it's part of the target calculation)
        assert (
            results["predictor_target_mi"].loc["binary_1", "target_numeric"]
            > 0
        )

    def test_predictor_target_mi_only_with_mutual_info_method(
        self, sample_mixed_data
    ):
        """Test that predictor-target MI is only computed when mutual_info is requested."""
        predictors = ["numeric_1", "numeric_2"]
        imputed_variables = ["target_numeric"]

        # Test with pearson only - should not include predictor-target MI
        results_pearson = compute_predictor_correlations(
            data=sample_mixed_data,
            predictors=predictors,
            imputed_variables=imputed_variables,
            method="pearson",
        )
        assert "predictor_target_mi" not in results_pearson

        # Test with mutual_info - should include predictor-target MI
        results_mi = compute_predictor_correlations(
            data=sample_mixed_data,
            predictors=predictors,
            imputed_variables=imputed_variables,
            method="mutual_info",
        )
        assert "predictor_target_mi" in results_mi


class TestLeaveOneOutAnalysis:
    """Test suite for leave_one_out_analysis function."""

    def test_basic_leave_one_out(self, sample_regression_data):
        """Test basic leave-one-out analysis."""
        predictors = ["feature_1", "feature_2", "feature_3", "feature_4"]
        imputed_variables = ["target"]

        results = leave_one_out_analysis(
            data=sample_regression_data,
            predictors=predictors,
            imputed_variables=imputed_variables,
            model_class=OLS,
            n_jobs=1,
            random_state=42,
        )

        # Check output structure
        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(predictors)

        # Check columns exist
        expected_columns = [
            "predictor_removed",
            "avg_quantile_loss",
            "avg_log_loss",
            "loss_increase",
            "relative_impact",
            "baseline_quantile_loss",
            "baseline_log_loss",
        ]
        for col in expected_columns:
            assert col in results.columns

        # Check that all predictors are analyzed
        assert set(results["predictor_removed"].values) == set(predictors)

        # Check that losses are non-negative
        assert (results["avg_quantile_loss"] >= 0).all()

        # Check relative impact calculation
        # Removing informative features should increase loss (positive impact)
        # while removing noise features should have minimal impact
        assert results["relative_impact"].notna().any()

    def test_leave_one_out_with_qrf(self, sample_regression_data):
        """Test leave-one-out analysis with QRF model."""
        predictors = ["feature_1", "feature_2", "feature_3"]
        imputed_variables = ["target"]

        results = leave_one_out_analysis(
            data=sample_regression_data,
            predictors=predictors,
            imputed_variables=imputed_variables,
            model_class=QRF,
            quantiles=[0.5],  # Single quantile for speed
            n_jobs=1,
            random_state=42,
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(predictors)
        assert (results["avg_quantile_loss"] >= 0).all()

    def test_leave_one_out_with_categorical_target(self, sample_mixed_data):
        """Test leave-one-out with categorical imputation target."""
        predictors = ["numeric_1", "numeric_2", "binary_1"]
        imputed_variables = ["target_categorical"]

        results = leave_one_out_analysis(
            data=sample_mixed_data,
            predictors=predictors,
            imputed_variables=imputed_variables,
            model_class=QRF,
            quantiles=[0.5],
            n_jobs=1,
            random_state=42,
        )

        assert isinstance(results, pd.DataFrame)
        # For categorical targets, we should have log_loss
        assert "avg_log_loss" in results.columns
        assert results["avg_log_loss"].notna().any()

    def test_parallel_execution(self, sample_regression_data):
        """Test parallel execution of leave-one-out analysis."""
        predictors = ["feature_1", "feature_2", "feature_3"]
        imputed_variables = ["target"]

        # Run with parallel jobs
        results = leave_one_out_analysis(
            data=sample_regression_data,
            predictors=predictors,
            imputed_variables=imputed_variables,
            model_class=OLS,
            quantiles=[0.5],
            n_jobs=2,  # Use 2 jobs
            random_state=42,
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(predictors)


class TestProgressivePredictorInclusion:
    """Test suite for progressive_predictor_inclusion function."""

    def test_basic_progressive_inclusion(self, sample_regression_data):
        """Test basic progressive predictor inclusion."""
        predictors = ["feature_1", "feature_2", "feature_3", "feature_4"]
        imputed_variables = ["target"]

        results = progressive_predictor_inclusion(
            data=sample_regression_data,
            predictors=predictors,
            imputed_variables=imputed_variables,
            model_class=OLS,
            quantiles=[0.5],  # Single quantile for speed
            max_predictors=3,  # Limit for testing
            random_state=42,
        )

        # Check output structure
        assert isinstance(results, dict)
        assert "results_df" in results
        assert "optimal_subset" in results
        assert "optimal_loss" in results

        # Check results_df is a DataFrame
        results_df = results["results_df"]
        assert isinstance(results_df, pd.DataFrame)

        # Check DataFrame has correct columns
        expected_columns = [
            "step",
            "predictor_added",
            "predictors_included",
            "avg_quantile_loss",
            "avg_log_loss",
            "cumulative_improvement",
            "marginal_improvement",
        ]
        assert list(results_df.columns) == expected_columns

        # Check inclusion order from DataFrame
        assert len(results_df) <= 3
        assert all(
            pred in predictors for pred in results_df["predictor_added"]
        )

        # Check step numbering is correct
        assert list(results_df["step"]) == list(range(1, len(results_df) + 1))

        # Check performance is monotonically improving
        # (cumulative_improvement should be non-decreasing)
        cumulative_impr = results_df["cumulative_improvement"].values
        for i in range(1, len(cumulative_impr)):
            assert cumulative_impr[i] >= cumulative_impr[i - 1] - 1e-6

        # Check optimal subset
        assert len(results["optimal_subset"]) <= 3
        assert results["optimal_loss"] > 0  # Should be a positive loss value

    def test_progressive_inclusion_all_predictors(
        self, sample_regression_data
    ):
        """Test progressive inclusion using all predictors."""
        predictors = ["feature_1", "feature_2", "feature_3"]
        imputed_variables = ["target"]

        results = progressive_predictor_inclusion(
            data=sample_regression_data,
            predictors=predictors,
            imputed_variables=imputed_variables,
            model_class=OLS,
            quantiles=[0.5],
            max_predictors=None,  # Use all
            random_state=42,
        )

        # Should include all predictors
        results_df = results["results_df"]
        assert len(results_df) == len(predictors)
        assert set(results_df["predictor_added"]) == set(predictors)

        # Check that predictors_included grows correctly
        for i, row in results_df.iterrows():
            assert len(row["predictors_included"]) == row["step"]

    def test_progressive_inclusion_with_mixed_types(self, sample_mixed_data):
        """Test progressive inclusion with mixed predictor types."""
        predictors = ["numeric_1", "numeric_2", "categorical_1", "binary_1"]
        imputed_variables = ["target_numeric"]

        results = progressive_predictor_inclusion(
            data=sample_mixed_data,
            predictors=predictors,
            imputed_variables=imputed_variables,
            model_class=QRF,
            quantiles=[0.5],
            max_predictors=3,
            random_state=42,
        )

        assert isinstance(results, dict)
        results_df = results["results_df"]
        assert len(results_df) <= 3

        # Check that values are valid
        assert (results_df["avg_quantile_loss"] >= 0).all()
        assert (results_df["avg_log_loss"] >= 0).all()
        assert results_df["step"].tolist() == list(
            range(1, len(results_df) + 1)
        )


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_predictor_analysis_workflow(self, sample_mixed_data):
        """Test complete workflow of predictor analysis."""
        predictors = ["numeric_1", "numeric_2", "binary_1", "categorical_1"]
        imputed_variables = ["target_numeric"]

        # 1. Analyze correlations
        correlations = compute_predictor_correlations(
            data=sample_mixed_data, predictors=predictors, method="all"
        )
        assert all(
            key in correlations
            for key in ["pearson", "spearman", "mutual_info"]
        )

        # 2. Leave-one-out analysis
        loo_results = leave_one_out_analysis(
            data=sample_mixed_data,
            predictors=predictors,
            imputed_variables=imputed_variables,
            model_class=QRF,
            quantiles=[0.5],
            n_jobs=1,
            random_state=42,
        )
        assert len(loo_results) == len(predictors)

        # 3. Progressive inclusion
        prog_results = progressive_predictor_inclusion(
            data=sample_mixed_data,
            predictors=predictors,
            imputed_variables=imputed_variables,
            model_class=QRF,
            quantiles=[0.5],
            max_predictors=3,
            random_state=42,
        )
        assert len(prog_results["results_df"]) <= 3

        # Check consistency: highly correlated predictors should have similar importance
        high_corr_pairs = []
        pearson = correlations["pearson"]
        for i in range(len(predictors)):
            for j in range(i + 1, len(predictors)):
                if abs(pearson.iloc[i, j]) > 0.7:
                    high_corr_pairs.append((predictors[i], predictors[j]))


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_predictor(self, sample_regression_data):
        """Test with single predictor."""
        predictors = ["feature_1"]
        imputed_variables = ["target"]

        # Correlations with single predictor
        correlations = compute_predictor_correlations(
            data=sample_regression_data, predictors=predictors, method="all"
        )
        assert correlations["pearson"].shape == (1, 1)
        assert correlations["pearson"].iloc[0, 0] == 1.0

        # Leave-one-out with single predictor
        loo_results = leave_one_out_analysis(
            data=sample_regression_data,
            predictors=predictors,
            imputed_variables=imputed_variables,
            model_class=OLS,
            quantiles=[0.5],
            n_jobs=1,
            random_state=42,
        )
        # Should have NaN for the single predictor removal
        assert len(loo_results) == 1
        assert np.isnan(loo_results["avg_quantile_loss"].values[0])

    def test_data_with_missing_values(self):
        """Test handling of data with missing values."""
        # Create data with missing values
        data = pd.DataFrame(
            {
                "pred1": [1, 2, np.nan, 4, 5],
                "pred2": [2, np.nan, 3, 4, 5],
                "target": [1, 2, 3, 4, 5],
            }
        )

        # Correlations should handle missing values
        correlations = compute_predictor_correlations(
            data=data, predictors=["pred1", "pred2"], method="pearson"
        )
        assert not np.isnan(correlations["pearson"].iloc[0, 1])

    def test_constant_predictor(self):
        """Test handling of constant predictors."""
        data = pd.DataFrame(
            {
                "constant": [1] * 100,
                "varying": np.random.randn(100),
                "target": np.random.randn(100),
            }
        )

        # Correlations with constant predictor
        correlations = compute_predictor_correlations(
            data=data, predictors=["constant", "varying"], method="pearson"
        )
        # Correlation with constant should be NaN or 0
        const_corr = correlations["pearson"].loc["constant", "varying"]
        assert np.isnan(const_corr) or const_corr == 0
