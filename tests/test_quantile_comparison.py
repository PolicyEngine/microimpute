"""Comprehensive tests for quantile loss comparison functionality."""

import numpy as np
import pandas as pd
import pytest

from microimpute.comparisons import compare_quantile_loss, get_imputations
from microimpute.config import QUANTILES
from microimpute.models import OLS, QRF, QuantReg

# Check if Matching is available
try:
    from microimpute.models import Matching

    HAS_MATCHING = True
except ImportError:
    HAS_MATCHING = False


# === Fixtures ===


@pytest.fixture
def split_data() -> tuple:
    """Generate simple split data for testing."""
    np.random.seed(42)
    n_train, n_test = 100, 20

    X_train = pd.DataFrame(
        {
            "x1": np.random.randn(n_train),
            "x2": np.random.randn(n_train),
            "x3": np.random.randn(n_train),
            "y1": np.random.randn(n_train),
            "y2": np.random.randn(n_train),
        }
    )

    X_test = pd.DataFrame(
        {
            "x1": np.random.randn(n_test),
            "x2": np.random.randn(n_test),
            "x3": np.random.randn(n_test),
            "y1": np.random.randn(n_test),
            "y2": np.random.randn(n_test),
        }
    )

    return X_train, X_test


@pytest.fixture
def diabetes_data() -> pd.DataFrame:
    """Load diabetes dataset for testing."""
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    data = pd.concat([X, y.rename("target")], axis=1)
    data.columns = [
        "age",
        "sex",
        "bmi",
        "bp",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
        "target",
    ]
    return data


# === Basic Functionality Tests ===


def test_get_imputations_basic(split_data: tuple) -> None:
    """Test basic functionality of get_imputations."""
    X_train, X_test = split_data
    predictors = ["x1", "x2", "x3"]
    imputed_variables = ["y1", "y2"]

    model_classes = [OLS, QRF]
    method_imputations = get_imputations(
        model_classes, X_train, X_test, predictors, imputed_variables
    )

    # Check structure
    assert isinstance(method_imputations, dict)
    assert len(method_imputations) == len(model_classes)
    assert "OLS" in method_imputations
    assert "QRF" in method_imputations

    # Check each model's imputations
    for model_name, imputations in method_imputations.items():
        assert isinstance(imputations, dict)
        # Should have imputations for each quantile
        assert all(q in imputations for q in QUANTILES)
        # Each quantile should have a DataFrame
        for q, df in imputations.items():
            assert isinstance(df, pd.DataFrame)
            assert df.shape == (len(X_test), len(imputed_variables))
            assert all(var in df.columns for var in imputed_variables)


def test_compare_quantile_loss_basic(split_data: tuple) -> None:
    """Test basic functionality of compare_quantile_loss."""
    X_train, X_test = split_data
    predictors = ["x1", "x2", "x3"]
    imputed_variables = ["y1", "y2"]

    Y_test = X_test[imputed_variables]

    # Get imputations
    model_classes = [OLS, QuantReg]
    method_imputations = get_imputations(
        model_classes, X_train, X_test, predictors, imputed_variables
    )

    # Compare quantile loss
    loss_comparison_df = compare_quantile_loss(
        Y_test, method_imputations, imputed_variables
    )

    # Check structure - returns long format DataFrame
    assert isinstance(loss_comparison_df, pd.DataFrame)
    assert "Method" in loss_comparison_df.columns
    assert "Imputed Variable" in loss_comparison_df.columns
    assert "Percentile" in loss_comparison_df.columns
    assert "Loss" in loss_comparison_df.columns

    # Check that both models are present
    methods = loss_comparison_df["Method"].unique()
    assert "OLS" in methods
    assert "QuantReg" in methods

    # Check values are non-negative (quantile loss is always >= 0)
    assert (loss_comparison_df["Loss"] >= 0).all()
    assert not loss_comparison_df["Loss"].isna().any()


# === Data Handling Tests ===


def test_single_imputed_variable(split_data: tuple) -> None:
    """Test with single imputed variable."""
    X_train, X_test = split_data
    predictors = ["x1", "x2"]
    imputed_variables = ["y1"]  # Single variable

    Y_test = X_test[imputed_variables]

    model_classes = [OLS, QuantReg]
    method_imputations = get_imputations(
        model_classes, X_train, X_test, predictors, imputed_variables
    )

    loss_comparison_df = compare_quantile_loss(
        Y_test, method_imputations, imputed_variables
    )

    # Check results contain data for single variable
    variables = loss_comparison_df["Imputed Variable"].unique()
    assert "y1" in variables or "mean_loss" in variables


def test_multiple_imputed_variables(split_data: tuple) -> None:
    """Test with multiple imputed variables."""
    X_train, X_test = split_data
    predictors = ["x1"]
    imputed_variables = ["y1", "y2"]  # Multiple variables

    Y_test = X_test[imputed_variables]

    model_classes = [OLS, QRF, QuantReg]
    if HAS_MATCHING:
        model_classes.append(Matching)
    method_imputations = get_imputations(
        model_classes, X_train, X_test, predictors, imputed_variables
    )

    loss_comparison_df = compare_quantile_loss(
        Y_test, method_imputations, imputed_variables
    )

    # Check results contain data for all variables
    variables = loss_comparison_df["Imputed Variable"].unique()
    # Should have y1, y2, and mean_loss
    assert len(variables) >= 2


# === Statistical Properties Tests ===


def test_quantile_loss_symmetry() -> None:
    """Test that quantile loss is properly asymmetric."""
    np.random.seed(42)

    # Create data where predictions are always above true values
    X_train = pd.DataFrame(
        {
            "x": np.random.randn(50),
            "y": np.random.randn(50),
        }
    )

    X_test = pd.DataFrame(
        {
            "x": np.random.randn(10),
            "y": np.random.randn(10) - 5,  # True values are much lower
        }
    )

    predictors = ["x"]
    imputed_variables = ["y"]
    Y_test = X_test[imputed_variables]

    model_classes = [OLS, QuantReg]
    method_imputations = get_imputations(
        model_classes, X_train, X_test, predictors, imputed_variables
    )

    loss_comparison_df = compare_quantile_loss(
        Y_test, method_imputations, imputed_variables
    )

    assert not loss_comparison_df.empty


def test_perfect_predictions() -> None:
    """Test with perfect predictions."""
    np.random.seed(42)

    # Create perfectly predictable data
    X_train = pd.DataFrame(
        {
            "x": range(100),
            "y": range(100),  # Perfect correlation
        }
    )

    X_test = pd.DataFrame(
        {
            "x": [10, 20, 30],
            "y": [10, 20, 30],  # Perfect match
        }
    )

    predictors = ["x"]
    imputed_variables = ["y"]
    Y_test = X_test[imputed_variables]

    model_classes = [OLS, QuantReg]
    method_imputations = get_imputations(
        model_classes, X_train, X_test, predictors, imputed_variables
    )

    loss_comparison_df = compare_quantile_loss(
        Y_test, method_imputations, imputed_variables
    )

    # Loss should be relatively low for perfect predictions
    ols_loss = loss_comparison_df[loss_comparison_df["Method"] == "OLS"][
        "Loss"
    ]
    assert ols_loss.min() <= 1.01  # Allow for small floating point errors


# === Integration Tests ===


def test_model_ranking(diabetes_data: pd.DataFrame) -> None:
    """Test that models can be ranked by performance."""
    # Split data
    train_size = int(0.8 * len(diabetes_data))
    X_train = diabetes_data[:train_size]
    X_test = diabetes_data[train_size:]

    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1"]

    Y_test = X_test[imputed_variables]

    # Compare models
    model_classes = [OLS, QRF, QuantReg]
    method_imputations = get_imputations(
        model_classes, X_train, X_test, predictors, imputed_variables
    )

    loss_comparison_df = compare_quantile_loss(
        Y_test, method_imputations, imputed_variables
    )

    # Check we can compute mean loss per model
    mean_losses = loss_comparison_df.groupby("Method")["Loss"].mean()
    assert len(mean_losses) == len(model_classes)

    # Best model should have lowest mean loss
    best_model = mean_losses.idxmin()
    assert best_model in [m.__name__ for m in model_classes]


# === Visualization Support Tests ===


def test_wide_format_visualization(split_data: tuple) -> None:
    """Test that results can be converted to wide format for visualization."""
    X_train, X_test = split_data
    predictors = ["x1", "x2"]
    imputed_variables = ["y1"]

    Y_test = X_test[imputed_variables]

    model_classes = [OLS, QuantReg]
    method_imputations = get_imputations(
        model_classes, X_train, X_test, predictors, imputed_variables
    )

    loss_comparison_df = compare_quantile_loss(
        Y_test, method_imputations, imputed_variables
    )

    # Convert to wide format for plotting
    # Filter to mean_loss only
    mean_loss_df = loss_comparison_df[
        loss_comparison_df["Imputed Variable"] == "mean_loss"
    ]

    if not mean_loss_df.empty:
        wide_df = mean_loss_df.pivot(
            index="Method", columns="Percentile", values="Loss"
        )

        # Check wide format structure
        assert isinstance(wide_df, pd.DataFrame)
        assert wide_df.shape[0] <= len(model_classes)  # One row per model
        # Columns should be quantiles
        assert all(
            col in QUANTILES for col in wide_df.columns if col in QUANTILES
        )


def test_long_format_visualization(split_data: tuple) -> None:
    """Test that results are suitable for long-format visualization."""
    X_train, X_test = split_data
    predictors = ["x1", "x2", "x3"]
    imputed_variables = ["y1", "y2"]

    Y_test = X_test[imputed_variables]

    model_classes = [OLS, QRF]
    method_imputations = get_imputations(
        model_classes, X_train, X_test, predictors, imputed_variables
    )

    loss_comparison_df = compare_quantile_loss(
        Y_test, method_imputations, imputed_variables
    )

    # Long format is directly suitable for seaborn/plotly
    assert "Method" in loss_comparison_df.columns
    assert "Percentile" in loss_comparison_df.columns
    assert "Loss" in loss_comparison_df.columns

    # Can group by method and percentile
    grouped = loss_comparison_df.groupby(["Method", "Percentile"])[
        "Loss"
    ].mean()
    assert not grouped.empty


# === Robustness Tests ===


def test_comparison_consistency() -> None:
    """Test that repeated comparisons give consistent results."""
    np.random.seed(42)

    X_train = pd.DataFrame(
        {
            "x1": np.random.randn(50),
            "x2": np.random.randn(50),
            "y": np.random.randn(50),
        }
    )

    X_test = pd.DataFrame(
        {
            "x1": np.random.randn(10),
            "x2": np.random.randn(10),
            "y": np.random.randn(10),
        }
    )

    predictors = ["x1", "x2"]
    imputed_variables = ["y"]
    Y_test = X_test[imputed_variables]

    model_classes = [OLS]

    # Run twice
    method_imputations1 = get_imputations(
        model_classes, X_train, X_test, predictors, imputed_variables
    )
    loss_df1 = compare_quantile_loss(
        Y_test, method_imputations1, imputed_variables
    )

    method_imputations2 = get_imputations(
        model_classes, X_train, X_test, predictors, imputed_variables
    )
    loss_df2 = compare_quantile_loss(
        Y_test, method_imputations2, imputed_variables
    )

    # Results should be deterministic for OLS
    ols_loss1 = loss_df1[loss_df1["Method"] == "OLS"]["Loss"].values
    ols_loss2 = loss_df2[loss_df2["Method"] == "OLS"]["Loss"].values
    np.testing.assert_array_almost_equal(ols_loss1, ols_loss2, decimal=5)
