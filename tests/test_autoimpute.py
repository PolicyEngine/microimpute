"""Comprehensive tests for the autoimpute functionality."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes

from microimpute.comparisons.autoimpute import autoimpute, AutoImputeResult
from microimpute.visualizations import *

# Check if Matching is available
try:
    from microimpute.models import Matching

    HAS_MATCHING = True
except ImportError:
    HAS_MATCHING = False


# === Fixtures ===


@pytest.fixture
def diabetes_donor() -> pd.DataFrame:
    """Create donor dataset from diabetes data."""
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    # Add boolean variable for testing
    np.random.seed(42)
    df["bool"] = np.random.choice([True, False], size=len(df))
    # Add categorical variable
    df["category"] = np.random.choice(["A", "B", "C"], size=len(df))
    return df


@pytest.fixture
def diabetes_receiver() -> pd.DataFrame:
    """Create receiver dataset from diabetes data."""
    diabetes = load_diabetes()
    return pd.DataFrame(diabetes.data, columns=diabetes.feature_names)


@pytest.fixture
def simple_data() -> tuple:
    """Create simple donor and receiver datasets."""
    np.random.seed(42)
    n_samples = 100

    donor = pd.DataFrame(
        {
            "x1": np.random.randn(n_samples),
            "x2": np.random.randn(n_samples),
            "y1": np.random.randn(n_samples),
            "y2": np.random.randn(n_samples),
        }
    )

    receiver = pd.DataFrame(
        {"x1": np.random.randn(50), "x2": np.random.randn(50)}
    )

    return donor, receiver


# === Basic Functionality Tests ===


def test_autoimpute_basic_structure(
    diabetes_donor: pd.DataFrame, diabetes_receiver: pd.DataFrame
) -> None:
    """Test that autoimpute returns expected data structures."""
    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1", "bool"]

    hyperparams = {"QRF": {"n_estimators": 100}}
    if HAS_MATCHING:
        hyperparams["Matching"] = {"constrained": True}

    results = autoimpute(
        donor_data=diabetes_donor,
        receiver_data=diabetes_receiver,
        predictors=predictors,
        imputed_variables=imputed_variables,
        hyperparameters={
            "QRF": {"n_estimators": 50},
            "Matching": {"constrained": True},
        },
        log_level="WARNING",
    )

    # Check return type
    assert isinstance(results, AutoImputeResult)

    # Check imputations structure
    assert isinstance(results.imputations, dict)
    assert "best_method" in results.imputations
    for model_name, imputations in results.imputations.items():
        assert isinstance(imputations, pd.DataFrame)
        if model_name != "best_method":
            assert all(var in imputations.columns for var in imputed_variables)

    # Check receiver_data structure
    assert isinstance(results.receiver_data, pd.DataFrame)
    assert len(results.receiver_data) == len(diabetes_receiver)
    assert all(
        var in results.receiver_data.columns for var in imputed_variables
    )

    # Check cv_results structure
    assert isinstance(results.cv_results, pd.DataFrame)
    assert "mean_loss" in results.cv_results.columns
    assert 0.05 in results.cv_results.columns  # First quantile
    assert 0.95 in results.cv_results.columns  # Last quantile


def test_autoimpute_all_models(
    diabetes_donor: pd.DataFrame, diabetes_receiver: pd.DataFrame
) -> None:
    """Test autoimpute with all available models."""
    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1"]

    results = autoimpute(
        donor_data=diabetes_donor,
        receiver_data=diabetes_receiver,
        predictors=predictors,
        imputed_variables=imputed_variables,
        models=None,  # Use all available models
        impute_all=True,  # Return results for all models
        log_level="WARNING",
    )

    # Should have results for multiple models
    assert len(results.imputations) > 2  # At least 2 models + best_method

    # Check that different models might produce different results
    model_names = [
        name for name in results.imputations.keys() if name != "best_method"
    ]
    if len(model_names) >= 2:
        model1_imputations = results.imputations[model_names[0]]
        model2_imputations = results.imputations[model_names[1]]
        # Different models should generally produce different imputations
        assert not model1_imputations.equals(model2_imputations)


def test_autoimpute_specific_models(
    diabetes_donor: pd.DataFrame, diabetes_receiver: pd.DataFrame
) -> None:
    """Test autoimpute with specific models only."""
    from microimpute.models import OLS, QRF

    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1"]

    results = autoimpute(
        donor_data=diabetes_donor,
        receiver_data=diabetes_receiver,
        predictors=predictors,
        imputed_variables=imputed_variables,
        models=[OLS, QRF],
        impute_all=True,  # Return results for all models
        log_level="WARNING",
    )

    # Should have best_method and at least one of the specified models
    assert "best_method" in results.imputations
    # At least one of the specified models should be present
    model_names = [
        name for name in results.imputations.keys() if name != "best_method"
    ]
    assert len(model_names) >= 1

    # CV results should have both models
    assert "OLS" in results.cv_results.index
    assert "QRF" in results.cv_results.index


# === Hyperparameter Handling ===


def test_autoimpute_with_hyperparameters(simple_data: tuple) -> None:
    """Test autoimpute with custom hyperparameters."""
    donor, receiver = simple_data

    hyperparameters = {
        "QRF": {"n_estimators": 20, "min_samples_leaf": 10},
        "OLS": {},  # Empty dict for models without hyperparameters
        "Matching": {"k": 3, "dist_fun": "Manhattan"},
    }

    results = autoimpute(
        donor_data=donor,
        receiver_data=receiver,
        predictors=["x1", "x2"],
        imputed_variables=["y1"],
        hyperparameters=hyperparameters,
        log_level="WARNING",
    )

    # Should run without errors
    assert results is not None
    assert "best_method" in results.imputations


# === Edge Cases ===


def test_autoimpute_multiple_imputed_variables(simple_data: tuple) -> None:
    """Test autoimpute with multiple variables to impute."""
    donor, receiver = simple_data

    results = autoimpute(
        donor_data=donor,
        receiver_data=receiver,
        predictors=["x1", "x2"],
        imputed_variables=["y1", "y2"],  # Multiple variables
        log_level="WARNING",
    )

    assert results is not None
    assert all(var in results.receiver_data.columns for var in ["y1", "y2"])
    assert not results.receiver_data[["y1", "y2"]].isna().any().any()


def test_autoimpute_large_receiver() -> None:
    """Test autoimpute with receiver larger than donor."""
    np.random.seed(42)

    donor = pd.DataFrame({"x": np.random.randn(50), "y": np.random.randn(50)})

    receiver = pd.DataFrame({"x": np.random.randn(100)})  # Larger than donor

    results = autoimpute(
        donor_data=donor,
        receiver_data=receiver,
        predictors=["x"],
        imputed_variables=["y"],
        log_level="WARNING",
    )

    assert results is not None
    assert len(results.receiver_data) == 100
    assert not results.receiver_data["y"].isna().any()


# === Best Method Selection ===


def test_autoimpute_best_method_selection(simple_data: tuple) -> None:
    """Test that best method is selected based on CV results."""
    donor, receiver = simple_data

    results = autoimpute(
        donor_data=donor,
        receiver_data=receiver,
        predictors=["x1", "x2"],
        imputed_variables=["y1"],
        log_level="WARNING",
    )

    # Best method should have lowest mean loss
    best_method_name = results.cv_results["mean_loss"].idxmin()

    # Best method imputations should match the best performing model
    if best_method_name in results.imputations:
        best_method_imputations = results.imputations["best_method"]
        specific_model_imputations = results.imputations[best_method_name]

        # They should be the same
        pd.testing.assert_frame_equal(
            best_method_imputations, specific_model_imputations
        )


def test_autoimpute_cv_results_structure(simple_data: tuple) -> None:
    """Test the structure of cross-validation results."""
    donor, receiver = simple_data

    results = autoimpute(
        donor_data=donor,
        receiver_data=receiver,
        predictors=["x1", "x2"],
        imputed_variables=["y1"],
        log_level="WARNING",
    )

    cv_results = results.cv_results

    # Check structure
    assert isinstance(cv_results, pd.DataFrame)
    assert "mean_loss" in cv_results.columns

    # Check quantile columns
    quantile_cols = [
        col for col in cv_results.columns if isinstance(col, float)
    ]
    assert len(quantile_cols) > 0
    assert min(quantile_cols) >= 0.0
    assert max(quantile_cols) <= 1.0

    # Check that all models have results
    assert len(cv_results) > 0
    assert not cv_results["mean_loss"].isna().any()


# === Visualization Compatibility ===


def test_autoimpute_visualization_compatibility(simple_data: tuple) -> None:
    """Test that autoimpute results work with visualization functions."""
    donor, receiver = simple_data

    results = autoimpute(
        donor_data=donor,
        receiver_data=receiver,
        predictors=["x1", "x2"],
        imputed_variables=["y1"],
        log_level="WARNING",
    )

    # Test that visualization can be created
    comparison_viz = method_comparison_results(
        data=results.cv_results,
        metric_name="Test Quantile Loss",
        data_format="wide",
    )

    assert comparison_viz is not None

    # Test that plot can be generated (without saving)
    fig = comparison_viz.plot(
        title="Test Autoimpute Comparison",
        show_mean=True,
        save_path=None,  # Don't save
    )

    assert fig is not None


# === Error Handling ===


def test_autoimpute_missing_predictors() -> None:
    """Test autoimpute with missing predictors in receiver."""
    np.random.seed(42)

    donor = pd.DataFrame(
        {
            "x1": np.random.randn(50),
            "x2": np.random.randn(50),
            "y": np.random.randn(50),
        }
    )

    receiver = pd.DataFrame(
        {
            "x1": np.random.randn(10)
            # x2 is missing
        }
    )

    with pytest.raises(Exception):
        autoimpute(
            donor_data=donor,
            receiver_data=receiver,
            predictors=["x1", "x2"],  # x2 not in receiver
            imputed_variables=["y"],
            log_level="WARNING",
        )


def test_autoimpute_invalid_model_specification() -> None:
    """Test autoimpute with invalid model specification."""
    np.random.seed(42)

    donor = pd.DataFrame({"x": np.random.randn(50), "y": np.random.randn(50)})

    receiver = pd.DataFrame({"x": np.random.randn(10)})

    # Invalid model type
    with pytest.raises(Exception):
        autoimpute(
            donor_data=donor,
            receiver_data=receiver,
            predictors=["x"],
            imputed_variables=["y"],
            models=["InvalidModel"],  # String instead of class
            log_level="WARNING",
        )


# === Performance Tests ===


def test_autoimpute_consistency(simple_data: tuple) -> None:
    """Test that autoimpute produces consistent results."""
    donor, receiver = simple_data

    # Run autoimpute twice with same data
    results1 = autoimpute(
        donor_data=donor,
        receiver_data=receiver,
        predictors=["x1", "x2"],
        imputed_variables=["y1"],
        log_level="WARNING",
    )

    results2 = autoimpute(
        donor_data=donor,
        receiver_data=receiver,
        predictors=["x1", "x2"],
        imputed_variables=["y1"],
        log_level="WARNING",
    )

    # CV results should be very similar (allowing for small numerical differences)
    np.testing.assert_allclose(
        results1.cv_results["mean_loss"].values,
        results2.cv_results["mean_loss"].values,
        rtol=0.01,
    )
