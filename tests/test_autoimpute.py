"""Comprehensive tests for the autoimpute functionality."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes

from microimpute.comparisons.autoimpute import autoimpute, AutoImputeResult
from microimpute.visualizations import *
from microimpute.models import QRF, QuantReg, OLS

# Check if Matching is available
try:
    from microimpute.models import Matching

    HAS_MATCHING = True
except ImportError:
    HAS_MATCHING = False

# Check if MDN is available
try:
    from microimpute.models import MDN

    HAS_MDN = True
except ImportError:
    HAS_MDN = False

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
        models=[QRF, Matching, QuantReg, OLS] if not HAS_MDN else None,
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

    # Check cv_results structure - now a dict with dual metrics
    assert isinstance(results.cv_results, dict)
    assert len(results.cv_results) > 0  # At least one model

    for model_name in results.cv_results:
        model_results = results.cv_results[model_name]
        assert "quantile_loss" in model_results
        assert "log_loss" in model_results
        # Check structure for each metric type
        assert "mean_test" in model_results["quantile_loss"]
        assert "mean_train" in model_results["quantile_loss"]
        assert "variables" in model_results["quantile_loss"]


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
        models=[QRF, Matching, QuantReg, OLS] if not HAS_MDN else None,
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

    # CV results should have both models as dict keys
    assert "OLS" in results.cv_results
    assert "QRF" in results.cv_results


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
        models=[QRF, Matching, QuantReg, OLS] if not HAS_MDN else None,
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
        models=[QRF, Matching, QuantReg, OLS] if not HAS_MDN else None,
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
        models=[QRF, Matching, QuantReg, OLS] if not HAS_MDN else None,
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
        models=[QRF, Matching, QuantReg, OLS] if not HAS_MDN else None,
        log_level="WARNING",
    )

    # Find best method based on metrics
    # Since y1 is numerical, should use quantile_loss
    best_loss = float("inf")
    best_method_name = None
    for model_name, model_results in results.cv_results.items():
        # For numerical variables, check quantile_loss
        test_loss = model_results["quantile_loss"]["mean_test"]
        if not np.isnan(test_loss) and test_loss < best_loss:
            best_loss = test_loss
            best_method_name = model_name

    # Best method imputations should be present
    assert "best_method" in results.imputations
    assert best_method_name is not None

    # Check that best_method key exists in fitted_models
    assert (
        "best_method" in results.fitted_models
    ), "best_method key not found in fitted_models"

    # Get the actual class name of the selected best method
    best_method_instance = results.fitted_models["best_method"]
    # The instance is an ImputerResults object, get its parent model class name
    actual_best_model_name = best_method_instance.__class__.__name__.replace(
        "Results", ""
    )

    # Verify that autoimpute selected the model with the lowest loss
    assert (
        actual_best_model_name == best_method_name
    ), f"Expected {best_method_name} to be selected as best, but got {actual_best_model_name}"

    # Additionally verify the loss values are consistent
    all_losses = []
    for model_name, model_results in results.cv_results.items():
        test_loss = model_results["quantile_loss"]["mean_test"]
        if not np.isnan(test_loss):
            all_losses.append(test_loss)

    # The best method we found should have the minimum loss
    if all_losses:
        assert (
            abs(best_loss - min(all_losses)) < 1e-6
        ), f"Best loss {best_loss} doesn't match minimum loss {min(all_losses)}"


def test_autoimpute_cv_results_structure(simple_data: tuple) -> None:
    """Test the structure of cross-validation results."""
    donor, receiver = simple_data

    results = autoimpute(
        donor_data=donor,
        receiver_data=receiver,
        predictors=["x1", "x2"],
        imputed_variables=["y1"],
        models=[QRF, Matching, QuantReg, OLS] if not HAS_MDN else None,
        log_level="WARNING",
    )

    cv_results = results.cv_results

    # Check structure - now a dict with dual metrics
    assert isinstance(cv_results, dict)

    # Check each model's results
    for model_name, model_results in cv_results.items():
        assert "quantile_loss" in model_results
        assert "log_loss" in model_results

        # Check quantile_loss structure
        ql_results = model_results["quantile_loss"]
        assert not np.isnan(ql_results["mean_test"])
        assert not np.isnan(ql_results["mean_train"])
        assert "variables" in ql_results
        assert isinstance(ql_results["results"], pd.DataFrame)
        assert "train" in ql_results["results"].index
        assert "test" in ql_results["results"].index


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
            models=[QRF, Matching, QuantReg, OLS] if not HAS_MDN else None,
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
        models=[QRF, Matching, QuantReg, OLS] if not HAS_MDN else None,
        log_level="WARNING",
    )

    results2 = autoimpute(
        donor_data=donor,
        receiver_data=receiver,
        predictors=["x1", "x2"],
        imputed_variables=["y1"],
        models=[QRF, Matching, QuantReg, OLS] if not HAS_MDN else None,
        log_level="WARNING",
    )

    # CV results should be very similar (allowing for small numerical differences)
    # Compare quantile_loss mean_test values for each model
    for model_name in results1.cv_results:
        if model_name in results2.cv_results:
            loss1 = results1.cv_results[model_name]["quantile_loss"][
                "mean_test"
            ]
            loss2 = results2.cv_results[model_name]["quantile_loss"][
                "mean_test"
            ]
            if not np.isnan(loss1) and not np.isnan(loss2):
                np.testing.assert_allclose(loss1, loss2, rtol=0.05)
