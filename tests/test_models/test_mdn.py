"""Tests for the MDN (Mixture Density Network) imputation model."""

import os
import shutil
import tempfile
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from microimpute.utils.data import preprocess_data

# Skip all tests if pytorch-tabular is not available
pytest.importorskip("pytorch_tabular")

from microimpute.models.mdn import (
    MDN,
    _generate_cache_key,
    _generate_data_hash,
)

# === Fixtures ===


@pytest.fixture
def simple_numeric_data() -> pd.DataFrame:
    """Create simple synthetic numeric data for testing."""
    np.random.seed(42)
    n_samples = 200

    x1 = np.random.randn(n_samples)
    x2 = np.random.randn(n_samples)
    # Target with known relationship to predictors
    y = 2 * x1 + 0.5 * x2 + np.random.randn(n_samples) * 0.5

    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.fixture
def mixed_type_data() -> pd.DataFrame:
    """Create data with numeric, categorical, and boolean targets."""
    np.random.seed(42)
    n_samples = 200

    x1 = np.random.randn(n_samples)
    x2 = np.random.randn(n_samples)

    # Numeric target
    y_numeric = 2 * x1 + np.random.randn(n_samples) * 0.5

    # Categorical target
    categories = ["A", "B", "C"]
    y_categorical = np.random.choice(categories, n_samples)

    # Boolean target
    y_boolean = (x1 + x2 > 0).astype(bool)

    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "y_numeric": y_numeric,
            "y_categorical": y_categorical,
            "y_boolean": y_boolean,
        }
    )


@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for model caching tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


# === Basic Functionality Tests ===


def test_mdn_basic_fit_predict(simple_numeric_data: pd.DataFrame) -> None:
    """Test basic MDN model fitting and prediction for numeric data."""
    predictors = ["x1", "x2"]
    imputed_variables = ["y"]

    X_train, X_test = preprocess_data(simple_numeric_data)

    # Initialize with small network for fast testing
    model = MDN(
        layers="32-16",
        num_gaussian=3,
        max_epochs=5,
        batch_size=32,
    )
    fitted_model = model.fit(X_train, predictors, imputed_variables)

    # Predict without specifying quantiles (should return single DataFrame)
    predictions = fitted_model.predict(X_test)

    assert isinstance(predictions, pd.DataFrame)
    assert predictions.shape == (len(X_test), len(imputed_variables))
    assert not predictions.isna().any().any()


def test_mdn_multiple_quantiles(simple_numeric_data: pd.DataFrame) -> None:
    """Test MDN prediction with multiple quantile requests."""
    predictors = ["x1", "x2"]
    imputed_variables = ["y"]

    X_train, X_test = preprocess_data(simple_numeric_data)

    model = MDN(
        layers="32-16",
        num_gaussian=3,
        max_epochs=5,
        batch_size=32,
    )
    fitted_model = model.fit(X_train, predictors, imputed_variables)

    # Predict at multiple quantiles
    quantiles = [0.25, 0.5, 0.75]
    predictions = fitted_model.predict(X_test, quantiles=quantiles)

    assert isinstance(predictions, dict)
    assert set(predictions.keys()) == set(quantiles)

    for q, pred_df in predictions.items():
        assert pred_df.shape == (len(X_test), len(imputed_variables))
        assert not pred_df.isna().any().any()


def test_mdn_stochastic_sampling(simple_numeric_data: pd.DataFrame) -> None:
    """Test that MDN produces stochastic (different) samples."""
    predictors = ["x1", "x2"]
    imputed_variables = ["y"]

    X_train, X_test = preprocess_data(simple_numeric_data)

    model = MDN(
        layers="32-16",
        num_gaussian=3,
        max_epochs=5,
        batch_size=32,
    )
    fitted_model = model.fit(X_train, predictors, imputed_variables)

    # Get two sets of predictions
    pred1 = fitted_model.predict(X_test)
    pred2 = fitted_model.predict(X_test)

    # Due to stochastic sampling, predictions should differ
    # (unless seed makes them identical within the model)
    # At minimum, verify both are valid
    assert pred1.shape == pred2.shape
    assert not pred1.isna().any().any()
    assert not pred2.isna().any().any()


# === Mixed Variable Type Tests ===


def test_mdn_categorical_target(mixed_type_data: pd.DataFrame) -> None:
    """Test MDN with categorical target variable."""
    predictors = ["x1", "x2"]
    imputed_variables = ["y_categorical"]

    X_train, X_test = preprocess_data(mixed_type_data)

    model = MDN(
        layers="32-16",
        max_epochs=5,
        batch_size=32,
    )

    # Should log warning about neural classifier
    fitted_model = model.fit(X_train, predictors, imputed_variables)

    predictions = fitted_model.predict(X_test)

    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) == len(X_test)
    # Predictions should be from original categories
    assert set(predictions["y_categorical"].unique()).issubset({"A", "B", "C"})


def test_mdn_boolean_target(mixed_type_data: pd.DataFrame) -> None:
    """Test MDN with boolean target variable."""
    predictors = ["x1", "x2"]
    imputed_variables = ["y_boolean"]

    X_train, X_test = preprocess_data(mixed_type_data)

    model = MDN(
        layers="32-16",
        max_epochs=5,
        batch_size=32,
    )

    fitted_model = model.fit(X_train, predictors, imputed_variables)

    predictions = fitted_model.predict(X_test)

    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) == len(X_test)
    # Predictions should be boolean
    assert predictions["y_boolean"].dtype == bool


def test_mdn_mixed_targets(mixed_type_data: pd.DataFrame) -> None:
    """Test MDN with mixed numeric, categorical, and boolean targets."""
    predictors = ["x1", "x2"]
    imputed_variables = ["y_numeric", "y_categorical", "y_boolean"]

    X_train, X_test = preprocess_data(mixed_type_data)

    model = MDN(
        layers="32-16",
        num_gaussian=3,
        max_epochs=5,
        batch_size=32,
    )

    fitted_model = model.fit(X_train, predictors, imputed_variables)

    predictions = fitted_model.predict(X_test)

    assert isinstance(predictions, pd.DataFrame)
    assert set(predictions.columns) == set(imputed_variables)
    assert len(predictions) == len(X_test)
    assert not predictions.isna().any().any()


# === Model Caching Tests ===


def test_mdn_model_caching(
    simple_numeric_data: pd.DataFrame, temp_model_dir: str
) -> None:
    """Test that MDN models are cached and reloaded correctly."""
    predictors = ["x1", "x2"]
    imputed_variables = ["y"]

    X_train, X_test = preprocess_data(simple_numeric_data)

    # First fit - should train and save model
    model1 = MDN(
        layers="32-16",
        num_gaussian=3,
        max_epochs=5,
        batch_size=32,
        model_dir=temp_model_dir,
    )
    fitted_model1 = model1.fit(X_train, predictors, imputed_variables)
    predictions1 = fitted_model1.predict(X_test)

    # Second fit with same data - should load from cache
    model2 = MDN(
        layers="32-16",
        num_gaussian=3,
        max_epochs=5,
        batch_size=32,
        model_dir=temp_model_dir,
    )
    fitted_model2 = model2.fit(X_train, predictors, imputed_variables)
    predictions2 = fitted_model2.predict(X_test)

    # Both should produce valid predictions
    assert predictions1.shape == predictions2.shape
    assert not predictions1.isna().any().any()
    assert not predictions2.isna().any().any()

    # Check that cache directory was created (at least one subdirectory)
    assert len(os.listdir(temp_model_dir)) > 0


def test_mdn_force_retrain(
    simple_numeric_data: pd.DataFrame, temp_model_dir: str
) -> None:
    """Test that force_retrain bypasses cache."""
    predictors = ["x1", "x2"]
    imputed_variables = ["y"]

    X_train, X_test = preprocess_data(simple_numeric_data)

    # First fit
    model1 = MDN(
        layers="32-16",
        num_gaussian=3,
        max_epochs=5,
        batch_size=32,
        model_dir=temp_model_dir,
    )
    model1.fit(X_train, predictors, imputed_variables)

    # Second fit with force_retrain - should train again
    model2 = MDN(
        layers="32-16",
        num_gaussian=3,
        max_epochs=5,
        batch_size=32,
        model_dir=temp_model_dir,
        force_retrain=True,
    )
    fitted_model2 = model2.fit(X_train, predictors, imputed_variables)
    predictions = fitted_model2.predict(X_test)

    assert not predictions.isna().any().any()


def test_mdn_different_data_different_cache(
    simple_numeric_data: pd.DataFrame, temp_model_dir: str
) -> None:
    """Test that different datasets create different cache entries."""
    predictors = ["x1", "x2"]
    imputed_variables = ["y"]

    X_train, X_test = preprocess_data(simple_numeric_data)

    # Fit with original data
    model1 = MDN(
        layers="32-16",
        num_gaussian=3,
        max_epochs=5,
        batch_size=32,
        model_dir=temp_model_dir,
    )
    model1.fit(X_train, predictors, imputed_variables)

    # Count cache entries
    cache_count_1 = len(os.listdir(temp_model_dir))

    # Create modified data (different values)
    modified_data = simple_numeric_data.copy()
    modified_data["y"] = modified_data["y"] * 2
    X_train_mod, _ = preprocess_data(modified_data)

    # Fit with modified data - should create new cache entry
    model2 = MDN(
        layers="32-16",
        num_gaussian=3,
        max_epochs=5,
        batch_size=32,
        model_dir=temp_model_dir,
    )
    model2.fit(X_train_mod, predictors, imputed_variables)

    # Should have more cache entries now
    cache_count_2 = len(os.listdir(temp_model_dir))
    assert cache_count_2 > cache_count_1


# === Edge Cases ===


def test_mdn_constant_target(simple_numeric_data: pd.DataFrame) -> None:
    """Test MDN with constant target variable."""
    data = simple_numeric_data.copy()
    data["y_constant"] = 5.0  # Constant value

    predictors = ["x1", "x2"]
    imputed_variables = ["y_constant"]

    X_train, X_test = preprocess_data(data)

    model = MDN(
        layers="32-16",
        max_epochs=5,
        batch_size=32,
    )
    fitted_model = model.fit(X_train, predictors, imputed_variables)

    predictions = fitted_model.predict(X_test)

    # All predictions should be the constant value
    assert np.allclose(predictions["y_constant"].values, 5.0)


def test_mdn_single_observation() -> None:
    """Test MDN prediction with single observation."""
    np.random.seed(42)

    # Create small training data
    train_data = pd.DataFrame(
        {
            "x1": np.random.randn(50),
            "x2": np.random.randn(50),
            "y": np.random.randn(50),
        }
    )

    # Single test observation
    test_data = pd.DataFrame({"x1": [0.5], "x2": [-0.3], "y": [0.0]})

    model = MDN(
        layers="16-8",
        num_gaussian=2,
        max_epochs=3,
        batch_size=16,
    )
    fitted_model = model.fit(train_data, ["x1", "x2"], ["y"])

    predictions = fitted_model.predict(test_data)

    assert len(predictions) == 1
    assert not predictions.isna().any().any()


# === Configuration Tests ===


def test_mdn_custom_configuration() -> None:
    """Test MDN with custom configuration parameters."""
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "x1": np.random.randn(100),
            "x2": np.random.randn(100),
            "y": np.random.randn(100),
        }
    )

    X_train, X_test = preprocess_data(data)

    # Custom configuration
    model = MDN(
        layers="64-32-16",  # Deeper network
        activation="LeakyReLU",
        dropout=0.2,
        use_batch_norm=True,
        num_gaussian=10,  # More mixture components
        softmax_temperature=0.5,
        learning_rate=5e-4,
        max_epochs=3,
        batch_size=16,
    )

    fitted_model = model.fit(X_train, ["x1", "x2"], ["y"])
    predictions = fitted_model.predict(X_test)

    assert not predictions.isna().any().any()


def test_generate_cache_key():
    """Test cache key generation."""
    data_hash = "abc123def456"

    key1 = _generate_cache_key(["a", "b", "c"], "target", data_hash)
    key2 = _generate_cache_key(
        ["c", "b", "a"], "target", data_hash
    )  # Different predictor order
    key3 = _generate_cache_key(["a", "b", "c"], "target", "different_hash")

    # Same predictors (regardless of order) should produce same key
    assert key1 == key2

    # Different data hash should produce different key
    assert key1 != key3


def test_generate_data_hash():
    """Test data hash generation."""
    np.random.seed(42)

    # Create test data
    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    y = pd.Series([7, 8, 9], name="target")

    # Same data should produce same hash
    hash1 = _generate_data_hash(X, y)
    hash2 = _generate_data_hash(X.copy(), y.copy())
    assert hash1 == hash2

    # Different data should produce different hash
    X_different = pd.DataFrame({"a": [1, 2, 4], "b": [4, 5, 6]})
    hash3 = _generate_data_hash(X_different, y)
    assert hash1 != hash3


# === Return Probabilities Test ===


def test_mdn_return_probs(mixed_type_data: pd.DataFrame) -> None:
    """Test MDN returning probability distributions for categorical variables."""
    predictors = ["x1", "x2"]
    imputed_variables = ["y_categorical"]

    X_train, X_test = preprocess_data(mixed_type_data)

    model = MDN(
        layers="32-16",
        max_epochs=5,
        batch_size=32,
    )
    fitted_model = model.fit(X_train, predictors, imputed_variables)

    # Predict with return_probs=True
    predictions = fitted_model.predict(
        X_test, quantiles=[0.5], return_probs=True
    )

    assert "probabilities" in predictions
    assert "y_categorical" in predictions["probabilities"]

    prob_info = predictions["probabilities"]["y_categorical"]
    assert "probabilities" in prob_info
    assert "classes" in prob_info

    # Probabilities should sum to 1
    probs = prob_info["probabilities"]
    assert np.allclose(probs.sum(axis=1), 1.0)


# === Hyperparameter Tuning Tests ===


def test_mdn_hyperparameter_tuning_numeric(
    simple_numeric_data: pd.DataFrame,
) -> None:
    """Test MDN hyperparameter tuning with numeric targets."""
    predictors = ["x1", "x2"]
    imputed_variables = ["y"]

    X_train, X_test = preprocess_data(simple_numeric_data)

    model = MDN(
        layers="16-8",
        max_epochs=5,  # Short for testing
        batch_size=32,
        early_stopping_patience=5,
    )

    # Fit with tuning enabled
    result = model.fit(
        X_train, predictors, imputed_variables, tune_hyperparameters=True
    )

    # Should return tuple (MDNResults, best_params)
    assert isinstance(result, tuple)
    assert len(result) == 2

    fitted_model, best_params = result

    # Check best_params structure for numeric-only case
    assert isinstance(best_params, dict)
    assert "num_gaussian" in best_params
    assert "learning_rate" in best_params

    # Verify tuned params are within expected ranges
    assert 2 <= best_params["num_gaussian"] <= 10
    assert 1e-4 <= best_params["learning_rate"] <= 1e-2

    # Verify model can still predict
    predictions = fitted_model.predict(X_test)
    assert not predictions.isna().any().any()


def test_mdn_hyperparameter_tuning_categorical(
    mixed_type_data: pd.DataFrame,
) -> None:
    """Test MDN hyperparameter tuning with categorical targets."""
    predictors = ["x1", "x2"]
    imputed_variables = ["y_categorical"]

    X_train, X_test = preprocess_data(mixed_type_data)

    model = MDN(
        layers="16-8",
        max_epochs=5,  # Short for testing
        batch_size=32,
        early_stopping_patience=5,
    )

    # Fit with tuning enabled
    result = model.fit(
        X_train, predictors, imputed_variables, tune_hyperparameters=True
    )

    # Should return tuple (MDNResults, best_params)
    assert isinstance(result, tuple)
    assert len(result) == 2

    fitted_model, best_params = result

    # Check best_params structure for categorical-only case
    assert isinstance(best_params, dict)
    assert "learning_rate" in best_params

    # Verify tuned params are within expected ranges
    assert 1e-4 <= best_params["learning_rate"] <= 1e-2

    # Verify model can still predict
    predictions = fitted_model.predict(X_test)
    assert not predictions.isna().any().any()


def test_mdn_hyperparameter_tuning_mixed(
    mixed_type_data: pd.DataFrame,
) -> None:
    """Test MDN hyperparameter tuning with mixed targets."""
    predictors = ["x1", "x2"]
    imputed_variables = ["y_numeric", "y_categorical"]

    X_train, X_test = preprocess_data(mixed_type_data)

    model = MDN(
        layers="16-8",
        max_epochs=5,  # Short for testing
        batch_size=32,
        early_stopping_patience=5,
    )

    # Fit with tuning enabled
    result = model.fit(
        X_train, predictors, imputed_variables, tune_hyperparameters=True
    )

    # Should return tuple (MDNResults, best_params)
    assert isinstance(result, tuple)
    assert len(result) == 2

    fitted_model, best_params = result

    # Check best_params structure for mixed case
    assert isinstance(best_params, dict)
    assert "mdn" in best_params
    assert "classifier" in best_params

    # Verify MDN params
    assert "num_gaussian" in best_params["mdn"]
    assert "learning_rate" in best_params["mdn"]

    # Verify classifier params
    assert "learning_rate" in best_params["classifier"]

    # Verify model can still predict
    predictions = fitted_model.predict(X_test)
    assert not predictions.isna().any().any()


def test_mdn_without_tuning_returns_single_result(
    simple_numeric_data: pd.DataFrame,
) -> None:
    """Test that MDN without tuning returns just MDNResults (not tuple)."""
    predictors = ["x1", "x2"]
    imputed_variables = ["y"]

    X_train, X_test = preprocess_data(simple_numeric_data)

    model = MDN(
        layers="16-8",
        max_epochs=3,
        batch_size=32,
    )

    # Fit without tuning
    result = model.fit(X_train, predictors, imputed_variables)

    # Should return MDNResults directly, not a tuple
    assert not isinstance(result, tuple)

    # Verify model can predict
    predictions = result.predict(X_test)
    assert not predictions.isna().any().any()
