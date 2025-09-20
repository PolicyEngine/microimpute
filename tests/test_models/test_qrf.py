"""Comprehensive tests for the Quantile Regression Forest imputation model.

This file combines and consolidates tests from test_qrf.py and test_qrf_extended.py,
removing duplicates and ensuring comprehensive coverage.
"""

import io
import logging
import re
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error

from microimpute.config import QUANTILES
from microimpute.evaluations import *
from microimpute.models.qrf import QRF, _QRFModel
from microimpute.utils.data import preprocess_data
from microimpute.visualizations import *


# === Fixtures and Test Data ===


@pytest.fixture
def diabetes_data() -> pd.DataFrame:
    """Load and prepare diabetes dataset for testing."""
    diabetes = load_diabetes()
    return pd.DataFrame(diabetes.data, columns=diabetes.feature_names)


@pytest.fixture
def simple_data() -> pd.DataFrame:
    """Create simple synthetic data for testing basic functionality."""
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame(
        {
            "x1": np.random.randn(n_samples),
            "x2": np.random.randn(n_samples),
            "y": np.random.randn(n_samples),
        }
    )


@pytest.fixture
def categorical_data() -> pd.DataFrame:
    """Create data with categorical variables for testing encoding."""
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame(
        {
            "numeric1": np.random.randn(n_samples),
            "numeric2": np.random.randn(n_samples),
            "category": np.random.choice(["A", "B", "C"], n_samples),
            "target": np.random.randn(n_samples),
        }
    )


# === Basic Functionality Tests ===


def test_qrf_basic_fit_predict(diabetes_data: pd.DataFrame) -> None:
    """Test basic QRF model fitting and prediction."""
    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1", "s4"]
    data = diabetes_data[predictors + imputed_variables]

    X_train, X_test = preprocess_data(data)

    # Initialize and fit model
    model = QRF()
    fitted_model = model.fit(
        X_train,
        predictors,
        imputed_variables,
        n_estimators=50,
        min_samples_leaf=5,
    )

    # Predict at multiple quantiles
    predictions = fitted_model.predict(X_test, quantiles=QUANTILES)

    # Validate predictions
    assert isinstance(predictions, dict)
    assert set(predictions.keys()) == set(QUANTILES)

    for q, pred_df in predictions.items():
        assert pred_df.shape == (len(X_test), len(imputed_variables))
        assert not pred_df.isna().any().any()

    # Test default quantiles - returns DataFrame directly for single quantile
    default_predictions = fitted_model.predict(X_test)
    assert isinstance(default_predictions, pd.DataFrame)


def test_qrf_sequential_imputation(diabetes_data: pd.DataFrame) -> None:
    """Test sequential imputation where later variables use earlier ones as predictors."""
    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1", "s2", "s3"]
    data = diabetes_data[predictors + imputed_variables]

    X_train, X_test = preprocess_data(data)

    # Fit model with sequential imputation
    model = QRF()
    fitted_model = model.fit(
        X_train, predictors, imputed_variables, n_estimators=30
    )

    # Get predictions
    small_test = X_test.head(5).copy()
    sequential_preds = fitted_model.predict(small_test, quantiles=[0.5])[0.5]

    # Compare with parallel imputation (each variable independently)
    parallel_predictions = {}
    for var in imputed_variables:
        single_model = QRF()
        single_fitted = single_model.fit(
            X_train, predictors, [var], n_estimators=30
        )
        single_pred = single_fitted.predict(small_test, quantiles=[0.5])
        parallel_predictions[var] = single_pred[0.5][var]

    # Sequential should differ from parallel for later variables
    differences_found = False
    for var in imputed_variables[1:]:
        seq_values = sequential_preds[var].values
        par_values = parallel_predictions[var].values
        if not np.allclose(seq_values, par_values, rtol=1e-5):
            differences_found = True
            break

    assert (
        differences_found
    ), "Sequential imputation should differ from parallel"

    # Test that order matters
    reversed_model = QRF()
    reversed_fitted = reversed_model.fit(
        X_train, predictors, imputed_variables[::-1], n_estimators=30
    )
    reversed_preds = reversed_fitted.predict(small_test, quantiles=[0.5])[0.5]

    # Middle variable should differ when imputed in different orders
    assert not np.allclose(
        sequential_preds["s2"].values, reversed_preds["s2"].values, rtol=1e-5
    ), "Imputation order should affect results"


def test_qrf_beta_distribution_sampling():
    """Test different mean_quantile values for beta distribution sampling."""
    np.random.seed(42)

    # Create simple dataset
    data = pd.DataFrame(
        {
            "x": np.random.randn(200),
            "y": np.random.randn(200),
        }
    )

    train_data = data[:150]
    test_data = data[150:]

    model = QRF()
    fitted_model = model.fit(
        train_data,
        predictors=["x"],
        imputed_variables=["y"],
        n_estimators=50,
    )

    # Test extreme quantiles
    extreme_low = fitted_model.predict(test_data[["x"]], quantiles=[0.01])
    extreme_high = fitted_model.predict(test_data[["x"]], quantiles=[0.99])
    median = fitted_model.predict(test_data[["x"]], quantiles=[0.5])

    # Verify ordering
    for i in range(len(test_data)):
        assert extreme_low[0.01]["y"].iloc[i] <= median[0.5]["y"].iloc[i]
        assert median[0.5]["y"].iloc[i] <= extreme_high[0.99]["y"].iloc[i]


# === Categorical Variable Tests ===


def test_qrf_missing_categorical_levels_in_test(
    categorical_data: pd.DataFrame,
) -> None:
    """Test handling of missing categorical levels in test data."""
    # Training has A, B, C
    train_data = categorical_data.copy()

    # Test only has A, B (missing C)
    test_data = pd.DataFrame(
        {
            "numeric1": np.random.randn(20),
            "numeric2": np.random.randn(20),
            "category": np.random.choice(["A", "B"], 20),
            "target": np.nan,
        }
    )

    model = QRF()
    fitted_model = model.fit(
        train_data,
        predictors=["numeric1", "numeric2", "category"],
        imputed_variables=["target"],
        n_estimators=20,
    )

    # Should handle missing category gracefully
    predictions = fitted_model.predict(
        test_data[["numeric1", "numeric2", "category"]]
    )
    # Default returns DataFrame directly
    assert not predictions["target"].isna().any()


# === Hyperparameter Tuning Tests ===


def test_qrf_hyperparameter_tuning(diabetes_data: pd.DataFrame) -> None:
    """Test hyperparameter tuning functionality."""
    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1", "s4"]
    data = diabetes_data[predictors + imputed_variables]

    # Split data
    np.random.seed(42)
    train_idx = np.random.choice(
        len(data), int(0.7 * len(data)), replace=False
    )
    valid_idx = np.array([i for i in range(len(data)) if i not in train_idx])

    train_data = data.iloc[train_idx].reset_index(drop=True)
    valid_data = data.iloc[valid_idx].reset_index(drop=True)

    X_train = preprocess_data(
        train_data, full_data=True, train_size=1.0, test_size=0.0
    )
    X_valid = preprocess_data(
        valid_data, full_data=True, train_size=1.0, test_size=0.0
    )

    # Fit models with and without tuning
    default_model = QRF()
    default_fitted = default_model.fit(X_train, predictors, imputed_variables)

    tuned_model = QRF()
    tuned_fitted, best_params = tuned_model.fit(
        X_train, predictors, imputed_variables, tune_hyperparameters=True
    )

    # Compare predictions
    default_preds = default_fitted.predict(X_valid, quantiles=[0.5])
    tuned_preds = tuned_fitted.predict(X_valid, quantiles=[0.5])

    # Calculate MSE
    default_mse = {}
    tuned_mse = {}

    for var in imputed_variables:
        default_mse[var] = mean_squared_error(
            X_valid[var], default_preds[0.5][var]
        )
        tuned_mse[var] = mean_squared_error(
            X_valid[var], tuned_preds[0.5][var]
        )

    # Verify hyperparameters are reasonable
    for var in imputed_variables:
        model = tuned_fitted.models[var]
        if hasattr(model, "rf"):
            if hasattr(model.rf, "n_estimators"):
                assert 50 <= model.rf.n_estimators <= 300
            if hasattr(model.rf, "min_samples_leaf"):
                assert 1 <= model.rf.min_samples_leaf <= 10


# === Memory Management and Performance Tests ===


def test_qrf_memory_efficient_mode() -> None:
    """Test memory-efficient mode with cleanup intervals."""
    np.random.seed(42)
    n_samples = 30

    # Create data with many variables
    data_dict = {
        "predictor1": np.random.randn(n_samples),
        "predictor2": np.random.randn(n_samples),
    }

    # Add multiple target variables
    for i in range(15):
        data_dict[f"target{i}"] = np.random.randn(n_samples)

    data = pd.DataFrame(data_dict)

    # Set up logging
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.INFO)

    # Initialize with memory-efficient mode
    model = QRF(log_level="INFO", memory_efficient=True, cleanup_interval=3)
    model.logger.addHandler(handler)

    # Fit model
    fitted_model = model.fit(
        data,
        predictors=["predictor1", "predictor2"],
        imputed_variables=[f"target{i}" for i in range(15)],
        n_estimators=5,
    )

    log_output = log_stream.getvalue()

    # Verify memory management features
    assert model.cleanup_interval == 3
    assert model.memory_efficient == True
    assert "Final memory usage:" in log_output

    model.logger.removeHandler(handler)


def test_qrf_batch_processing() -> None:
    """Test batch processing functionality."""
    np.random.seed(42)
    n_samples = 30

    data_dict = {
        "predictor1": np.random.randn(n_samples),
        "predictor2": np.random.randn(n_samples),
    }

    for i in range(10):
        data_dict[f"target{i}"] = np.random.randn(n_samples)

    data = pd.DataFrame(data_dict)

    # Set up logging
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.INFO)

    # Initialize with batch processing
    model = QRF(
        log_level="INFO",
        memory_efficient=True,
        batch_size=3,
        cleanup_interval=2,
    )
    model.logger.addHandler(handler)

    # Fit model
    fitted_model = model.fit(
        data,
        predictors=["predictor1", "predictor2"],
        imputed_variables=[f"target{i}" for i in range(10)],
        n_estimators=5,
    )

    log_output = log_stream.getvalue()

    # Verify batch processing
    assert model.batch_size == 3
    assert "Processing 10 variables in batches of 3" in log_output
    assert "Processing batch" in log_output

    # Test predictions
    test_data = data[["predictor1", "predictor2"]].head(5)
    predictions = fitted_model.predict(test_data)

    # Default returns DataFrame directly
    assert not predictions.isna().any().any()

    model.logger.removeHandler(handler)


# === Logging and Progress Tracking Tests ===


def test_qrf_detailed_logging() -> None:
    """Test detailed progress logging functionality."""
    np.random.seed(42)
    n_samples = 50

    data = pd.DataFrame(
        {
            "predictor1": np.random.randn(n_samples),
            "predictor2": np.random.randn(n_samples),
            "target1": np.random.randn(n_samples),
            "target2": np.random.randn(n_samples),
            "target3": np.random.randn(n_samples),
        }
    )

    # Set up logging
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.INFO)

    model = QRF(log_level="INFO")
    model.logger.addHandler(handler)

    # Fit model
    fitted_model = model.fit(
        data,
        predictors=["predictor1", "predictor2"],
        imputed_variables=["target1", "target2", "target3"],
        n_estimators=10,
    )

    log_output = log_stream.getvalue()

    # Verify logging messages
    assert "Training data shape:" in log_output
    assert "Memory usage:" in log_output
    assert "Starting imputation for 'target1'" in log_output
    assert "Starting imputation for 'target2'" in log_output
    assert "Starting imputation for 'target3'" in log_output
    assert "Features:" in log_output
    assert "Success:" in log_output
    assert "fitted in" in log_output
    assert "Model complexity:" in log_output
    assert "QRF model fitting completed" in log_output

    # Test prediction logging
    log_stream.truncate(0)
    log_stream.seek(0)

    test_data = data[["predictor1", "predictor2"]].head(10)
    predictions = fitted_model.predict(test_data)

    prediction_logs = log_stream.getvalue()
    assert "Predicting for 'target1'" in prediction_logs
    assert "predicted in" in prediction_logs

    model.logger.removeHandler(handler)


def test_qrf_sequential_imputation_logging() -> None:
    """Test sequential imputation logging shows correct progression."""
    np.random.seed(42)
    n_samples = 40

    data = pd.DataFrame(
        {
            "x1": np.random.randn(n_samples),
            "x2": np.random.randn(n_samples),
            "y1": np.random.randn(n_samples),
            "y2": np.random.randn(n_samples),
            "y3": np.random.randn(n_samples),
        }
    )

    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.INFO)

    model = QRF(log_level="INFO")
    model.logger.addHandler(handler)

    fitted_model = model.fit(
        data,
        predictors=["x1", "x2"],
        imputed_variables=["y1", "y2", "y3"],
        n_estimators=8,
    )

    log_output = log_stream.getvalue()

    # Verify sequential progression
    assert "[1/3] Starting imputation for 'y1'" in log_output
    assert "[2/3] Starting imputation for 'y2'" in log_output
    assert "[3/3] Starting imputation for 'y3'" in log_output

    # Verify feature counts increase
    lines = log_output.split("\n")
    feature_lines = [
        line for line in lines if "Features:" in line and "predictors" in line
    ]
    assert len(feature_lines) == 3

    # Extract and verify feature counts
    feature_counts = []
    for line in feature_lines:
        match = re.search(r"Features: (\d+) predictors", line)
        if match:
            feature_counts.append(int(match.group(1)))

    assert feature_counts == [2, 3, 4]  # Sequential addition of predictors

    model.logger.removeHandler(handler)


# === Error Handling and Validation Tests ===


def test_qrf_missing_variables_handling() -> None:
    """Test handling of missing variables in imputation."""
    np.random.seed(42)
    n_samples = 50

    data = pd.DataFrame(
        {
            "x1": np.random.randn(n_samples),
            "x2": np.random.randn(n_samples),
            "existing_var": np.random.randn(n_samples),
        }
    )

    # Test with skip_missing=False (should raise error)
    model_strict = QRF(log_level="WARNING")

    with pytest.raises(ValueError) as excinfo:
        model_strict.fit(
            data,
            predictors=["x1", "x2"],
            imputed_variables=["existing_var", "missing_var1", "missing_var2"],
            skip_missing=False,
            n_estimators=5,
        )

    error_str = str(excinfo.value)
    assert "missing_var1" in error_str
    assert "missing_var2" in error_str

    # Test with skip_missing=True (should work)
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.WARNING)

    model_lenient = QRF(log_level="WARNING")
    model_lenient.logger.addHandler(handler)

    fitted_model = model_lenient.fit(
        data,
        predictors=["x1", "x2"],
        imputed_variables=["existing_var", "missing_var1", "missing_var2"],
        skip_missing=True,
        n_estimators=5,
    )

    log_output = log_stream.getvalue()
    assert "Variables not found in X_train" in log_output

    # Check only existing variable was included
    assert len(fitted_model.imputed_variables) == 1
    assert "existing_var" in fitted_model.imputed_variables

    model_lenient.logger.removeHandler(handler)


def test_qrf_all_variables_missing() -> None:
    """Test behavior when all imputed variables are missing."""
    np.random.seed(42)
    n_samples = 30

    data = pd.DataFrame(
        {"x1": np.random.randn(n_samples), "x2": np.random.randn(n_samples)}
    )

    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.WARNING)

    model = QRF(log_level="WARNING")
    model.logger.addHandler(handler)

    fitted_model = model.fit(
        data,
        predictors=["x1", "x2"],
        imputed_variables=["missing_var1", "missing_var2"],
        skip_missing=True,
        n_estimators=5,
    )

    log_output = log_stream.getvalue()
    assert "Variables not found in X_train" in log_output
    assert "Skipping missing variables" in log_output

    # Check that no variables were included
    assert len(fitted_model.imputed_variables) == 0
    assert fitted_model.models == {}

    # Test prediction with empty model
    test_data = data[["x1", "x2"]].head(5)
    predictions = fitted_model.predict(test_data)

    # Default returns DataFrame directly
    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions.columns) == 0

    model.logger.removeHandler(handler)


def test_qrf_error_handling() -> None:
    """Test error handling in QRF model."""
    # Test with empty data
    with pytest.raises(Exception):
        model = QRF()
        model.fit(pd.DataFrame(), predictors=[], imputed_variables=[])

    # Test with mismatched predictors
    data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    model = QRF()
    fitted_model = model.fit(data, predictors=["x"], imputed_variables=["y"])

    # Try to predict with missing predictor
    test_data = pd.DataFrame({"z": [7, 8, 9]})

    try:
        predictions = fitted_model.predict(test_data)
    except Exception as e:
        assert "none of" in str(e).lower() and "are in the" in str(e).lower()


# === Internal Model Tests ===


def test_qrf_internal_model_class() -> None:
    """Test the internal _QRFModel class directly."""
    np.random.seed(42)
    n_samples = 100

    X = pd.DataFrame(
        {
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "cat_feature": np.random.choice(["X", "Y", "Z"], n_samples),
        }
    )
    y = pd.Series(np.random.randn(n_samples), name="target")

    logger = logging.getLogger(__name__)

    # Initialize internal model
    internal_model = _QRFModel(seed=42, logger=logger)

    # Preprocess categorical features
    X_encoded = pd.get_dummies(X, columns=["cat_feature"], drop_first=True)

    # Fit the model
    internal_model.fit(X_encoded, y, n_estimators=30, min_samples_leaf=5)

    # Verify model attributes
    assert internal_model.qrf is not None
    assert internal_model.output_column == "target"

    # Test predictions at different quantiles
    for quantile in [0.1, 0.25, 0.5, 0.75, 0.9]:
        predictions = internal_model.predict(
            X_encoded, mean_quantile=quantile, count_samples=20
        )
        assert isinstance(predictions, pd.Series)
        assert len(predictions) == n_samples
        assert predictions.name == "target"


# === Cross-Validation Test ===


def test_qrf_cross_validation(diabetes_data: pd.DataFrame) -> None:
    """Test QRF model with cross-validation."""
    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1", "s4"]
    data = diabetes_data[predictors + imputed_variables]

    qrf_results = cross_validate_model(
        QRF, data, predictors, imputed_variables
    )

    # Validate cross-validation results - now a dict with dual metrics
    assert isinstance(qrf_results, dict)
    assert "quantile_loss" in qrf_results
    assert "log_loss" in qrf_results

    # Check quantile_loss results (for numerical variables)
    ql_results = qrf_results["quantile_loss"]
    assert isinstance(ql_results["test"], pd.DataFrame)
    assert isinstance(ql_results["train"], pd.DataFrame)
    assert not ql_results["test"].isna().all().all()
    assert ql_results["mean_test"] > 0

    # Test visualization capability with quantile_loss results
    perf_results_viz = model_performance_results(
        results=ql_results["test"],
        model_name="QRF",
        method_name="Cross-Validation Quantile Loss Average",
    )
    assert perf_results_viz is not None


# === Integration Tests ===


def test_qrf_memory_usage_tracking() -> None:
    """Test memory usage tracking functionality."""
    np.random.seed(42)
    n_samples = 50

    data = pd.DataFrame(
        {
            "x1": np.random.randn(n_samples),
            "x2": np.random.randn(n_samples),
            "y1": np.random.randn(n_samples),
            "y2": np.random.randn(n_samples),
        }
    )

    model = QRF()
    memory_info = model._get_memory_usage_info()

    # Should return string with memory info or "N/A"
    assert isinstance(memory_info, str)
    assert ("MB" in memory_info) or (memory_info == "N/A")

    # Test with actual fitting
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.INFO)

    model_with_logging = QRF(log_level="INFO", memory_efficient=True)
    model_with_logging.logger.addHandler(handler)

    fitted_model = model_with_logging.fit(
        data,
        predictors=["x1", "x2"],
        imputed_variables=["y1", "y2"],
        n_estimators=10,
    )

    log_output = log_stream.getvalue()
    assert "Memory usage:" in log_output

    model_with_logging.logger.removeHandler(handler)


# === Performance Test ===


def test_qrf_performance_characteristics(diabetes_data: pd.DataFrame) -> None:
    """Test QRF performance characteristics and validate reasonable accuracy."""
    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1"]
    data = diabetes_data[predictors + imputed_variables]

    # Create train/test split
    np.random.seed(42)
    train_idx = np.random.choice(
        len(data), int(0.8 * len(data)), replace=False
    )
    test_idx = np.array([i for i in range(len(data)) if i not in train_idx])

    train_data = data.iloc[train_idx].reset_index(drop=True)
    test_data = data.iloc[test_idx].reset_index(drop=True)

    X_train = preprocess_data(
        train_data, full_data=True, train_size=1.0, test_size=0.0
    )
    X_test = preprocess_data(
        test_data, full_data=True, train_size=1.0, test_size=0.0
    )

    # Fit model
    model = QRF()
    fitted_model = model.fit(
        X_train, predictors, imputed_variables, n_estimators=50
    )

    # Get predictions
    predictions = fitted_model.predict(X_test, quantiles=[0.5])

    # Calculate correlation with true values
    # When quantiles specified, returns dictionary
    true_values = X_test["s1"].values
    pred_values = predictions[0.5]["s1"].values

    correlation = np.corrcoef(true_values, pred_values)[0, 1]

    # QRF should achieve positive correlation
    assert (
        correlation > 0.0
    ), f"QRF correlation should be positive: {correlation}"

    # Calculate MSE
    mse = mean_squared_error(true_values, pred_values)

    # MSE should be reasonable (not infinite or NaN)
    assert np.isfinite(mse)
    assert mse < 1e6  # Reasonable upper bound
