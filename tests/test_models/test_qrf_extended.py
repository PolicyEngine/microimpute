"""Extended tests for QRF model to improve coverage."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes

from microimpute.models.qrf import QRF, _QRFModel


def test_qrf_model_with_categorical_variables():
    """Test QRF with categorical variables to ensure proper encoding."""
    # Create test data with categorical variable
    np.random.seed(42)
    n_samples = 100

    data = pd.DataFrame(
        {
            "numeric1": np.random.randn(n_samples),
            "numeric2": np.random.randn(n_samples),
            "category": np.random.choice(["A", "B", "C"], n_samples),
            "target": np.random.randn(n_samples),
        }
    )

    # Split data
    train_idx = np.random.choice(
        n_samples, int(0.8 * n_samples), replace=False
    )
    test_idx = np.array([i for i in range(n_samples) if i not in train_idx])

    X_train = data.iloc[train_idx].reset_index(drop=True)
    X_test = data.iloc[test_idx].reset_index(drop=True)

    # Initialize and fit model
    model = QRF()
    fitted_model = model.fit(
        X_train,
        predictors=["numeric1", "numeric2", "category"],
        imputed_variables=["target"],
        n_estimators=50,
    )

    # Predict
    predictions = fitted_model.predict(X_test, quantiles=[0.25, 0.5, 0.75])

    # Verify predictions exist for all quantiles
    assert len(predictions) == 3
    assert all(q in predictions for q in [0.25, 0.5, 0.75])
    assert all(len(predictions[q]) == len(X_test) for q in predictions)


def test_qrf_model_internal_class():
    """Test the internal _QRFModel class directly."""
    # Create test data
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

    # Create logger mock
    import logging

    logger = logging.getLogger(__name__)

    # Initialize internal model
    internal_model = _QRFModel(seed=42, logger=logger)

    # Preprocess categorical features manually for this test
    X_encoded = pd.get_dummies(X, columns=["cat_feature"], drop_first=True)

    # Fit the model
    internal_model.fit(X_encoded, y, n_estimators=50, min_samples_leaf=5)

    # Verify model attributes
    assert internal_model.qrf is not None
    assert internal_model.output_column == "target"

    # Test prediction with different quantiles
    predictions = internal_model.predict(
        X_encoded, mean_quantile=0.25, count_samples=20
    )
    assert isinstance(predictions, pd.Series)
    assert len(predictions) == n_samples
    assert predictions.name == "target"

    # Test with high quantile
    high_quantile_preds = internal_model.predict(
        X_encoded, mean_quantile=0.9, count_samples=15
    )
    assert len(high_quantile_preds) == n_samples

    # Test with low quantile
    low_quantile_preds = internal_model.predict(
        X_encoded, mean_quantile=0.1, count_samples=10
    )
    assert len(low_quantile_preds) == n_samples


def test_qrf_with_missing_categorical_columns_in_test():
    """Test QRF behavior when test data has missing categorical levels."""
    np.random.seed(42)

    # Create training data with categories A, B, C
    train_data = pd.DataFrame(
        {
            "numeric": np.random.randn(100),
            "category": np.random.choice(["A", "B", "C"], 100),
            "target": np.random.randn(100),
        }
    )

    # Create test data with only categories A and B (missing C)
    test_data = pd.DataFrame(
        {
            "numeric": np.random.randn(20),
            "category": np.random.choice(["A", "B"], 20),
            "target": np.nan,  # Will be imputed
        }
    )

    # Fit model
    model = QRF()
    fitted_model = model.fit(
        train_data,
        predictors=["numeric", "category"],
        imputed_variables=["target"],
        n_estimators=30,
    )

    # Predict - should handle missing category gracefully
    predictions = fitted_model.predict(test_data[["numeric", "category"]])

    assert 0.5 in predictions
    assert len(predictions[0.5]) == len(test_data)
    assert not predictions[0.5]["target"].isna().any()


def test_qrf_with_single_predictor():
    """Test QRF with only one predictor variable."""
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "x": np.linspace(0, 10, 100),
            "y": np.sin(np.linspace(0, 10, 100)) + np.random.randn(100) * 0.1,
        }
    )

    train_data = data[:80]
    test_data = data[80:]

    model = QRF()
    fitted_model = model.fit(
        train_data,
        predictors=["x"],
        imputed_variables=["y"],
        n_estimators=50,
    )

    predictions = fitted_model.predict(
        test_data[["x"]], quantiles=[0.1, 0.5, 0.9]
    )

    # Verify prediction intervals make sense
    for i in range(len(test_data)):
        assert predictions[0.1]["y"].iloc[i] <= predictions[0.5]["y"].iloc[i]
        assert predictions[0.5]["y"].iloc[i] <= predictions[0.9]["y"].iloc[i]


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


def test_qrf_reproducibility():
    """Test that QRF produces reproducible results with same seed."""
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "x1": np.random.randn(100),
            "x2": np.random.randn(100),
            "y": np.random.randn(100),
        }
    )

    train_data = data[:80]
    test_data = data[80:]

    # First run
    model1 = QRF()
    fitted1 = model1.fit(
        train_data,
        predictors=["x1", "x2"],
        imputed_variables=["y"],
        n_estimators=30,
    )
    predictions1 = fitted1.predict(test_data[["x1", "x2"]])

    # Second run with same seed
    model2 = QRF()
    fitted2 = model2.fit(
        train_data,
        predictors=["x1", "x2"],
        imputed_variables=["y"],
        n_estimators=30,
    )
    predictions2 = fitted2.predict(test_data[["x1", "x2"]])

    # Results should be identical
    np.testing.assert_array_almost_equal(
        predictions1[0.5]["y"].values,
        predictions2[0.5]["y"].values,
    )


def test_qrf_with_highly_correlated_predictors():
    """Test QRF with highly correlated predictors."""
    np.random.seed(42)
    n_samples = 200

    # Create correlated predictors
    x1 = np.random.randn(n_samples)
    x2 = x1 + np.random.randn(n_samples) * 0.1  # Highly correlated with x1
    x3 = np.random.randn(n_samples)  # Independent

    data = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "y": 2 * x1 + x3 + np.random.randn(n_samples) * 0.5,
        }
    )

    train_data = data[:150]
    test_data = data[150:]

    model = QRF()
    fitted_model = model.fit(
        train_data,
        predictors=["x1", "x2", "x3"],
        imputed_variables=["y"],
        n_estimators=50,
    )

    predictions = fitted_model.predict(test_data[["x1", "x2", "x3"]])

    # Model should still produce reasonable predictions despite correlation
    assert len(predictions[0.5]) == len(test_data)
    assert not predictions[0.5]["y"].isna().any()

    # Check that predictions are somewhat correlated with true values
    true_y = test_data["y"].values
    pred_y = predictions[0.5]["y"].values
    correlation = np.corrcoef(true_y, pred_y)[0, 1]
    assert correlation > 0.5  # Should have reasonable correlation


def test_qrf_error_handling():
    """Test error handling in QRF model."""
    # Test with empty data
    with pytest.raises(Exception):
        model = QRF()
        model.fit(
            pd.DataFrame(),
            predictors=[],
            imputed_variables=[],
        )

    # Test with mismatched predictors
    data = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [4, 5, 6],
        }
    )

    model = QRF()
    fitted_model = model.fit(
        data,
        predictors=["x"],
        imputed_variables=["y"],
    )

    # Try to predict with missing predictor
    test_data = pd.DataFrame(
        {
            "z": [7, 8, 9],  # Wrong column name
        }
    )

    # Should handle gracefully or raise informative error
    try:
        predictions = fitted_model.predict(test_data)
    except Exception as e:
        assert "x" in str(e) or "column" in str(e).lower()
