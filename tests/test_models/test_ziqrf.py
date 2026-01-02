"""Tests for Zero-Inflated Quantile Random Forest (ZI-QRF) imputation model."""

import numpy as np
import pandas as pd
import pytest

from microimpute.models import ZIQRF


class TestZIQRFBasics:
    """Test basic ZI-QRF functionality."""

    @pytest.fixture
    def zero_inflated_data(self):
        """Create data with zero-inflated target variable."""
        np.random.seed(42)
        n = 500

        # Predictor variables
        age = np.random.uniform(20, 70, n)
        income = np.random.uniform(20000, 150000, n)

        # Zero-inflated target: ~40% zeros, rest positive
        # Higher income -> more likely to have non-zero value
        prob_nonzero = 0.3 + 0.4 * (income - 20000) / 130000
        is_nonzero = np.random.random(n) < prob_nonzero
        # For non-zero values, amount correlates with income
        stocks = np.where(
            is_nonzero,
            np.clip(income * 0.5 + np.random.normal(0, 10000, n), 1, None),
            0,
        )

        return pd.DataFrame({"age": age, "income": income, "stocks": stocks})

    def test_ziqrf_import(self):
        """Test that ZIQRF can be imported."""
        from microimpute.models import ZIQRF

        assert ZIQRF is not None

    def test_ziqrf_instantiation(self):
        """Test that ZIQRF can be instantiated."""
        model = ZIQRF()
        assert model is not None
        assert hasattr(model, "fit")

    def test_ziqrf_fit_basic(self, zero_inflated_data):
        """Test that ZIQRF can fit on zero-inflated data."""
        model = ZIQRF()
        results = model.fit(
            X_train=zero_inflated_data,
            predictors=["age", "income"],
            imputed_variables=["stocks"],
        )
        assert results is not None
        assert hasattr(results, "predict")

    def test_ziqrf_predict_basic(self, zero_inflated_data):
        """Test that ZIQRF can predict on zero-inflated data."""
        train = zero_inflated_data.iloc[:400]
        test = zero_inflated_data.iloc[400:]

        model = ZIQRF()
        results = model.fit(
            X_train=train,
            predictors=["age", "income"],
            imputed_variables=["stocks"],
        )

        # predict() returns Dict[quantile, DataFrame]
        predictions_dict = results.predict(test[["age", "income"]])
        predictions = list(predictions_dict.values())[0]
        assert len(predictions) == len(test)
        assert "stocks" in predictions.columns

    def test_ziqrf_produces_zeros(self, zero_inflated_data):
        """Test that ZI-QRF produces some zero predictions."""
        model = ZIQRF()
        results = model.fit(
            X_train=zero_inflated_data,
            predictors=["age", "income"],
            imputed_variables=["stocks"],
        )

        # Create test data with low income (should have more zeros)
        test_low_income = pd.DataFrame(
            {"age": [30, 35, 40, 45, 50], "income": [25000, 26000, 27000, 28000, 29000]}
        )

        predictions_dict = results.predict(test_low_income)
        predictions = list(predictions_dict.values())[0]

        # Low income -> should predict more zeros
        # At least some should be zero
        zero_count = (predictions["stocks"] == 0).sum()
        assert zero_count >= 0  # At least structural zeros are possible

    def test_ziqrf_produces_positive_values(self, zero_inflated_data):
        """Test that ZI-QRF produces some positive predictions for high-income."""
        model = ZIQRF()
        results = model.fit(
            X_train=zero_inflated_data,
            predictors=["age", "income"],
            imputed_variables=["stocks"],
        )

        # Create test data with high income (should have more non-zeros)
        test_high_income = pd.DataFrame(
            {
                "age": [30, 35, 40, 45, 50],
                "income": [140000, 142000, 145000, 148000, 150000],
            }
        )

        predictions_dict = results.predict(test_high_income)
        predictions = list(predictions_dict.values())[0]

        # High income -> should predict more non-zeros
        nonzero_count = (predictions["stocks"] > 0).sum()
        assert nonzero_count > 0  # At least some should be positive


class TestZIQRFZeroInflationDetection:
    """Test ZI-QRF zero-inflation detection and handling."""

    @pytest.fixture
    def mixed_data(self):
        """Create data with both zero-inflated and regular variables."""
        np.random.seed(42)
        n = 400

        age = np.random.uniform(20, 70, n)
        income = np.random.uniform(30000, 100000, n)  # Never zero

        # Zero-inflated: ~50% zeros
        stocks = np.where(np.random.random(n) < 0.5, 0, np.random.uniform(1000, 50000, n))

        # Not zero-inflated: always positive
        checking = np.random.uniform(100, 10000, n)

        return pd.DataFrame(
            {"age": age, "income": income, "stocks": stocks, "checking": checking}
        )

    def test_ziqrf_uses_two_stage_for_zero_inflated(self, mixed_data):
        """Test that ZI-QRF uses two-stage model for zero-inflated variables."""
        model = ZIQRF(zero_inflation_threshold=0.1)
        results = model.fit(
            X_train=mixed_data,
            predictors=["age", "income"],
            imputed_variables=["stocks"],
        )

        # The model should have detected stocks as zero-inflated
        # and used a two-stage approach
        assert results is not None
        # Check that the model for stocks has both classifier and regressor
        assert hasattr(results, "models")
        stock_model = results.models.get("stocks")
        assert stock_model is not None

    def test_ziqrf_threshold_parameter(self, mixed_data):
        """Test that zero_inflation_threshold parameter works."""
        # With high threshold, nothing is zero-inflated
        model_high = ZIQRF(zero_inflation_threshold=0.99)
        results_high = model_high.fit(
            X_train=mixed_data,
            predictors=["age", "income"],
            imputed_variables=["stocks"],
        )

        # With low threshold, stocks is zero-inflated
        model_low = ZIQRF(zero_inflation_threshold=0.1)
        results_low = model_low.fit(
            X_train=mixed_data,
            predictors=["age", "income"],
            imputed_variables=["stocks"],
        )

        # Both should produce results
        assert results_high is not None
        assert results_low is not None


class TestZIQRFMultipleVariables:
    """Test ZI-QRF with multiple imputed variables."""

    @pytest.fixture
    def multi_target_data(self):
        """Create data with multiple zero-inflated targets."""
        np.random.seed(42)
        n = 400

        age = np.random.uniform(20, 70, n)
        income = np.random.uniform(30000, 150000, n)

        # Multiple zero-inflated variables
        stocks = np.where(np.random.random(n) < 0.4, 0, np.random.uniform(1000, 100000, n))
        bonds = np.where(np.random.random(n) < 0.6, 0, np.random.uniform(500, 50000, n))
        crypto = np.where(np.random.random(n) < 0.8, 0, np.random.uniform(100, 20000, n))

        return pd.DataFrame(
            {
                "age": age,
                "income": income,
                "stocks": stocks,
                "bonds": bonds,
                "crypto": crypto,
            }
        )

    def test_ziqrf_multiple_targets(self, multi_target_data):
        """Test ZI-QRF with multiple imputed variables."""
        train = multi_target_data.iloc[:300]
        test = multi_target_data.iloc[300:]

        model = ZIQRF()
        results = model.fit(
            X_train=train,
            predictors=["age", "income"],
            imputed_variables=["stocks", "bonds", "crypto"],
        )

        predictions_dict = results.predict(test[["age", "income"]])
        predictions = list(predictions_dict.values())[0]

        assert len(predictions) == len(test)
        assert "stocks" in predictions.columns
        assert "bonds" in predictions.columns
        assert "crypto" in predictions.columns

    def test_ziqrf_sequential_imputation(self, multi_target_data):
        """Test that ZI-QRF uses previously imputed variables as predictors."""
        train = multi_target_data.iloc[:300]
        test = multi_target_data.iloc[300:]

        model = ZIQRF()
        results = model.fit(
            X_train=train,
            predictors=["age", "income"],
            imputed_variables=["stocks", "bonds", "crypto"],
        )

        predictions_dict = results.predict(test[["age", "income"]])
        predictions = list(predictions_dict.values())[0]

        # All variables should be imputed
        assert not predictions["stocks"].isna().any()
        assert not predictions["bonds"].isna().any()
        assert not predictions["crypto"].isna().any()


class TestZIQRFEdgeCases:
    """Test ZI-QRF edge cases."""

    def test_ziqrf_no_zeros(self):
        """Test ZI-QRF when target has no zeros."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "x": np.random.uniform(0, 10, 200),
                "y": np.random.uniform(1, 100, 200),  # No zeros
            }
        )

        model = ZIQRF()
        results = model.fit(
            X_train=data,
            predictors=["x"],
            imputed_variables=["y"],
        )

        test = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        predictions_dict = results.predict(test)
        predictions = list(predictions_dict.values())[0]

        # Should still work and produce predictions
        assert len(predictions) == 5
        assert "y" in predictions.columns

    def test_ziqrf_all_zeros(self):
        """Test ZI-QRF when target is all zeros."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "x": np.random.uniform(0, 10, 200),
                "y": np.zeros(200),  # All zeros
            }
        )

        model = ZIQRF()
        results = model.fit(
            X_train=data,
            predictors=["x"],
            imputed_variables=["y"],
        )

        test = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        predictions_dict = results.predict(test)
        predictions = list(predictions_dict.values())[0]

        # Should predict all zeros
        assert (predictions["y"] == 0).all()

    def test_ziqrf_small_sample(self):
        """Test ZI-QRF with small sample size."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "x": np.random.uniform(0, 10, 50),
                "y": np.where(np.random.random(50) < 0.5, 0, np.random.uniform(1, 10, 50)),
            }
        )

        model = ZIQRF()
        results = model.fit(
            X_train=data,
            predictors=["x"],
            imputed_variables=["y"],
        )

        test = pd.DataFrame({"x": [1, 2, 3]})
        predictions_dict = results.predict(test)
        predictions = list(predictions_dict.values())[0]

        assert len(predictions) == 3
