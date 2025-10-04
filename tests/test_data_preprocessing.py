"""Tests for data preprocessing utilities."""

import numpy as np
import pandas as pd
import pytest

from microimpute.utils.data import normalize_data, preprocess_data


class TestNormalize:
    """Test the normalize function."""

    def test_normalize_excludes_categorical_columns(self):
        """Test that categorical columns are not normalized."""
        data = pd.DataFrame(
            {
                "numeric_col": [1.0, 2.5, 3.7, 4.2, 5.9],  # Non-equally spaced
                "categorical_col": [1, 2, 3, 1, 2],
                "boolean_col": [0, 1, 0, 1, 0],
            }
        )

        normalized_data, norm_params = normalize_data(data)

        # Categorical and boolean columns should be unchanged
        pd.testing.assert_series_equal(
            normalized_data["categorical_col"], data["categorical_col"]
        )
        pd.testing.assert_series_equal(
            normalized_data["boolean_col"], data["boolean_col"]
        )

        # Numeric column should be normalized
        assert not np.allclose(
            normalized_data["numeric_col"].values, data["numeric_col"].values
        )

        # Only numeric column should have normalization params
        assert "numeric_col" in norm_params
        assert "categorical_col" not in norm_params
        assert "boolean_col" not in norm_params

    def test_normalize_preserves_column_names(self):
        """Test that normalization doesn't modify column names."""
        data = pd.DataFrame(
            {
                "age": [25, 30, 35, 40, 45],
                "race": [1, 2, 3, 1, 2],
                "is_female": [0, 1, 0, 1, 0],
                "income": [50000, 60000, 70000, 80000, 90000],
            }
        )

        normalized_data, norm_params = normalize_data(data)

        # Column names should be identical
        assert list(normalized_data.columns) == list(data.columns)

        # Categorical columns should have exact same values
        pd.testing.assert_series_equal(normalized_data["race"], data["race"])
        pd.testing.assert_series_equal(
            normalized_data["is_female"], data["is_female"]
        )

    def test_normalize_correctly_normalizes_numeric_columns(self):
        """Test that numeric columns are normalized with mean=0, std=1."""
        data = pd.DataFrame(
            {
                "value1": [10.5, 20.3, 30.1, 40.7, 50.2],  # Non-equally spaced
                "value2": [
                    105.0,
                    215.0,
                    295.0,
                    410.0,
                    505.0,
                ],  # Non-equally spaced
                "category": [1, 2, 1, 2, 1],
            }
        )

        normalized_data, norm_params = normalize_data(data)

        # Check that numeric columns have mean ≈ 0 and std ≈ 1
        assert np.isclose(normalized_data["value1"].mean(), 0.0, atol=1e-10)
        assert np.isclose(normalized_data["value1"].std(), 1.0, atol=1e-10)
        assert np.isclose(normalized_data["value2"].mean(), 0.0, atol=1e-10)
        assert np.isclose(normalized_data["value2"].std(), 1.0, atol=1e-10)

        # Check normalization params are stored correctly
        assert "value1" in norm_params
        assert "value2" in norm_params
        assert np.isclose(norm_params["value1"]["mean"], 30.36, atol=0.01)
        assert np.isclose(norm_params["value2"]["mean"], 306.0, atol=1.0)

    def test_normalize_handles_constant_columns(self):
        """Test that constant columns (std=0) are handled correctly."""
        data = pd.DataFrame(
            {
                "constant": [5.0, 5.0, 5.0, 5.0, 5.0],
                "varying": [1.2, 2.7, 3.1, 4.5, 5.9],  # Non-equally spaced
            }
        )

        normalized_data, norm_params = normalize_data(data)

        # Constant columns are detected as numeric_categorical and excluded
        # So they should remain unchanged
        pd.testing.assert_series_equal(
            normalized_data["constant"], data["constant"]
        )

        # Only varying column should have normalization params
        assert "constant" not in norm_params
        assert "varying" in norm_params

    def test_normalize_returns_copy(self):
        """Test that normalize returns a copy and doesn't modify original."""
        data = pd.DataFrame(
            {
                "value": [1.3, 2.7, 3.2, 4.8, 5.1],  # Non-equally spaced
                "category": [1, 2, 1, 2, 1],
            }
        )
        original_data = data.copy()

        normalized_data, _ = normalize_data(data)

        # Original data should be unchanged
        pd.testing.assert_frame_equal(data, original_data)

        # Normalized data should be different
        assert not normalized_data["value"].equals(data["value"])

    def test_normalize_with_no_numeric_columns(self):
        """Test normalize with only categorical columns."""
        data = pd.DataFrame({"cat1": [1, 2, 3, 1, 2], "cat2": [0, 1, 0, 1, 0]})

        normalized_data, norm_params = normalize_data(data)

        # Data should be unchanged
        pd.testing.assert_frame_equal(normalized_data, data)

        # No normalization params should be returned
        assert norm_params == {}


class TestPreprocessDataWithNormalize:
    """Test that preprocess_data correctly uses the normalize function."""

    def test_preprocess_data_excludes_categoricals_from_normalization(self):
        """Test that preprocess_data doesn't normalize categorical columns."""
        data = pd.DataFrame(
            {
                "age": [
                    25.3,
                    30.7,
                    35.2,
                    40.9,
                    45.1,
                ],  # Non-equally spaced floats
                "race": [1, 2, 3, 1, 2],
                "is_female": [0, 1, 0, 1, 0],
                "income": [
                    50123.45,
                    60987.23,
                    70456.78,
                    80234.56,
                    90876.12,
                ],  # Non-equally spaced
            }
        )

        result, norm_params = preprocess_data(
            data, full_data=True, normalize=True
        )

        # Categorical columns should be unchanged
        pd.testing.assert_series_equal(result["race"], data["race"])
        pd.testing.assert_series_equal(result["is_female"], data["is_female"])

        # Numeric columns should be normalized
        assert not np.allclose(result["age"].values, data["age"].values)
        assert not np.allclose(result["income"].values, data["income"].values)

        # Only numeric columns in norm_params
        assert "age" in norm_params
        assert "income" in norm_params
        assert "race" not in norm_params
        assert "is_female" not in norm_params

    def test_categorical_columns_dont_get_weird_suffixes_when_dummified(
        self,
    ):
        """
        Test that categorical columns normalized then dummified
        don't get random float suffixes.

        This is the core bug we're fixing.
        """
        data = pd.DataFrame(
            {
                "race": [1, 2, 3, 1, 2, 3, 1, 2],
                "income": [
                    50000,
                    60000,
                    70000,
                    80000,
                    90000,
                    100000,
                    110000,
                    120000,
                ],
            }
        )

        # Normalize the data
        normalized_data, norm_params = normalize_data(data)

        # Now apply pd.get_dummies to the race column
        dummies = pd.get_dummies(
            normalized_data[["race"]],
            columns=["race"],
            drop_first=True,
        )

        # Check that dummy column names are clean (no float suffixes)
        for col in dummies.columns:
            # Should be like "race_2", "race_3", not "race_1.234567"
            assert col in [
                "race_2",
                "race_3",
            ], f"Unexpected column name: {col}"

            # Column name should not contain decimal points
            assert "." not in col, f"Column {col} has decimal point in name"
