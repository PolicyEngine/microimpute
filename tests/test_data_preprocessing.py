"""Tests for data preprocessing utilities."""

import numpy as np
import pandas as pd
import pytest

from microimpute.utils.data import (
    asinh_transform_data,
    log_transform_data,
    normalize_data,
    preprocess_data,
    un_asinh_transform_predictions,
    unlog_transform_predictions,
)


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

        result, transform_params = preprocess_data(
            data, full_data=True, normalize=True
        )

        # Extract normalization params from nested dict
        norm_params = transform_params["normalization"]

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


class TestLogTransform:
    """Test the log_transform_data function."""

    def test_log_transform_excludes_categorical_columns(self):
        """Test that categorical columns are not log transformed."""
        data = pd.DataFrame(
            {
                "numeric_col": [1.0, 2.5, 3.7, 4.2, 5.9],
                "categorical_col": [1, 2, 3, 1, 2],
                "boolean_col": [0, 1, 0, 1, 0],
            }
        )

        log_data, log_params = log_transform_data(data)

        # Categorical and boolean columns should be unchanged
        pd.testing.assert_series_equal(
            log_data["categorical_col"], data["categorical_col"]
        )
        pd.testing.assert_series_equal(
            log_data["boolean_col"], data["boolean_col"]
        )

        # Numeric column should be log transformed
        assert not np.allclose(
            log_data["numeric_col"].values, data["numeric_col"].values
        )

        # Only numeric column should have log transform params
        assert "numeric_col" in log_params
        assert "categorical_col" not in log_params
        assert "boolean_col" not in log_params

    def test_log_transform_correctly_transforms_numeric_columns(self):
        """Test that numeric columns are correctly log transformed."""
        data = pd.DataFrame(
            {
                "value1": [
                    1.5,
                    2.7,
                    3.2,
                    4.8,
                    5.1,
                    6.3,
                    7.9,
                    8.4,
                    9.6,
                    10.2,
                ],
                "value2": [
                    15.5,
                    27.3,
                    32.1,
                    48.7,
                    51.9,
                    63.2,
                    79.8,
                    84.5,
                    96.1,
                    102.4,
                ],
                "category": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            }
        )

        log_data, log_params = log_transform_data(data)

        # Check that numeric columns are log transformed
        expected_value1 = np.log(data["value1"].values)
        expected_value2 = np.log(data["value2"].values)

        np.testing.assert_array_almost_equal(
            log_data["value1"].values, expected_value1
        )
        np.testing.assert_array_almost_equal(
            log_data["value2"].values, expected_value2
        )

        # Check log transform params are stored
        assert "value1" in log_params
        assert "value2" in log_params

    def test_log_transform_rejects_non_positive_values(self):
        """Test that log transform raises error for non-positive values."""
        data = pd.DataFrame(
            {
                "value": [1.0, 2.0, 0.0, 4.0, 5.0],  # Contains zero
            }
        )

        with pytest.raises(ValueError, match="non-positive values"):
            log_transform_data(data)

        data_negative = pd.DataFrame(
            {
                "value": [1.0, 2.0, -1.0, 4.0, 5.0],  # Contains negative
            }
        )

        with pytest.raises(ValueError, match="non-positive values"):
            log_transform_data(data_negative)

    def test_log_transform_returns_copy(self):
        """Test that log transform returns a copy."""
        data = pd.DataFrame(
            {
                "value": [1.5, 2.7, 3.2, 4.8, 5.1, 6.3, 7.9, 8.4, 9.6, 10.2],
                "category": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            }
        )
        original_data = data.copy()

        log_data, _ = log_transform_data(data)

        # Original data should be unchanged
        pd.testing.assert_frame_equal(data, original_data)

        # Log transformed data should be different
        assert not log_data["value"].equals(data["value"])

    def test_log_transform_with_no_numeric_columns(self):
        """Test log transform with only categorical columns."""
        data = pd.DataFrame({"cat1": [1, 2, 3, 1, 2], "cat2": [0, 1, 0, 1, 0]})

        log_data, log_params = log_transform_data(data)

        # Data should be unchanged
        pd.testing.assert_frame_equal(log_data, data)

        # No log transform params should be returned
        assert log_params == {}


class TestUnlogTransformPredictions:
    """Test the unlog_transform_predictions function."""

    def test_unlog_transform_reverses_log_transform(self):
        """Test that unlog transform correctly reverses log transform."""
        original = pd.DataFrame(
            {
                "value1": [
                    1.5,
                    2.7,
                    3.2,
                    4.8,
                    5.1,
                    6.3,
                    7.9,
                    8.4,
                    9.6,
                    10.2,
                ],
                "value2": [
                    15.5,
                    27.3,
                    32.1,
                    48.7,
                    51.9,
                    63.2,
                    79.8,
                    84.5,
                    96.1,
                    102.4,
                ],
            }
        )

        # Apply log transform
        log_data, log_params = log_transform_data(original)

        # Create imputations dict (simulating prediction output)
        imputations = {0.5: log_data}

        # Reverse log transform
        reversed_data = unlog_transform_predictions(imputations, log_params)

        # Should match original data
        pd.testing.assert_frame_equal(
            reversed_data[0.5], original, check_exact=False, atol=1e-10
        )

    def test_unlog_transform_raises_error_for_missing_params(self):
        """Test that unlog transform raises error when params are missing."""
        imputations = {
            0.5: pd.DataFrame(
                {
                    "value1": [0.0, 0.69, 1.10],
                    "value2": [2.3, 3.0, 3.9],
                }
            )
        }

        # Only have params for value1, not value2
        log_params = {"value1": {}}

        with pytest.raises(
            ValueError, match="Missing log transformation parameters"
        ):
            unlog_transform_predictions(imputations, log_params)


class TestPreprocessDataWithLogTransform:
    """Test that preprocess_data correctly uses log transformation."""

    def test_preprocess_data_excludes_categoricals_from_log_transform(self):
        """Test that preprocess_data doesn't log transform categorical columns."""
        data = pd.DataFrame(
            {
                "age": [
                    25.3,
                    30.7,
                    35.2,
                    40.9,
                    45.1,
                    50.6,
                    55.8,
                    60.3,
                    65.7,
                    70.2,
                ],
                "race": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
                "is_female": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                "income": [
                    50123.45,
                    60987.23,
                    70456.78,
                    80234.56,
                    90876.12,
                    100543.89,
                    110234.67,
                    120789.34,
                    130456.78,
                    140987.23,
                ],
            }
        )

        result, transform_params = preprocess_data(
            data, full_data=True, log_transform=True
        )

        # Extract log transform params from nested dict
        log_params = transform_params["log_transform"]

        # Categorical columns should be unchanged
        pd.testing.assert_series_equal(result["race"], data["race"])
        pd.testing.assert_series_equal(result["is_female"], data["is_female"])

        # Numeric columns should be log transformed
        assert not np.allclose(result["age"].values, data["age"].values)
        assert not np.allclose(result["income"].values, data["income"].values)

        # Only numeric columns in log_params
        assert "age" in log_params
        assert "income" in log_params
        assert "race" not in log_params
        assert "is_female" not in log_params

    def test_preprocess_data_raises_error_for_both_normalize_and_log(
        self,
    ):
        """Test that preprocess_data raises error if both normalize and log_transform are True."""
        data = pd.DataFrame(
            {
                "value1": [1.5, 2.7, 3.2, 4.8, 5.1, 6.3, 7.9, 8.4, 9.6, 10.2],
                "value2": [
                    15.5,
                    27.3,
                    32.1,
                    48.7,
                    51.9,
                    63.2,
                    79.8,
                    84.5,
                    96.1,
                    102.4,
                ],
            }
        )

        with pytest.raises(
            ValueError,
            match="Cannot apply multiple transformations",
        ):
            preprocess_data(
                data, full_data=True, normalize=True, log_transform=True
            )

    def test_preprocess_data_with_log_transform_and_split(self):
        """Test that preprocess_data correctly splits and log transforms data."""
        data = pd.DataFrame(
            {
                "value1": [
                    1.5,
                    2.7,
                    3.2,
                    4.8,
                    5.1,
                    6.3,
                    7.9,
                    8.4,
                    9.6,
                    10.2,
                    11.5,
                    12.8,
                    13.3,
                    14.9,
                    15.2,
                ],
                "value2": [
                    15.5,
                    27.3,
                    32.1,
                    48.7,
                    51.9,
                    63.2,
                    79.8,
                    84.5,
                    96.1,
                    102.4,
                    115.7,
                    128.9,
                    133.6,
                    149.2,
                    152.8,
                ],
            }
        )

        X_train, X_test, transform_params = preprocess_data(
            data,
            full_data=False,
            test_size=0.2,
            train_size=None,
            random_state=42,
            log_transform=True,
        )

        # Extract log transform params from nested dict
        log_params = transform_params["log_transform"]

        # Check that data is split
        assert len(X_train) == 12
        assert len(X_test) == 3

        # Check that log params are returned
        assert "value1" in log_params
        assert "value2" in log_params

        # Check that values are log transformed (compare to original)
        assert not any(X_train["value1"].isin(data["value1"]))
        assert not any(X_test["value1"].isin(data["value1"]))


class TestPreprocessDataWithSelectiveTransformation:
    """Test preprocess_data with selective column transformation."""

    def test_normalize_only_specified_columns(self):
        """Test that only specified columns are normalized."""
        data = pd.DataFrame(
            {
                "age": [
                    23,
                    30,
                    35,
                    46,
                    45,
                    52,
                    55,
                    61,
                    68,
                    72,
                ],
                "income": [
                    50123.45,
                    60987.23,
                    70456.78,
                    80234.56,
                    90876.12,
                    100543.89,
                    110234.67,
                    120789.34,
                    130456.78,
                    140987.23,
                ],
                "wealth": [
                    150000.5,
                    250000.3,
                    350000.7,
                    450000.2,
                    550000.9,
                    650000.1,
                    750000.4,
                    850000.8,
                    950000.6,
                    1050000.3,
                ],
            }
        )

        # Only normalize income column
        result, transform_params = preprocess_data(
            data, full_data=True, normalize=["income"]
        )

        # Extract normalization params from nested dict
        norm_params = transform_params["normalization"]

        # Income should be normalized
        assert not np.allclose(result["income"].values, data["income"].values)
        assert "income" in norm_params

        # Age and wealth should NOT be normalized
        pd.testing.assert_series_equal(result["age"], data["age"])
        pd.testing.assert_series_equal(result["wealth"], data["wealth"])
        assert "age" not in norm_params
        assert "wealth" not in norm_params

    def test_log_transform_only_specified_columns(self):
        """Test that only specified columns are log transformed."""
        data = pd.DataFrame(
            {
                "age": [
                    23,
                    30,
                    35,
                    46,
                    45,
                    52,
                    55,
                    61,
                    68,
                    72,
                ],
                "income": [
                    50123.45,
                    60987.23,
                    70456.78,
                    80234.56,
                    90876.12,
                    100543.89,
                    110234.67,
                    120789.34,
                    130456.78,
                    140987.23,
                ],
                "wealth": [
                    150000.5,
                    250000.3,
                    350000.7,
                    450000.2,
                    550000.9,
                    650000.1,
                    750000.4,
                    850000.8,
                    950000.6,
                    1050000.3,
                ],
            }
        )

        # Only log transform income column
        result, transform_params = preprocess_data(
            data, full_data=True, log_transform=["income"]
        )

        # Extract log transform params from nested dict
        log_params = transform_params["log_transform"]

        # Income should be log transformed
        assert not np.allclose(result["income"].values, data["income"].values)
        assert "income" in log_params

        # Age and wealth should NOT be transformed
        pd.testing.assert_series_equal(result["age"], data["age"])
        pd.testing.assert_series_equal(result["wealth"], data["wealth"])
        assert "age" not in log_params
        assert "wealth" not in log_params

    def test_normalize_multiple_specified_columns(self):
        """Test normalizing multiple specified columns."""
        data = pd.DataFrame(
            {
                "age": [
                    23,
                    30,
                    35,
                    46,
                    45,
                    52,
                    55,
                    61,
                    68,
                    72,
                ],
                "income": [
                    50123.45,
                    60987.23,
                    70456.78,
                    80234.56,
                    90876.12,
                    100543.89,
                    110234.67,
                    120789.34,
                    130456.78,
                    140987.23,
                ],
                "wealth": [
                    150000.5,
                    250000.3,
                    350000.7,
                    450000.2,
                    550000.9,
                    650000.1,
                    750000.4,
                    850000.8,
                    950000.6,
                    1050000.3,
                ],
            }
        )

        # Normalize income and wealth, but not age
        result, transform_params = preprocess_data(
            data, full_data=True, normalize=["income", "wealth"]
        )

        # Extract normalization params from nested dict
        norm_params = transform_params["normalization"]

        # Income and wealth should be normalized
        assert not np.allclose(result["income"].values, data["income"].values)
        assert not np.allclose(result["wealth"].values, data["wealth"].values)
        assert "income" in norm_params
        assert "wealth" in norm_params

        # Age should NOT be normalized
        pd.testing.assert_series_equal(result["age"], data["age"])
        assert "age" not in norm_params

    def test_error_on_nonexistent_column_normalize(self):
        """Test that error is raised when specifying non-existent column."""
        data = pd.DataFrame(
            {
                "age": [25.3, 30.7, 35.2, 40.9, 45.1],
                "income": [50123.45, 60987.23, 70456.78, 80234.56, 90876.12],
            }
        )

        with pytest.raises(ValueError, match="not found in data"):
            preprocess_data(
                data, full_data=True, normalize=["income", "nonexistent"]
            )

    def test_error_on_nonexistent_column_log_transform(self):
        """Test that error is raised when specifying non-existent column."""
        data = pd.DataFrame(
            {
                "age": [25.3, 30.7, 35.2, 40.9, 45.1],
                "income": [50123.45, 60987.23, 70456.78, 80234.56, 90876.12],
            }
        )

        with pytest.raises(ValueError, match="not found in data"):
            preprocess_data(
                data, full_data=True, log_transform=["income", "nonexistent"]
            )

    def test_error_on_overlapping_columns(self):
        """Test error when both normalize and log_transform target same columns."""
        data = pd.DataFrame(
            {
                "age": [25.3, 30.7, 35.2, 40.9, 45.1],
                "income": [50123.45, 60987.23, 70456.78, 80234.56, 90876.12],
            }
        )

        # Error when same column is in both lists
        with pytest.raises(
            ValueError, match="Cannot apply both .* to the same columns"
        ):
            preprocess_data(
                data,
                full_data=True,
                normalize=["income", "age"],
                log_transform=["age"],
            )

    def test_both_transformations_on_different_columns(self):
        """Test that both transformations work when applied to different columns."""
        data = pd.DataFrame(
            {
                "age": [
                    23,
                    30,
                    35,
                    46,
                    45,
                    52,
                    55,
                    61,
                    68,
                    72,
                ],
                "income": [
                    50123.45,
                    60987.23,
                    70456.78,
                    80234.56,
                    90876.12,
                    100543.89,
                    110234.67,
                    120789.34,
                    130456.78,
                    140987.23,
                ],
                "wealth": [
                    150000.5,
                    250000.3,
                    350000.7,
                    450000.2,
                    550000.9,
                    650000.1,
                    750000.4,
                    850000.8,
                    950000.6,
                    1050000.3,
                ],
            }
        )

        # Normalize age, log transform income, leave wealth unchanged
        result, transform_params = preprocess_data(
            data,
            full_data=True,
            normalize=["age"],
            log_transform=["income"],
        )

        # Extract both parameter dicts
        norm_params = transform_params["normalization"]
        log_params = transform_params["log_transform"]

        # Age should be normalized
        assert not np.allclose(result["age"].values, data["age"].values)
        assert "age" in norm_params
        assert "age" not in log_params

        # Income should be log transformed
        assert not np.allclose(result["income"].values, data["income"].values)
        assert "income" in log_params
        assert "income" not in norm_params

        # Wealth should be unchanged
        pd.testing.assert_series_equal(result["wealth"], data["wealth"])
        assert "wealth" not in norm_params
        assert "wealth" not in log_params


class TestAsinhTransform:
    """Test the asinh_transform_data function."""

    def test_asinh_transform_excludes_categorical_columns(self):
        """Test that categorical columns are not asinh transformed."""
        data = pd.DataFrame(
            {
                "numeric_col": [1.0, 2.5, 3.7, 4.2, 5.9],
                "categorical_col": [1, 2, 3, 1, 2],
                "boolean_col": [0, 1, 0, 1, 0],
            }
        )

        asinh_data, asinh_params = asinh_transform_data(data)

        # Categorical and boolean columns should be unchanged
        pd.testing.assert_series_equal(
            asinh_data["categorical_col"], data["categorical_col"]
        )
        pd.testing.assert_series_equal(
            asinh_data["boolean_col"], data["boolean_col"]
        )

        # Numeric column should be asinh transformed
        assert not np.allclose(
            asinh_data["numeric_col"].values, data["numeric_col"].values
        )

        # Only numeric column should have asinh transform params
        assert "numeric_col" in asinh_params
        assert "categorical_col" not in asinh_params
        assert "boolean_col" not in asinh_params

    def test_asinh_transform_correctly_transforms_numeric_columns(self):
        """Test that numeric columns are correctly asinh transformed."""
        data = pd.DataFrame(
            {
                "value1": [
                    -10.5,
                    -2.7,
                    0.0,
                    2.8,
                    10.1,
                    100.3,
                    1000.9,
                    10000.4,
                    100000.6,
                    1000000.2,
                ],
                "value2": [
                    -1000.5,
                    -100.3,
                    -10.1,
                    0.0,
                    10.9,
                    100.2,
                    1000.8,
                    10000.5,
                    100000.1,
                    1000000.7,
                ],
                "category": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            }
        )

        asinh_data, asinh_params = asinh_transform_data(data)

        # Check that numeric columns are asinh transformed
        expected_value1 = np.arcsinh(data["value1"].values)
        expected_value2 = np.arcsinh(data["value2"].values)

        np.testing.assert_array_almost_equal(
            asinh_data["value1"].values, expected_value1
        )
        np.testing.assert_array_almost_equal(
            asinh_data["value2"].values, expected_value2
        )

        # Check asinh transform params are stored
        assert "value1" in asinh_params
        assert "value2" in asinh_params

    def test_asinh_transform_handles_negative_and_zero_values(self):
        """Test that asinh transform correctly handles negative and zero values."""
        data = pd.DataFrame(
            {
                "value": [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0],
            }
        )

        # Should NOT raise an error (unlike log transform)
        asinh_data, asinh_params = asinh_transform_data(data)

        # Check that transformation is symmetric
        assert np.isclose(
            asinh_data["value"].iloc[0], -asinh_data["value"].iloc[6]
        )
        assert np.isclose(
            asinh_data["value"].iloc[1], -asinh_data["value"].iloc[5]
        )
        assert np.isclose(
            asinh_data["value"].iloc[2], -asinh_data["value"].iloc[4]
        )
        assert np.isclose(asinh_data["value"].iloc[3], 0.0)

    def test_asinh_transform_returns_copy(self):
        """Test that asinh transform returns a copy."""
        data = pd.DataFrame(
            {
                "value": [-10.5, -2.7, 0.0, 2.8, 10.1, 100.3, 1000.9, 10000.4],
                "category": [1, 2, 1, 2, 1, 2, 1, 2],
            }
        )
        original_data = data.copy()

        asinh_data, _ = asinh_transform_data(data)

        # Original data should be unchanged
        pd.testing.assert_frame_equal(data, original_data)

        # Asinh transformed data should be different
        assert not asinh_data["value"].equals(data["value"])


class TestUnAsinhTransformPredictions:
    """Test the un_asinh_transform_predictions function."""

    def test_un_asinh_transform_reverses_asinh_transform(self):
        """Test that un_asinh transform correctly reverses asinh transform."""
        original = pd.DataFrame(
            {
                "value1": [
                    -100.5,
                    -10.7,
                    0.0,
                    10.8,
                    100.1,
                    1000.3,
                ],
                "value2": [
                    -1000.5,
                    -100.3,
                    -10.1,
                    0.0,
                    100.9,
                    1000.2,
                ],
            }
        )

        # Apply asinh transform
        asinh_data, asinh_params = asinh_transform_data(original)

        # Create imputations dict (simulating prediction output)
        imputations = {0.5: asinh_data}

        # Reverse asinh transform
        reversed_data = un_asinh_transform_predictions(
            imputations, asinh_params
        )

        # Should match original data
        pd.testing.assert_frame_equal(
            reversed_data[0.5], original, check_exact=False, atol=1e-10
        )

    def test_un_asinh_transform_raises_error_for_missing_params(self):
        """Test that un_asinh transform raises error when params are missing."""
        imputations = {
            0.5: pd.DataFrame(
                {
                    "value1": [0.0, 0.69, 1.10],
                    "value2": [2.3, 3.0, 3.9],
                }
            )
        }

        # Only have params for value1, not value2
        asinh_params = {"value1": {}}

        with pytest.raises(
            ValueError, match="Missing asinh transformation parameters"
        ):
            un_asinh_transform_predictions(imputations, asinh_params)


class TestPreprocessDataWithAsinhTransform:
    """Test that preprocess_data correctly uses asinh transformation."""

    def test_preprocess_data_excludes_categoricals_from_asinh_transform(self):
        """Test that preprocess_data doesn't asinh transform categorical cols."""
        data = pd.DataFrame(
            {
                "age": [
                    -5.3,
                    0.0,
                    35.2,
                    40.9,
                    45.1,
                    50.6,
                    55.8,
                    60.3,
                    65.7,
                    70.2,
                ],
                "race": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
                "is_female": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                "income": [
                    -10000.0,
                    0.0,
                    70456.78,
                    80234.56,
                    90876.12,
                    100543.89,
                    110234.67,
                    120789.34,
                    130456.78,
                    140987.23,
                ],
            }
        )

        result, transform_params = preprocess_data(
            data, full_data=True, asinh_transform=True
        )

        # Extract asinh transform params from nested dict
        asinh_params = transform_params["asinh_transform"]

        # Categorical columns should be unchanged
        pd.testing.assert_series_equal(result["race"], data["race"])
        pd.testing.assert_series_equal(result["is_female"], data["is_female"])

        # Numeric columns should be asinh transformed
        assert not np.allclose(result["age"].values, data["age"].values)
        assert not np.allclose(result["income"].values, data["income"].values)

        # Only numeric columns in asinh_params
        assert "age" in asinh_params
        assert "income" in asinh_params
        assert "race" not in asinh_params
        assert "is_female" not in asinh_params

    def test_preprocess_data_with_asinh_transform_on_specific_columns(self):
        """Test asinh transform on specific columns only."""
        data = pd.DataFrame(
            {
                "age": [
                    23,
                    30,
                    35,
                    46,
                    45,
                    52,
                    55,
                    61,
                    68,
                    72,
                ],
                "income": [
                    -10000.0,
                    0.0,
                    70456.78,
                    80234.56,
                    90876.12,
                    100543.89,
                    110234.67,
                    120789.34,
                    130456.78,
                    140987.23,
                ],
                "wealth": [
                    -50000.5,
                    0.0,
                    350000.7,
                    450000.2,
                    550000.9,
                    650000.1,
                    750000.4,
                    850000.8,
                    950000.6,
                    1050000.3,
                ],
            }
        )

        # Only asinh transform income column
        result, transform_params = preprocess_data(
            data, full_data=True, asinh_transform=["income"]
        )

        # Extract asinh transform params from nested dict
        asinh_params = transform_params["asinh_transform"]

        # Income should be asinh transformed
        assert not np.allclose(result["income"].values, data["income"].values)
        assert "income" in asinh_params

        # Age and wealth should NOT be transformed
        pd.testing.assert_series_equal(result["age"], data["age"])
        pd.testing.assert_series_equal(result["wealth"], data["wealth"])
        assert "age" not in asinh_params
        assert "wealth" not in asinh_params

    def test_preprocess_data_raises_error_for_asinh_and_log_overlap(self):
        """Test error when both asinh and log target same columns."""
        data = pd.DataFrame(
            {
                "age": [25.3, 30.7, 35.2, 40.9, 45.1],
                "income": [50123.45, 60987.23, 70456.78, 80234.56, 90876.12],
            }
        )

        with pytest.raises(
            ValueError, match="Cannot apply both .* to the same columns"
        ):
            preprocess_data(
                data,
                full_data=True,
                asinh_transform=["income"],
                log_transform=["income"],
            )

    def test_preprocess_data_raises_error_for_asinh_and_normalize_overlap(
        self,
    ):
        """Test error when both asinh and normalize target same columns."""
        data = pd.DataFrame(
            {
                "age": [25.3, 30.7, 35.2, 40.9, 45.1],
                "income": [50123.45, 60987.23, 70456.78, 80234.56, 90876.12],
            }
        )

        with pytest.raises(
            ValueError, match="Cannot apply both .* to the same columns"
        ):
            preprocess_data(
                data,
                full_data=True,
                asinh_transform=["income"],
                normalize=["income"],
            )
