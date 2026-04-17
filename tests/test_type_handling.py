"""Tests for microimpute.utils.type_handling."""

import logging
import warnings

import numpy as np
import pandas as pd
import pytest

from microimpute.utils.type_handling import (
    DummyVariableProcessor,
    VariableTypeDetector,
)


# === Regression tests for #9: float {0.0, 1.0} misclassified as boolean ===


def test_is_boolean_variable_true_for_bool_dtype() -> None:
    s = pd.Series([True, False, True], dtype=bool)
    assert VariableTypeDetector.is_boolean_variable(s) is True


def test_is_boolean_variable_true_for_int_0_1() -> None:
    s = pd.Series([0, 1, 0, 1], dtype=int)
    assert VariableTypeDetector.is_boolean_variable(s) is True


def test_is_boolean_variable_false_for_float_0_1() -> None:
    """Regression test for #9: float column with values {0.0, 1.0} must
    NOT be treated as boolean. A probability, indicator, or small-sample
    feature that happens to contain only {0.0, 1.0} should stay numeric
    and be handed to the regressor, not silently routed to a classifier.
    """
    s = pd.Series([0.0, 1.0, 0.0, 1.0], dtype=float)
    assert VariableTypeDetector.is_boolean_variable(s) is False


def test_is_boolean_variable_false_for_float_probability() -> None:
    """A float probability column is definitely not boolean."""
    s = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
    assert VariableTypeDetector.is_boolean_variable(s) is False


# === Regression tests for #10: unseen categorical silently mapped to reference ===


def test_unseen_category_warns_at_test_time() -> None:
    """Regression test for #10: a category present at test time but not
    in training must emit a ``UserWarning``. Previously it silently
    collapsed to all-zero dummies equivalent to the dropped reference
    level, giving those rows the reference category's prediction
    without the caller noticing."""
    logger = logging.getLogger("test_unseen")
    processor = DummyVariableProcessor(logger)

    # Use non-equally-spaced numeric values so x isn't mis-detected as
    # numeric_categorical (which would trigger dummy encoding and an
    # extraneous unseen-category warning at test time).
    train = pd.DataFrame(
        {
            "x": [0.13, 0.27, 0.38, 0.44, 0.59, 0.71, 0.82, 0.91, 1.05, 1.2],
            "cat": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
            "y": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 1.2, 2.2, 3.2, 1.3],
        }
    )
    processor.preprocess_predictors(train, ["x", "cat"], ["y"])

    # Test data has level "Z" which is brand new.
    test = pd.DataFrame({"x": [0.9, 0.8], "cat": ["A", "Z"]})

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        processor.apply_dummy_encoding_to_test(test, ["x", "cat"])

    unseen_warnings = [x for x in w if "not present at training time" in str(x.message)]
    assert len(unseen_warnings) == 1, (
        f"Expected 1 unseen-category warning; got {len(w)} warnings: "
        f"{[str(x.message) for x in w]}"
    )
    assert "Z" in str(unseen_warnings[0].message)


def test_reference_level_does_not_trigger_warning() -> None:
    """The reference level (dropped via drop_first=True) is a valid
    training-time value and should NOT trigger the unseen-category
    warning — even though it has no dummy column of its own."""
    logger = logging.getLogger("test_reference")
    processor = DummyVariableProcessor(logger)

    # Use non-equally-spaced numeric values so x isn't mis-detected as
    # numeric_categorical (which would trigger dummy encoding and an
    # extraneous unseen-category warning at test time).
    train = pd.DataFrame(
        {
            "x": [0.13, 0.27, 0.38, 0.44, 0.59, 0.71, 0.82, 0.91, 1.05, 1.2],
            "cat": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
            "y": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 1.2, 2.2, 3.2, 1.3],
        }
    )
    processor.preprocess_predictors(train, ["x", "cat"], ["y"])

    # "A" is typically the reference level (alphabetical first dropped).
    test = pd.DataFrame({"x": [0.9, 0.8], "cat": ["A", "B"]})

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        processor.apply_dummy_encoding_to_test(test, ["x", "cat"])

    unseen_warnings = [x for x in w if "not present at training time" in str(x.message)]
    assert len(unseen_warnings) == 0, (
        "Reference level 'A' triggered a false-positive unseen-category warning: "
        f"{[str(x.message) for x in unseen_warnings]}"
    )


def test_all_known_levels_do_not_warn() -> None:
    """When test data only contains levels seen at training time, no
    warning should be emitted."""
    logger = logging.getLogger("test_known")
    processor = DummyVariableProcessor(logger)

    # Use non-equally-spaced numeric values so x isn't mis-detected as
    # numeric_categorical (which would trigger dummy encoding and an
    # extraneous unseen-category warning at test time).
    train = pd.DataFrame(
        {
            "x": [0.13, 0.27, 0.38, 0.44, 0.59, 0.71, 0.82, 0.91, 1.05, 1.2],
            "cat": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
            "y": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 1.2, 2.2, 3.2, 1.3],
        }
    )
    processor.preprocess_predictors(train, ["x", "cat"], ["y"])

    test = pd.DataFrame({"x": [0.9, 0.8, 0.7], "cat": ["A", "B", "C"]})

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        processor.apply_dummy_encoding_to_test(test, ["x", "cat"])

    unseen_warnings = [x for x in w if "not present at training time" in str(x.message)]
    assert unseen_warnings == []
