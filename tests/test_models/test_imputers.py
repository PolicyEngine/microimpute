"""
Comprehensive test module for the BaseImputer abstract class and its implementations.

This module tests the compatibility and interchangeability of different
imputer models through the common BaseImputer interface, including edge cases
and error handling.
"""

from typing import Type

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes

from microimpute.config import QUANTILES
from microimpute.models import *
from microimpute.utils.data import preprocess_data

# === Fixtures ===


@pytest.fixture
def diabetes_data() -> pd.DataFrame:
    """Create a dataset from the Diabetes dataset for testing."""
    diabetes = load_diabetes()
    data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1"]

    return data[predictors + imputed_variables]


@pytest.fixture
def simple_data() -> pd.DataFrame:
    """Create simple synthetic data for edge case testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "x1": np.random.randn(100),
            "x2": np.random.randn(100),
            "y": np.random.randn(100),
        }
    )


@pytest.fixture
def data_with_edge_cases() -> pd.DataFrame:
    """Create data with various edge cases."""
    np.random.seed(42)
    n_samples = 100

    return pd.DataFrame(
        {
            "numeric": np.random.randn(n_samples),
            "constant": np.ones(n_samples),  # Constant predictor
            "binary": np.random.choice([0, 1], n_samples),
            "categorical": np.random.choice(["A", "B", "C"], n_samples),
            "high_correlation": np.random.randn(
                n_samples
            ),  # Will be made correlated
            "target": np.random.randn(n_samples),
        }
    )


# Define all imputer model classes to test
ALL_IMPUTER_MODELS = [OLS, QuantReg, QRF]
CATEGORICAL_MODELS = [OLS, QRF]

try:
    from microimpute.models.matching import Matching

    ALL_IMPUTER_MODELS.append(Matching)
    CATEGORICAL_MODELS.append(Matching)
except ImportError:
    pass

try:
    from microimpute.models.mdn import MDN

    ALL_IMPUTER_MODELS.append(MDN)
    CATEGORICAL_MODELS.append(MDN)
except ImportError:
    pass


# === Basic Interface Tests ===


@pytest.mark.parametrize(
    "model_class", ALL_IMPUTER_MODELS, ids=lambda cls: cls.__name__
)
def test_init_signatures(model_class: Type[BaseImputer]) -> None:
    """Test that all models can be initialized without required arguments."""
    model = model_class()
    assert (
        model.predictors is None
    ), f"{model_class.__name__} should initialize predictors as None"
    assert (
        model.imputed_variables is None
    ), f"{model_class.__name__} should initialize imputed_variables as None"


@pytest.mark.parametrize(
    "model_class", ALL_IMPUTER_MODELS, ids=lambda cls: cls.__name__
)
def test_fit_predict_interface(
    model_class: Type[BaseImputer], diabetes_data: pd.DataFrame
) -> None:
    """Test the fit and predict methods for each model."""
    quantiles = QUANTILES
    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1"]

    X_train, X_test = preprocess_data(diabetes_data)

    # Initialize the model
    model = model_class()

    # Fit the model
    if model_class.__name__ == "QuantReg":
        fitted_model = model.fit(
            X_train, predictors, imputed_variables, quantiles=quantiles
        )
    else:
        fitted_model = model.fit(X_train, predictors, imputed_variables)

    # Predict with explicit quantiles
    predictions = fitted_model.predict(X_test, quantiles)

    # Check prediction format
    assert isinstance(
        predictions, dict
    ), f"{model_class.__name__} predict should return a dictionary"
    assert set(predictions.keys()).issubset(set(quantiles))

    # Check prediction shape
    for q, pred in predictions.items():
        assert pred.shape[0] == len(X_test)
        assert not pred.isna().any().any()


# === Data Type Handling Tests ===


@pytest.mark.parametrize(
    "model_class", ALL_IMPUTER_MODELS, ids=lambda cls: cls.__name__
)
def test_categorical_variables(model_class: Type[BaseImputer]) -> None:
    """Test that models handle categorical variables correctly."""
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "numeric": np.random.randn(100),
            "category": np.random.choice(["A", "B", "C"], 100),
            "target": np.random.randn(100),
        }
    )

    X_train, X_test = preprocess_data(data)

    model = model_class()

    if model_class.__name__ == "QuantReg":
        fitted = model.fit(
            X_train, ["numeric", "category"], ["target"], quantiles=[0.5]
        )
    else:
        fitted = model.fit(X_train, ["numeric", "category"], ["target"])

    predictions = fitted.predict(X_test, quantiles=[0.5])

    # When quantiles specified, returns dict
    assert isinstance(predictions, dict)
    assert 0.5 in predictions
    assert len(predictions[0.5]) == len(X_test)
    assert not predictions[0.5]["target"].isna().any()


@pytest.mark.parametrize(
    "model_class", ALL_IMPUTER_MODELS, ids=lambda cls: cls.__name__
)
def test_boolean_variables(model_class: Type[BaseImputer]) -> None:
    """Test that models handle boolean variables correctly."""
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "numeric": np.random.randn(100),
            "bool_var": np.random.choice([True, False], 100),
            "target": np.random.randn(100),
        }
    )

    X_train, X_test = preprocess_data(data)

    model = model_class()

    if model_class.__name__ == "QuantReg":
        fitted = model.fit(
            X_train, ["numeric", "bool_var"], ["target"], quantiles=[0.5]
        )
    else:
        fitted = model.fit(X_train, ["numeric", "bool_var"], ["target"])

    predictions = fitted.predict(X_test, quantiles=[0.5])

    # When quantiles specified, returns dict
    assert isinstance(predictions, dict)
    assert 0.5 in predictions
    assert not predictions[0.5]["target"].isna().any()


@pytest.mark.parametrize(
    "model_class", ALL_IMPUTER_MODELS, ids=lambda cls: cls.__name__
)
def test_imputation_bool_targets(
    model_class: Type[BaseImputer],
) -> None:
    """Test imputing boolean target variables."""
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

    # Add random boolean targets
    df["bool"] = np.random.choice([True, False], size=len(df))

    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["bool", "s1"]

    X_train, X_test = preprocess_data(df)

    model = model_class()
    fitted_model = model.fit(X_train, predictors, imputed_variables)
    predictions = fitted_model.predict(X_test)

    # Default behavior returns DataFrame directly
    assert isinstance(predictions, pd.DataFrame)
    assert predictions["bool"].dtype == "bool"
    assert not predictions["s1"].isna().any()


@pytest.mark.parametrize(
    "model_class", CATEGORICAL_MODELS, ids=lambda cls: cls.__name__
)
def test_imputation_categorical_targets(
    model_class: Type[BaseImputer],
) -> None:
    """Test imputing categorical target variables."""
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

    # Add random categorical targets
    df["categorical"] = np.random.choice(["one", "two", "three"], size=len(df))

    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["categorical"]

    X_train, X_test = preprocess_data(df)

    model = model_class()
    fitted_model = model.fit(X_train, predictors, imputed_variables)
    predictions = fitted_model.predict(X_test)

    # Default behavior returns DataFrame directly
    assert isinstance(predictions, pd.DataFrame)
    assert pd.api.types.is_string_dtype(predictions["categorical"])

    # Test probability predictions for models that support it
    if model_class.__name__ in ["OLS", "QRF", "Matching"]:
        # Get predictions with probabilities using quantiles
        # (this ensures consistent return format across models)
        predictions_with_probs = fitted_model.predict(
            X_test, quantiles=[0.5], return_probs=True
        )
        assert isinstance(predictions_with_probs, dict)
        assert 0.5 in predictions_with_probs
        assert "probabilities" in predictions_with_probs

        # Check that we still get the categorical predictions
        assert isinstance(predictions_with_probs[0.5], pd.DataFrame)
        assert pd.api.types.is_string_dtype(
            predictions_with_probs[0.5]["categorical"]
        )

        # Check probability format
        prob_info = predictions_with_probs["probabilities"]["categorical"]
        assert isinstance(prob_info, dict)
        assert "probabilities" in prob_info
        assert "classes" in prob_info

        probs = prob_info["probabilities"]
        classes = prob_info["classes"]

        # Check that we have probabilities as a numpy array
        assert isinstance(probs, np.ndarray)
        assert probs.shape[0] == len(X_test)  # One row per sample
        assert probs.shape[1] == 3  # Three categories

        # Check that classes contains the category labels
        assert len(classes) == 3
        assert set(classes) == {"one", "two", "three"}

        # Probabilities should sum to 1 for each row (within tolerance)
        row_sums = probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

        # All probabilities should be between 0 and 1
        assert (probs >= 0.0).all()
        assert (probs <= 1.0).all()


@pytest.mark.parametrize(
    "model_class", CATEGORICAL_MODELS, ids=lambda cls: cls.__name__
)
def test_categorical_return_probs_false(
    model_class: Type[BaseImputer],
) -> None:
    """Test that categorical imputation with return_probs=False returns DataFrame."""
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

    # Add random categorical targets
    df["categorical"] = np.random.choice(["A", "B", "C"], size=len(df))

    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["categorical"]

    X_train, X_test = preprocess_data(df)

    model = model_class()
    fitted_model = model.fit(X_train, predictors, imputed_variables)

    # Test 1: Default behavior (no return_probs, no quantiles) should return DataFrame
    predictions = fitted_model.predict(X_test)
    assert isinstance(predictions, pd.DataFrame)
    assert "categorical" in predictions.columns
    assert pd.api.types.is_string_dtype(predictions["categorical"])
    assert set(predictions["categorical"].unique()).issubset({"A", "B", "C"})

    # Test 2: Explicit return_probs=False with quantiles should return dict of DataFrames
    predictions_with_quantiles = fitted_model.predict(
        X_test, quantiles=[0.5], return_probs=False
    )
    assert isinstance(predictions_with_quantiles, dict)
    assert 0.5 in predictions_with_quantiles
    assert isinstance(predictions_with_quantiles[0.5], pd.DataFrame)
    assert "probabilities" not in predictions_with_quantiles

    # Test 3: return_probs=True should include probabilities
    predictions_with_probs = fitted_model.predict(
        X_test, quantiles=[0.5], return_probs=True
    )
    assert isinstance(predictions_with_probs, dict)
    assert "probabilities" in predictions_with_probs


# === Edge Cases and Error Handling ===


@pytest.mark.parametrize(
    "model_class", ALL_IMPUTER_MODELS, ids=lambda cls: cls.__name__
)
def test_single_predictor(
    model_class: Type[BaseImputer], simple_data: pd.DataFrame
) -> None:
    """Test models with only one predictor."""
    X_train, X_test = preprocess_data(simple_data)

    model = model_class()

    if model_class.__name__ == "QuantReg":
        fitted = model.fit(X_train, ["x1"], ["y"], quantiles=[0.5])
    else:
        fitted = model.fit(X_train, ["x1"], ["y"])

    predictions = fitted.predict(X_test, quantiles=[0.5])

    # When quantiles specified, returns dict
    assert isinstance(predictions, dict)
    assert 0.5 in predictions
    assert not predictions[0.5]["y"].isna().any()


@pytest.mark.parametrize(
    "model_class", ALL_IMPUTER_MODELS, ids=lambda cls: cls.__name__
)
def test_multiple_targets(
    model_class: Type[BaseImputer], diabetes_data: pd.DataFrame
) -> None:
    """Test models with multiple target variables."""
    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1", "s2", "s3"]

    # Get more columns from diabetes data
    diabetes = load_diabetes()
    full_data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    data = full_data[predictors + imputed_variables]

    X_train, X_test = preprocess_data(data)

    model = model_class()

    if model_class.__name__ == "QuantReg":
        fitted = model.fit(
            X_train, predictors, imputed_variables, quantiles=[0.5]
        )
    else:
        fitted = model.fit(X_train, predictors, imputed_variables)

    predictions = fitted.predict(X_test, quantiles=[0.5])

    # When quantiles specified, returns dict
    assert isinstance(predictions, dict)
    assert 0.5 in predictions
    assert predictions[0.5].shape[1] == len(imputed_variables)
    assert not predictions[0.5].isna().any().any()


@pytest.mark.parametrize(
    "model_class", ALL_IMPUTER_MODELS, ids=lambda cls: cls.__name__
)
def test_constant_predictor(model_class: Type[BaseImputer]) -> None:
    """Test models with a constant predictor (no variance)."""
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "x1": np.random.randn(100),
            "constant": np.ones(100),  # Constant predictor
            "y": np.random.randn(100),
        }
    )

    X_train, X_test = preprocess_data(data)

    model = model_class()

    # Models should handle constant predictors gracefully
    if model_class.__name__ == "QuantReg":
        fitted = model.fit(X_train, ["x1", "constant"], ["y"], quantiles=[0.5])
    else:
        fitted = model.fit(X_train, ["x1", "constant"], ["y"])

    predictions = fitted.predict(X_test, quantiles=[0.5])

    # When quantiles specified, returns dict
    assert isinstance(predictions, dict)
    assert 0.5 in predictions
    assert not predictions[0.5]["y"].isna().any()


@pytest.mark.parametrize(
    "model_class", ALL_IMPUTER_MODELS, ids=lambda cls: cls.__name__
)
def test_constant_target(model_class: Type[BaseImputer]) -> None:
    """Test models with a constant target variable."""
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "x1": np.random.randn(100),
            "x2": np.random.randn(100),
            "y": np.ones(100) * 100,  # Constant target
        }
    )

    X_train, X_test = preprocess_data(data)

    model = model_class()
    fitted_model = model.fit(X_train, ["x1", "x2"], ["y"])

    predictions = fitted_model.predict(X_test, quantiles=[0.1, 0.5, 0.9])

    # All predictions should be close to 100 (the constant value)
    for q in [0.1, 0.5, 0.9]:
        assert np.allclose(predictions[q]["y"].values, 100.0, rtol=0.1)


@pytest.mark.parametrize(
    "model_class", ALL_IMPUTER_MODELS, ids=lambda cls: cls.__name__
)
def test_highly_correlated_predictors(model_class: Type[BaseImputer]) -> None:
    """Test models with highly correlated predictors."""
    np.random.seed(42)
    n_samples = 100

    x1 = np.random.randn(n_samples)
    x2 = x1 + np.random.randn(n_samples) * 0.01  # Almost perfectly correlated

    data = pd.DataFrame(
        {"x1": x1, "x2": x2, "y": x1 + np.random.randn(n_samples) * 0.5}
    )

    X_train, X_test = preprocess_data(data)

    model = model_class()

    if model_class.__name__ == "QuantReg":
        fitted = model.fit(X_train, ["x1", "x2"], ["y"], quantiles=[0.5])
    else:
        fitted = model.fit(X_train, ["x1", "x2"], ["y"])

    predictions = fitted.predict(X_test, quantiles=[0.5])

    # When quantiles specified, returns dict
    assert isinstance(predictions, dict)
    assert 0.5 in predictions
    assert not predictions[0.5]["y"].isna().any()


# === Weighted Training Tests ===


@pytest.mark.parametrize(
    "model_class", ALL_IMPUTER_MODELS, ids=lambda cls: cls.__name__
)
def test_weighted_training(
    model_class: Type[BaseImputer], diabetes_data: pd.DataFrame
) -> None:
    """Ensure models can be trained using sampling weights."""
    X_train, _ = preprocess_data(diabetes_data)

    # Create a simple positive weight column
    X_train["wgt"] = range(1, len(X_train) + 1)

    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1"]

    model = model_class()

    # QuantReg and MDN don't support sample weights — they should raise
    # NotImplementedError rather than silently dropping weights.
    if model_class.__name__ in ("QuantReg", "MDN"):
        with pytest.raises(
            (NotImplementedError, RuntimeError),
            match="does not.*support.*weights|does not support sample weights",
        ):
            if model_class.__name__ == "QuantReg":
                model.fit(
                    X_train,
                    predictors,
                    imputed_variables,
                    weight_col="wgt",
                    quantiles=QUANTILES,
                )
            else:
                model.fit(
                    X_train,
                    predictors,
                    imputed_variables,
                    weight_col="wgt",
                )
        return

    fitted = model.fit(
        X_train, predictors, imputed_variables, weight_col="wgt"
    )

    assert fitted is not None

    # Test prediction
    X_test = X_train.drop(columns=["wgt"]).head(10)
    predictions = fitted.predict(X_test, quantiles=[0.5])

    # When quantiles specified, returns dict
    assert isinstance(predictions, dict)
    assert 0.5 in predictions
    assert not predictions[0.5].isna().any().any()


def test_imputer_rejects_nan_weights(diabetes_data: pd.DataFrame) -> None:
    """Regression test for the NaN-weight silent-corruption bug (#4): the
    imputer must raise a clear error when weights contain NaN values,
    rather than letting NaN propagate through .sample() or sample_weight.
    """
    X_train, _ = preprocess_data(diabetes_data)
    X_train["wgt"] = 1.0
    X_train.loc[X_train.index[0], "wgt"] = float("nan")

    model = OLS()
    with pytest.raises(ValueError, match="positive and finite|NaN"):
        model.fit(X_train, ["age", "sex"], ["bmi"], weight_col="wgt")


def test_imputer_rejects_zero_weights(diabetes_data: pd.DataFrame) -> None:
    """Regression test for the non-positive-weight bug (#4): weights of 0
    or negative values must raise a clear error."""
    X_train, _ = preprocess_data(diabetes_data)
    X_train["wgt"] = 1.0
    X_train.loc[X_train.index[0], "wgt"] = 0.0

    model = OLS()
    with pytest.raises(ValueError, match="positive and finite|positive"):
        model.fit(X_train, ["age", "sex"], ["bmi"], weight_col="wgt")


def test_weighted_fit_differs_from_unweighted(
    diabetes_data: pd.DataFrame,
) -> None:
    """Regression test for the weight-discard bug (#4): a truly weighted fit
    must produce different parameter estimates than an unweighted fit on
    an asymmetric dataset. Previously weights were used as bootstrap
    resample probabilities and not passed to the underlying estimator, so
    parameter estimates converged to the unweighted solution in
    expectation."""
    np.random.seed(0)
    n = 300

    # Asymmetric weights: first half gets weight 1, second half gets weight
    # 50. An unweighted OLS fit ignores this; a true WLS fit does not.
    x = np.linspace(-2, 2, n)
    # Introduce a slope shift in the second half so the WLS coefficient
    # should skew toward it.
    y = (
        np.where(np.arange(n) < n // 2, 1.0 * x, 5.0 * x)
        + np.random.randn(n) * 0.2
    )
    weights = np.where(np.arange(n) < n // 2, 1.0, 50.0)

    data = pd.DataFrame({"x": x, "y": y, "wgt": weights})

    unweighted_ols = OLS()
    unweighted_fit = unweighted_ols.fit(data, ["x"], ["y"])
    unweighted_pred = unweighted_fit.predict(
        data[["x"]].head(20), quantiles=[0.5]
    )[0.5]["y"].values

    weighted_ols = OLS()
    weighted_fit = weighted_ols.fit(data, ["x"], ["y"], weight_col="wgt")
    weighted_pred = weighted_fit.predict(
        data[["x"]].head(20), quantiles=[0.5]
    )[0.5]["y"].values

    # Weighted predictions should differ substantially from unweighted
    # ones when a large weight block has a different slope.
    assert not np.allclose(unweighted_pred, weighted_pred, atol=0.05), (
        "Weighted OLS fit should differ from unweighted fit on asymmetric "
        "data; previously weights were silently discarded"
    )


# === Quantile-Specific Tests ===


@pytest.mark.parametrize(
    "model_class", ALL_IMPUTER_MODELS, ids=lambda cls: cls.__name__
)
def test_extreme_quantiles(
    model_class: Type[BaseImputer], simple_data: pd.DataFrame
) -> None:
    """Test models with extreme quantile values."""
    X_train, X_test = preprocess_data(simple_data)

    extreme_quantiles = [0.01, 0.99]

    model = model_class()

    if model_class.__name__ == "QuantReg":
        fitted = model.fit(
            X_train, ["x1", "x2"], ["y"], quantiles=extreme_quantiles
        )
    else:
        fitted = model.fit(X_train, ["x1", "x2"], ["y"])

    predictions = fitted.predict(X_test, quantiles=extreme_quantiles)

    for q in extreme_quantiles:
        assert q in predictions
        assert not predictions[q]["y"].isna().any()


@pytest.mark.parametrize(
    "model_class", ALL_IMPUTER_MODELS, ids=lambda cls: cls.__name__
)
def test_single_quantile(
    model_class: Type[BaseImputer], simple_data: pd.DataFrame
) -> None:
    """Test models with a single quantile."""
    X_train, X_test = preprocess_data(simple_data)

    model = model_class()

    if model_class.__name__ == "QuantReg":
        fitted = model.fit(X_train, ["x1", "x2"], ["y"], quantiles=[0.5])
    else:
        fitted = model.fit(X_train, ["x1", "x2"], ["y"])

    predictions = fitted.predict(X_test, quantiles=[0.5])

    # When quantiles specified, returns dict
    assert isinstance(predictions, dict)
    assert 0.5 in predictions
    assert not predictions[0.5]["y"].isna().any()


# === Data Validation Tests ===


def test_string_column_validation() -> None:
    """Test that the _validate_data method handles string columns appropriately."""
    data = pd.DataFrame(
        {"numeric_col": [1, 2, 3], "string_col": ["a", "b", "c"]}
    )
    columns = ["numeric_col", "string_col"]

    model = OLS()

    # Preprocess will handle encoding
    data = preprocess_data(data, full_data=True)

    # Should not raise an error after preprocessing
    model._validate_data(data, columns)


@pytest.mark.parametrize(
    "model_class", ALL_IMPUTER_MODELS, ids=lambda cls: cls.__name__
)
def test_missing_predictors_in_test(model_class: Type[BaseImputer]) -> None:
    """Test behavior when test data is missing predictor columns."""
    np.random.seed(42)
    train_data = pd.DataFrame(
        {
            "x1": np.random.randn(50),
            "x2": np.random.randn(50),
            "y": np.random.randn(50),
        }
    )

    # Test data missing x2
    test_data = pd.DataFrame({"x1": np.random.randn(10)})

    model = model_class()

    if model_class.__name__ == "QuantReg":
        fitted = model.fit(train_data, ["x1", "x2"], ["y"], quantiles=[0.5])
    else:
        fitted = model.fit(train_data, ["x1", "x2"], ["y"])

    # Should raise an error when predictor is missing
    with pytest.raises(Exception):
        predictions = fitted.predict(test_data, quantiles=[0.5])


# === Reproducibility Tests ===


_REPRODUCIBILITY_MODELS = [OLS, QuantReg, QRF]
try:
    from microimpute.models.matching import Matching as _Matching_for_repro

    _REPRODUCIBILITY_MODELS.append(_Matching_for_repro)
except ImportError:
    pass


@pytest.mark.parametrize(
    "model_class",
    _REPRODUCIBILITY_MODELS,
    ids=lambda cls: cls.__name__,
)
def test_reproducibility(
    model_class: Type[BaseImputer], simple_data: pd.DataFrame
) -> None:
    # Note: MDN is excluded because PyTorch MPS (Apple Silicon) doesn't support
    # deterministic operations, making reproducibility tests unreliable.
    """Test that models produce reproducible results."""
    X_train, X_test = preprocess_data(simple_data)

    # First run
    model1 = model_class()
    fitted1 = model1.fit(X_train, ["x1", "x2"], ["y"])
    pred1 = fitted1.predict(X_test, quantiles=[0.5])

    # Second run with same data
    model2 = model_class()
    fitted2 = model2.fit(X_train, ["x1", "x2"], ["y"])
    pred2 = fitted2.predict(X_test, quantiles=[0.5])

    # Results should be very similar (allowing for minor numerical differences)
    # When quantiles specified, returns dict
    np.testing.assert_allclose(
        pred1[0.5]["y"].values, pred2[0.5]["y"].values, rtol=1e-5
    )


# === Performance and Memory Tests ===


@pytest.mark.parametrize(
    "model_class", ALL_IMPUTER_MODELS, ids=lambda cls: cls.__name__
)
def test_large_number_of_predictors(model_class: Type[BaseImputer]) -> None:
    """Test models with many predictors."""
    np.random.seed(42)
    n_samples = 50
    n_predictors = 20

    # Create data with many predictors
    data_dict = {
        f"x{i}": np.random.randn(n_samples) for i in range(n_predictors)
    }
    data_dict["y"] = np.random.randn(n_samples)
    data = pd.DataFrame(data_dict)

    X_train = data.iloc[:40].reset_index(drop=True)
    X_test = data.iloc[40:].reset_index(drop=True)

    predictors = [f"x{i}" for i in range(n_predictors)]

    model = model_class()

    if model_class.__name__ == "QuantReg":
        fitted = model.fit(X_train, predictors, ["y"], quantiles=[0.5])
    else:
        fitted = model.fit(X_train, predictors, ["y"])

    predictions = fitted.predict(X_test, quantiles=[0.5])

    # When quantiles specified, returns dict
    assert isinstance(predictions, dict)
    assert 0.5 in predictions
    assert not predictions[0.5]["y"].isna().any()
