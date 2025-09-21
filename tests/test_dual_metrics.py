"""Comprehensive tests for dual metric (quantile loss and log loss) functionality."""

import numpy as np
import pandas as pd
import pytest

from microimpute.comparisons import (
    compare_metrics,
    get_imputations,
)
from microimpute.comparisons.autoimpute import autoimpute
from microimpute.comparisons.autoimpute_helpers import (
    select_best_model_dual_metrics,
)
from microimpute.comparisons.metrics import (
    compute_loss,
    get_metric_for_variable_type,
    log_loss,
)
from microimpute.config import QUANTILES
from microimpute.evaluations.cross_validation import cross_validate_model
from microimpute.models import OLS, QRF, QuantReg

# Check if Matching is available
try:
    from microimpute.models import Matching

    HAS_MATCHING = True
except ImportError:
    HAS_MATCHING = False


# === Fixtures ===


@pytest.fixture
def mixed_type_data() -> pd.DataFrame:
    """Generate data with both numerical and categorical variables."""
    np.random.seed(42)
    n_samples = 200

    return pd.DataFrame(
        {
            # Numerical predictors
            "num_pred1": np.random.randn(n_samples),
            "num_pred2": np.random.randn(n_samples) * 2 + 1,
            # Categorical predictor
            "cat_pred": np.random.choice(["X", "Y", "Z"], size=n_samples),
            # Numerical targets
            "num_target1": np.random.randn(n_samples) * 3,
            "num_target2": np.random.randn(n_samples) + 5,
            # Categorical targets
            "binary_target": np.random.choice([0, 1], size=n_samples),
            "multiclass_target": np.random.choice([0, 1, 2], size=n_samples),
            "string_target": np.random.choice(["A", "B", "C"], size=n_samples),
        }
    )


@pytest.fixture
def split_mixed_data(mixed_type_data: pd.DataFrame) -> tuple:
    """Split mixed data into train and test sets."""
    train_size = int(0.8 * len(mixed_type_data))
    train_data = mixed_type_data[:train_size].copy()
    test_data = mixed_type_data[train_size:].copy()
    return train_data, test_data


# === Metric Detection Tests ===


def test_metric_detection_numerical() -> None:
    """Test that numerical variables are correctly identified."""
    # Continuous numerical data
    numerical_series = pd.Series(np.random.randn(100))
    assert (
        get_metric_for_variable_type(numerical_series, "num_var")
        == "quantile_loss"
    )

    # Integer numerical data with high cardinality
    int_series = pd.Series(np.random.randint(0, 100, size=100))
    assert (
        get_metric_for_variable_type(int_series, "int_var") == "quantile_loss"
    )


def test_metric_detection_categorical() -> None:
    """Test that categorical variables are correctly identified."""
    # Binary data
    binary_series = pd.Series([0, 1, 0, 1, 1, 0, 1, 0])
    assert (
        get_metric_for_variable_type(binary_series, "binary_var") == "log_loss"
    )

    # String categorical
    string_series = pd.Series(["A", "B", "C", "A", "B", "C"])
    assert (
        get_metric_for_variable_type(string_series, "string_var") == "log_loss"
    )

    # Low cardinality integer (categorical-like)
    low_card_series = pd.Series([0, 1, 2, 0, 1, 2, 0, 1, 2])
    assert (
        get_metric_for_variable_type(low_card_series, "low_card_var")
        == "log_loss"
    )

    # Boolean type
    bool_series = pd.Series([True, False, True, False, True])
    assert get_metric_for_variable_type(bool_series, "bool_var") == "log_loss"


# === Log Loss Function Tests ===


def test_log_loss_with_probabilities() -> None:
    """Test log loss computation with probability inputs."""
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.7])

    loss = log_loss(y_true, y_pred_proba)
    assert loss > 0  # Log loss should be positive
    assert loss < 1  # Should be reasonable for good predictions


def test_log_loss_with_class_labels() -> None:
    """Test log loss computation when class labels are provided instead of probabilities."""
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred_labels = np.array([0, 1, 1, 1, 0])  # Class predictions

    # Should convert to probabilities with a warning
    loss = log_loss(y_true, y_pred_labels)
    assert loss > 0
    # Loss should be higher since we're using high-confidence probabilities
    assert loss > 1


def test_log_loss_multiclass() -> None:
    """Test log loss with multiclass data."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    # Provide class predictions (should be converted)
    y_pred_classes = np.array([0, 1, 2, 1, 1, 2])

    loss = log_loss(y_true, y_pred_classes)
    assert loss > 0


# === Compute Loss Tests ===


def test_compute_loss_quantile() -> None:
    """Test compute_loss with quantile loss metric."""
    y_true = np.random.randn(50)
    y_pred = y_true + np.random.randn(50) * 0.1  # Add small noise

    losses, mean_loss = compute_loss(y_true, y_pred, "quantile_loss", q=0.5)
    assert len(losses) == len(y_true)
    assert mean_loss > 0
    assert mean_loss == np.mean(losses)


def test_compute_loss_log() -> None:
    """Test compute_loss with log loss metric."""
    y_true = np.random.choice([0, 1], size=50)
    y_pred = np.random.choice([0, 1], size=50)

    losses, mean_loss = compute_loss(
        y_true, y_pred, "log_loss", q=0.5, labels=np.array([0, 1])
    )
    assert len(losses) == len(y_true)
    assert mean_loss > 0
    # For log loss, all elements should be the same (it's a global metric)
    assert np.allclose(losses, losses[0])


# === Compare Metrics Tests ===


def test_compare_metrics_mixed_types(split_mixed_data: tuple) -> None:
    """Test compare_metrics with mixed variable types."""
    train_data, test_data = split_mixed_data
    predictors = ["num_pred1", "num_pred2"]
    mixed_targets = ["num_target1", "binary_target"]

    # Get imputations
    model_classes = [OLS]
    method_imputations = get_imputations(
        model_classes, train_data, test_data, predictors, mixed_targets
    )

    # Get true values for comparison
    Y_test = test_data[mixed_targets]

    # Compare metrics
    results_df = compare_metrics(Y_test, method_imputations, mixed_targets)

    # Check structure
    assert "Method" in results_df.columns
    assert "Imputed Variable" in results_df.columns
    assert "Metric" in results_df.columns
    assert "Loss" in results_df.columns
    assert "Percentile" in results_df.columns

    # Check both metrics are present
    metrics_used = results_df["Metric"].unique()
    assert "quantile_loss" in metrics_used
    assert "log_loss" in metrics_used

    # Check correct metric assignment
    num_target_metrics = results_df[
        results_df["Imputed Variable"] == "num_target1"
    ]["Metric"].unique()
    assert len(num_target_metrics) == 1
    assert num_target_metrics[0] == "quantile_loss"

    binary_target_metrics = results_df[
        results_df["Imputed Variable"] == "binary_target"
    ]["Metric"].unique()
    assert len(binary_target_metrics) == 1
    assert binary_target_metrics[0] == "log_loss"

    # Check separate averaging
    mean_vars = results_df[results_df["Percentile"] == "mean_loss"][
        "Imputed Variable"
    ].unique()
    assert "mean_quantile_loss" in mean_vars
    assert "mean_log_loss" in mean_vars


def test_compare_metrics_all_numerical(split_mixed_data: tuple) -> None:
    """Test compare_metrics with only numerical variables."""
    train_data, test_data = split_mixed_data
    predictors = ["num_pred1", "num_pred2"]
    numerical_targets = ["num_target1", "num_target2"]

    model_classes = [OLS]
    method_imputations = get_imputations(
        model_classes, train_data, test_data, predictors, numerical_targets
    )

    Y_test = test_data[numerical_targets]
    results_df = compare_metrics(Y_test, method_imputations, numerical_targets)

    # Should only have quantile loss
    assert all(results_df["Metric"].isin(["quantile_loss"]))


def test_compare_metrics_all_categorical(split_mixed_data: tuple) -> None:
    """Test compare_metrics with only categorical variables."""
    train_data, test_data = split_mixed_data
    predictors = ["num_pred1", "num_pred2"]
    categorical_targets = ["binary_target", "string_target"]

    model_classes = [OLS]
    method_imputations = get_imputations(
        model_classes, train_data, test_data, predictors, categorical_targets
    )

    Y_test = test_data[categorical_targets]
    results_df = compare_metrics(
        Y_test, method_imputations, categorical_targets
    )

    # Should only have log loss
    assert all(results_df["Metric"].isin(["log_loss"]))


# === Cross-Validation Dual Metrics Tests ===


def test_cross_validation_dual_metrics(mixed_type_data: pd.DataFrame) -> None:
    """Test cross-validation with dual metric support."""
    predictors = ["num_pred1", "num_pred2"]
    mixed_targets = ["num_target1", "binary_target"]

    cv_results = cross_validate_model(
        model_class=OLS,
        data=mixed_type_data,
        predictors=predictors,
        imputed_variables=mixed_targets,
        n_splits=3,
        random_state=42,
    )

    # Check structure
    assert isinstance(cv_results, dict)
    assert "quantile_loss" in cv_results
    assert "log_loss" in cv_results

    # Check quantile loss results
    ql_results = cv_results["quantile_loss"]
    assert "mean_train" in ql_results
    assert "mean_test" in ql_results
    assert "variables" in ql_results
    assert "num_target1" in ql_results["variables"]
    assert isinstance(ql_results["results"], pd.DataFrame)
    assert "train" in ql_results["results"].index
    assert "test" in ql_results["results"].index

    # Check log loss results
    ll_results = cv_results["log_loss"]
    assert "results" in ll_results  # Single DataFrame with train/test rows
    assert "mean_train" in ll_results
    assert "mean_test" in ll_results
    assert "variables" in ll_results
    assert "binary_target" in ll_results["variables"]

    # Mean values should be reasonable
    assert 0 <= ql_results["mean_test"] < float("inf")
    assert 0 <= ll_results["mean_test"] < float("inf")


def test_cross_validation_with_hyperparameter_tuning(
    mixed_type_data: pd.DataFrame,
) -> None:
    """Test cross-validation with hyperparameter tuning returns proper dual metrics."""
    predictors = ["num_pred1", "num_pred2"]
    mixed_targets = ["num_target1", "binary_target"]

    cv_results = cross_validate_model(
        model_class=QRF,
        data=mixed_type_data,
        predictors=predictors,
        imputed_variables=mixed_targets,
        n_splits=2,
        random_state=42,
        tune_hyperparameters=True,
    )

    # Should return tuple with hyperparameters
    results, best_params = cv_results
    assert isinstance(results, dict)
    assert "quantile_loss" in results
    assert "log_loss" in results
    assert best_params is not None


# === Model Selection Tests ===


def test_select_best_model_auto_priority() -> None:
    """Test model selection with auto (rank-based) priority."""
    # Mock results for multiple models
    method_results = {
        "OLS": {
            "quantile_loss": {"mean_test": 2.5, "variables": ["var1", "var2"]},
            "log_loss": {"mean_test": 0.8, "variables": ["var3"]},
        },
        "QRF": {
            "quantile_loss": {"mean_test": 2.0, "variables": ["var1", "var2"]},
            "log_loss": {"mean_test": 0.9, "variables": ["var3"]},
        },
        "QuantReg": {
            "quantile_loss": {"mean_test": 2.3, "variables": ["var1", "var2"]},
            "log_loss": {"mean_test": 0.7, "variables": ["var3"]},
        },
    }

    best_model, metrics = select_best_model_dual_metrics(
        method_results, metric_priority="auto"
    )

    # QRF should win overall (best at quantile loss, which has more variables)
    assert best_model in ["QRF", "QuantReg"]  # Depending on weighted ranking
    assert "quantile_loss" in metrics
    assert "log_loss" in metrics


def test_select_best_model_numerical_priority() -> None:
    """Test model selection with numerical priority."""
    method_results = {
        "OLS": {
            "quantile_loss": {"mean_test": 2.5, "variables": ["var1"]},
            "log_loss": {"mean_test": 0.3, "variables": ["var2", "var3"]},
        },
        "QRF": {
            "quantile_loss": {"mean_test": 2.0, "variables": ["var1"]},
            "log_loss": {"mean_test": 1.5, "variables": ["var2", "var3"]},
        },
    }

    best_model, metrics = select_best_model_dual_metrics(
        method_results, metric_priority="numerical"
    )

    # QRF should win (best quantile loss)
    assert best_model == "QRF"
    assert metrics["quantile_loss"] == 2.0


def test_select_best_model_categorical_priority() -> None:
    """Test model selection with categorical priority."""
    method_results = {
        "OLS": {
            "quantile_loss": {"mean_test": 1.0, "variables": ["var1", "var2"]},
            "log_loss": {"mean_test": 0.5, "variables": ["var3"]},
        },
        "QRF": {
            "quantile_loss": {"mean_test": 3.0, "variables": ["var1", "var2"]},
            "log_loss": {"mean_test": 0.3, "variables": ["var3"]},
        },
    }

    best_model, metrics = select_best_model_dual_metrics(
        method_results, metric_priority="categorical"
    )

    # QRF should win (best log loss)
    assert best_model == "QRF"
    assert metrics["log_loss"] == 0.3


def test_select_best_model_with_nan_metrics() -> None:
    """Test model selection handles NaN metrics correctly."""
    method_results = {
        "OLS": {
            "quantile_loss": {"mean_test": 2.5, "variables": ["var1"]},
            "log_loss": {"mean_test": np.nan, "variables": []},
        },
        "QRF": {
            "quantile_loss": {"mean_test": np.nan, "variables": []},
            "log_loss": {"mean_test": 0.5, "variables": ["var2"]},
        },
    }

    # Should handle NaN values gracefully
    best_model, metrics = select_best_model_dual_metrics(
        method_results, metric_priority="auto"
    )

    assert best_model in ["OLS", "QRF"]


# === AutoImpute Integration Tests ===


def test_autoimpute_with_metric_priority_auto(
    mixed_type_data: pd.DataFrame,
) -> None:
    """Test autoimpute with auto metric priority."""
    # Split data
    donor_data = mixed_type_data[:150].copy()
    receiver_data = mixed_type_data[150:].copy()

    predictors = ["num_pred1", "num_pred2"]
    mixed_targets = ["num_target1", "binary_target"]

    # Remove targets from receiver
    for target in mixed_targets:
        if target in receiver_data.columns:
            del receiver_data[target]

    result = autoimpute(
        donor_data=donor_data,
        receiver_data=receiver_data,
        predictors=predictors,
        imputed_variables=mixed_targets,
        models=[OLS, QuantReg],
        metric_priority="auto",
        k_folds=2,
        random_state=42,
        log_level="WARNING",
    )

    # Check results
    assert result.imputations is not None
    assert result.cv_results is not None
    assert isinstance(result.cv_results, dict)

    # Check that both metrics are in CV results
    for model in result.cv_results.keys():
        model_results = result.cv_results[model]
        assert "quantile_loss" in model_results
        assert "log_loss" in model_results

    # Check receiver data has imputed values
    for target in mixed_targets:
        assert target in result.receiver_data.columns


def test_autoimpute_all_numerical_variables(
    mixed_type_data: pd.DataFrame,
) -> None:
    """Test autoimpute with only numerical variables."""
    donor_data = mixed_type_data[:150].copy()
    receiver_data = mixed_type_data[150:].copy()

    predictors = ["num_pred1", "num_pred2"]
    numerical_targets = ["num_target1", "num_target2"]

    for target in numerical_targets:
        if target in receiver_data.columns:
            del receiver_data[target]

    result = autoimpute(
        donor_data=donor_data,
        receiver_data=receiver_data,
        predictors=predictors,
        imputed_variables=numerical_targets,
        models=[OLS, QRF],
        metric_priority="auto",
        k_folds=2,
        random_state=42,
        log_level="WARNING",
    )

    # Should only use quantile loss
    for model in result.cv_results.keys():
        model_results = result.cv_results[model]
        assert len(model_results["quantile_loss"]["variables"]) == 2
        assert len(model_results["log_loss"]["variables"]) == 0


def test_autoimpute_all_categorical_variables(
    mixed_type_data: pd.DataFrame,
) -> None:
    """Test autoimpute with only categorical variables."""
    donor_data = mixed_type_data[:150].copy()
    receiver_data = mixed_type_data[150:].copy()

    predictors = ["num_pred1", "num_pred2"]
    categorical_targets = ["binary_target", "string_target"]

    for target in categorical_targets:
        if target in receiver_data.columns:
            del receiver_data[target]

    result = autoimpute(
        donor_data=donor_data,
        receiver_data=receiver_data,
        predictors=predictors,
        imputed_variables=categorical_targets,
        models=[OLS],
        metric_priority="auto",
        k_folds=2,
        random_state=42,
        log_level="WARNING",
    )

    # Should only use log loss
    for model in result.cv_results.keys():
        model_results = result.cv_results[model]
        assert len(model_results["quantile_loss"]["variables"]) == 0
        assert len(model_results["log_loss"]["variables"]) == 2


# === Edge Cases and Error Handling ===


def test_log_loss_constant_across_quantiles(split_mixed_data: tuple) -> None:
    """Test that log loss doesn't vary with quantile."""
    train_data, test_data = split_mixed_data
    predictors = ["num_pred1", "num_pred2"]
    categorical_targets = ["binary_target"]

    model_classes = [OLS]
    method_imputations = get_imputations(
        model_classes, train_data, test_data, predictors, categorical_targets
    )

    Y_test = test_data[categorical_targets]
    results_df = compare_metrics(
        Y_test, method_imputations, categorical_targets
    )

    # Filter to log loss results for the categorical variable
    log_loss_results = results_df[
        (results_df["Metric"] == "log_loss")
        & (results_df["Imputed Variable"] == "binary_target")
    ]

    # Get losses at different quantiles
    losses_by_quantile = {}
    for q in QUANTILES:
        q_loss = log_loss_results[log_loss_results["Percentile"] == q][
            "Loss"
        ].values
        if len(q_loss) > 0:
            losses_by_quantile[q] = q_loss[0]

    # All quantiles should have the same log loss
    if len(losses_by_quantile) > 1:
        loss_values = list(losses_by_quantile.values())
        assert np.allclose(
            loss_values, loss_values[0], rtol=1e-10
        ), "Log loss should be constant across quantiles"


def test_empty_variable_lists() -> None:
    """Test handling of empty variable lists in model selection."""
    method_results = {
        "OLS": {
            "quantile_loss": {"mean_test": np.nan, "variables": []},
            "log_loss": {"mean_test": np.nan, "variables": []},
        }
    }

    # Should raise an error when no variables to evaluate with 'auto'
    with pytest.raises(
        ValueError, match="No variables compatible with any model"
    ):
        select_best_model_dual_metrics(method_results, metric_priority="auto")

    # Should raise error with 'numerical' priority
    with pytest.raises(ValueError, match="No numerical variables found"):
        select_best_model_dual_metrics(
            method_results, metric_priority="numerical"
        )

    # Should raise error with 'categorical' priority
    with pytest.raises(ValueError, match="No categorical variables found"):
        select_best_model_dual_metrics(
            method_results, metric_priority="categorical"
        )

    # Should raise error with 'combined' priority
    with pytest.raises(
        ValueError, match="No variables available for evaluation"
    ):
        select_best_model_dual_metrics(
            method_results, metric_priority="combined"
        )


def test_quantreg_with_numerical_only(split_mixed_data: tuple) -> None:
    """Test that QuantReg works correctly with only numerical variables."""
    train_data, test_data = split_mixed_data
    predictors = ["num_pred1", "num_pred2"]
    numerical_targets = ["num_target1", "num_target2"]

    # QuantReg should work fine with numerical targets
    model_classes = [QuantReg]
    method_imputations = get_imputations(
        model_classes, train_data, test_data, predictors, numerical_targets
    )

    Y_test = test_data[numerical_targets]
    results_df = compare_metrics(Y_test, method_imputations, numerical_targets)

    # Should only have quantile loss results
    assert all(results_df["Metric"].isin(["quantile_loss"]))
    assert len(results_df) > 0


def test_quantreg_fails_with_categorical(
    mixed_type_data: pd.DataFrame,
) -> None:
    """Test that QuantReg is handled gracefully with categorical variables."""
    predictors = ["num_pred1", "num_pred2"]
    categorical_targets = ["binary_target", "string_target"]

    # Try to use QuantReg with categorical targets - should return empty results
    cv_results = cross_validate_model(
        model_class=QuantReg,
        data=mixed_type_data,
        predictors=predictors,
        imputed_variables=categorical_targets,
        n_splits=2,
        random_state=42,
    )

    # Should return NaN results since QuantReg can't handle categorical
    assert cv_results["quantile_loss"]["mean_test"] == np.nan or np.isnan(
        cv_results["quantile_loss"]["mean_test"]
    )
    assert cv_results["log_loss"]["mean_test"] == np.nan or np.isnan(
        cv_results["log_loss"]["mean_test"]
    )
    assert len(cv_results["quantile_loss"]["variables"]) == 0
    assert len(cv_results["log_loss"]["variables"]) == 0


def test_autoimpute_with_all_models(mixed_type_data: pd.DataFrame) -> None:
    """Test autoimpute with all available models."""
    donor_data = mixed_type_data[:100].copy()
    receiver_data = mixed_type_data[100:120].copy()

    predictors = ["num_pred1", "num_pred2"]
    mixed_targets = ["num_target1", "binary_target"]

    for target in mixed_targets:
        if target in receiver_data.columns:
            del receiver_data[target]

    models = [OLS, QRF, QuantReg]
    if HAS_MATCHING:
        models.append(Matching)

    result = autoimpute(
        donor_data=donor_data,
        receiver_data=receiver_data,
        predictors=predictors,
        imputed_variables=mixed_targets,
        models=models,
        metric_priority="auto",
        k_folds=2,
        random_state=42,
        log_level="WARNING",
    )

    # Check all models were evaluated
    assert len(result.cv_results) == len(models)
    for model in models:
        assert model.__name__ in result.cv_results
