"""Comprehensive tests for dual metric (quantile loss and log loss) functionality."""

import numpy as np
import pandas as pd
import pytest

from microimpute.comparisons import compare_metrics, get_imputations
from microimpute.comparisons.autoimpute import autoimpute
from microimpute.comparisons.autoimpute_helpers import (
    select_best_model_dual_metrics,
)
from microimpute.comparisons.metrics import (
    compare_distributions,
    compute_loss,
    get_metric_for_variable_type,
    kl_divergence,
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


# === Categorical Probability Handling Tests ===


def test_categorical_probabilities_in_cross_validation() -> None:
    """Test that cross-validation properly uses probabilities for categorical log loss."""
    # Create synthetic data with categorical target
    np.random.seed(42)
    n_samples = 200

    # Create features
    X1 = np.random.randn(n_samples)
    X2 = np.random.randn(n_samples)

    # Create categorical target with 3 classes
    # Make it somewhat predictable based on X1
    y_prob = 1 / (1 + np.exp(-X1))  # Logistic function
    y_cat = np.where(y_prob < 0.33, "A", np.where(y_prob < 0.66, "B", "C"))

    # Create DataFrame
    df = pd.DataFrame({"x1": X1, "x2": X2, "cat_target": y_cat})

    # Run cross-validation
    results = cross_validate_model(
        model_class=OLS,
        data=df,
        predictors=["x1", "x2"],
        imputed_variables=["cat_target"],
        quantiles=[0.5],
        n_splits=3,
        random_state=42,
    )

    # Check that we have log_loss results (not quantile_loss) for categorical variable
    assert "log_loss" in results
    assert results["log_loss"] is not None
    assert "results" in results["log_loss"]

    # Check that log loss values are reasonable (not the dummy 0.99/0.01 values)
    # When using actual probabilities, log loss should typically be < 1.0 for reasonable models
    # When using dummy probabilities (0.99/0.01), log loss is usually > 2.0
    test_loss = results["log_loss"]["mean_test"]

    # This threshold distinguishes between using real probabilities vs dummy ones
    # Real probabilities should give lower log loss
    assert (
        test_loss < 2.0
    ), f"Log loss {test_loss} suggests dummy probabilities are being used instead of real ones"


def test_probability_ordering() -> None:
    """Test that probabilities are ordered alphabetically to match sklearn's log_loss expectation."""
    from microimpute.comparisons.metrics import (
        order_probabilities_alphabetically,
    )

    # Create test data with known probabilities
    np.random.seed(42)

    # True labels
    y_true = np.array(["B", "A", "C", "A", "B", "C"])

    # Create probability matrix with columns in non-alphabetical order
    # Columns: C, B, A (wrong order)
    probs_wrong_order = np.array(
        [
            [0.2, 0.7, 0.1],  # True: B, so B should have high prob
            [0.1, 0.2, 0.7],  # True: A, so A should have high prob
            [0.8, 0.1, 0.1],  # True: C, so C should have high prob
            [0.1, 0.1, 0.8],  # True: A, so A should have high prob
            [0.1, 0.8, 0.1],  # True: B, so B should have high prob
            [0.9, 0.05, 0.05],  # True: C, so C should have high prob
        ]
    )

    # If we don't reorder, log loss will be wrong
    labels_wrong = np.array(["C", "B", "A"])
    _, loss_wrong = compute_loss(
        y_true, probs_wrong_order, "log_loss", labels=labels_wrong
    )

    # Correct alphabetical order: A, B, C
    probs_correct_order, alphabetical_labels = (
        order_probabilities_alphabetically(probs_wrong_order, labels_wrong)
    )
    _, loss_correct = compute_loss(
        y_true, probs_correct_order, "log_loss", labels=alphabetical_labels
    )

    # The correctly ordered probabilities should give much lower loss
    assert (
        loss_correct < loss_wrong
    ), "Alphabetical ordering of probabilities is not working correctly"

    # Check that labels are alphabetically ordered
    assert list(alphabetical_labels) == sorted(alphabetical_labels)


def test_ols_returns_probabilities_for_categorical() -> None:
    """Test that OLS model returns probabilities when asked for categorical variables."""
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100

    df = pd.DataFrame(
        {
            "x1": np.random.randn(n_samples),
            "x2": np.random.randn(n_samples),
            "cat_target": np.random.choice(["X", "Y", "Z"], n_samples),
        }
    )

    # Split data
    train_data = df[:80]
    test_data = df[80:]

    # Fit OLS model
    model = OLS()
    fitted = model.fit(
        train_data, predictors=["x1", "x2"], imputed_variables=["cat_target"]
    )

    # Predict with return_probs=True
    predictions = fitted.predict(test_data, quantiles=[0.5], return_probs=True)

    # Check that probabilities are returned
    assert (
        "probabilities" in predictions
    ), "Model should return probabilities when return_probs=True"
    assert (
        "cat_target" in predictions["probabilities"]
    ), "Probabilities should include categorical variable"

    # Check probability structure
    prob_info = predictions["probabilities"]["cat_target"]
    assert isinstance(
        prob_info, dict
    ), "Probability info should be a dictionary"
    assert "probabilities" in prob_info, "Should contain probabilities array"
    assert "classes" in prob_info, "Should contain classes array"

    probs = prob_info["probabilities"]
    classes = prob_info["classes"]

    # Check shapes
    assert probs.shape[0] == len(
        test_data
    ), "Should have probabilities for each test sample"
    assert probs.shape[1] == len(
        np.unique(df["cat_target"])
    ), "Should have probability for each class"
    assert len(classes) == len(
        np.unique(df["cat_target"])
    ), "Should have all classes"

    # Check that probabilities sum to 1
    prob_sums = probs.sum(axis=1)
    np.testing.assert_allclose(
        prob_sums, 1.0, rtol=1e-5, err_msg="Probabilities should sum to 1"
    )


def test_qrf_returns_probabilities_for_categorical() -> None:
    """Test that QRF model returns probabilities when asked for categorical variables."""
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100

    df = pd.DataFrame(
        {
            "x1": np.random.randn(n_samples),
            "x2": np.random.randn(n_samples),
            "cat_target": np.random.choice(
                ["Apple", "Banana", "Cherry"], n_samples
            ),
        }
    )

    # Split data
    train_data = df[:80]
    test_data = df[80:]

    # Fit QRF model
    model = QRF()
    fitted = model.fit(
        train_data, predictors=["x1", "x2"], imputed_variables=["cat_target"]
    )

    # Predict with return_probs=True
    predictions = fitted.predict(test_data, quantiles=[0.5], return_probs=True)

    # Check that probabilities are returned
    assert (
        "probabilities" in predictions
    ), "Model should return probabilities when return_probs=True"
    assert (
        "cat_target" in predictions["probabilities"]
    ), "Probabilities should include categorical variable"

    # Check probability structure
    prob_info = predictions["probabilities"]["cat_target"]
    assert isinstance(
        prob_info, dict
    ), "Probability info should be a dictionary"
    assert "probabilities" in prob_info, "Should contain probabilities array"
    assert "classes" in prob_info, "Should contain classes array"

    probs = prob_info["probabilities"]
    classes = prob_info["classes"]

    # Check that we have the original string labels, not encoded values
    assert all(
        isinstance(c, str) for c in classes
    ), "Classes should be original string labels"
    assert set(classes) == set(
        df["cat_target"].unique()
    ), "Should have all original class labels"

    # Check shapes
    assert probs.shape[0] == len(
        test_data
    ), "Should have probabilities for each test sample"
    assert probs.shape[1] == len(
        classes
    ), "Should have probability for each class"

    # Check that probabilities sum to 1
    prob_sums = probs.sum(axis=1)
    np.testing.assert_allclose(
        prob_sums, 1.0, rtol=1e-5, err_msg="Probabilities should sum to 1"
    )


def test_probability_ordering_with_real_model() -> None:
    """Test that probability ordering works correctly with real model output."""
    from microimpute.comparisons.metrics import (
        order_probabilities_alphabetically,
    )

    np.random.seed(42)
    n_samples = 50

    # Create data where class C is most likely, then B, then A
    X = np.random.randn(n_samples, 2)
    y_true = ["C"] * 25 + ["B"] * 15 + ["A"] * 10

    # Shuffle the data
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y_true = [y_true[i] for i in indices]

    df = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "target": y_true})

    # Split data
    train_df = df[:40]
    test_df = df[40:]

    # Fit model
    model = OLS()
    fitted = model.fit(
        train_df, predictors=["x1", "x2"], imputed_variables=["target"]
    )

    # Get predictions with probabilities
    predictions = fitted.predict(test_df, quantiles=[0.5], return_probs=True)

    if (
        "probabilities" in predictions
        and "target" in predictions["probabilities"]
    ):
        prob_info = predictions["probabilities"]["target"]
        probs = prob_info["probabilities"]
        model_classes = prob_info["classes"]

        # Test ordering function
        probs_ordered, alphabetical_labels = (
            order_probabilities_alphabetically(probs, model_classes)
        )

        # Check that labels are alphabetical
        assert list(alphabetical_labels) == sorted(
            alphabetical_labels
        ), "Labels should be alphabetically ordered"

        # Compute log loss with ordered probabilities
        y_test = test_df["target"].values

        # Test with correctly ordered probabilities
        _, loss_ordered = compute_loss(
            y_test, probs_ordered, "log_loss", labels=alphabetical_labels
        )

        # The loss should be reasonable (not NaN or infinite)
        assert not np.isnan(loss_ordered), "Log loss should not be NaN"
        assert not np.isinf(loss_ordered), "Log loss should not be infinite"
        assert loss_ordered > 0, "Log loss should be positive"

        # Check if this is better than using dummy probabilities
        # With dummy probabilities (converting class predictions to 0.99/0.01)
        class_preds = predictions[0.5]["target"].values
        _, loss_dummy = compute_loss(y_test, class_preds, "log_loss")

        # Real probabilities should give better (lower) loss than dummy probabilities
        assert (
            loss_ordered < loss_dummy
        ), "Real probabilities should give better loss than dummy probabilities"


# === Distribution Comparison Tests ===


def test_kl_divergence_identical() -> None:
    """Test KL divergence between identical distributions."""
    values = np.array(["A", "B", "C", "A", "B", "C"])
    kl = kl_divergence(values, values)
    assert np.isclose(
        kl, 0.0, atol=1e-10
    ), "KL divergence should be 0 for identical distributions"


def test_kl_divergence_disjoint() -> None:
    """Test KL divergence between completely disjoint distributions."""
    donor = np.array(["A", "A", "A", "A"])
    receiver = np.array(["B", "B", "B", "B"])
    kl = kl_divergence(donor, receiver)
    # KL divergence should be very large (approaching infinity) for disjoint distributions
    assert (
        kl > 10
    ), "KL divergence should be very large for disjoint distributions"


def test_kl_divergence_partial_overlap() -> None:
    """Test KL divergence with partial distribution overlap."""
    # Donor: 75% A, 25% B
    donor = np.array(["A", "A", "A", "B"])
    # Receiver: 50% A, 50% B
    receiver = np.array(["A", "A", "B", "B"])
    kl = kl_divergence(donor, receiver)
    # KL(P||Q) = 0.75*log(0.75/0.50) + 0.25*log(0.25/0.50)
    #          = 0.75*log(1.5) + 0.25*log(0.5)
    #          ≈ 0.75*0.405 + 0.25*(-0.693)
    #          ≈ 0.304 - 0.173 ≈ 0.131
    assert (
        kl > 0
    ), "KL divergence should be positive for different distributions"
    assert (
        kl < 1
    ), f"KL divergence should be reasonable for similar distributions, got {kl}"


def test_kl_divergence_different_categories() -> None:
    """Test KL divergence when distributions have different category sets."""
    donor = np.array(["A", "B", "A", "B"])
    receiver = np.array(["B", "C", "B", "C"])
    kl = kl_divergence(donor, receiver)
    # Donor: A=0.5, B=0.5, C=0.0
    # Receiver: A=0.0+ε, B=0.5, C=0.5  (ε added to avoid log(0))
    # KL divergence should be large due to A having 0 probability in receiver
    assert (
        kl > 5
    ), f"KL divergence should be large when categories are missing, got {kl}"


def test_kl_divergence_empty_input() -> None:
    """Test KL divergence raises error with empty inputs."""
    with pytest.raises(ValueError, match="non-empty"):
        kl_divergence(np.array([]), np.array(["A"]))

    with pytest.raises(ValueError, match="non-empty"):
        kl_divergence(np.array(["A"]), np.array([]))


def test_compare_distributions_numerical_only() -> None:
    """Test compare_distributions with only numerical variables."""
    np.random.seed(42)

    # Create donor data
    donor = pd.DataFrame(
        {
            "income": np.random.normal(50000, 10000, 100),
            "age": np.random.normal(40, 10, 100),
        }
    )

    # Create receiver data with similar but slightly different distribution
    receiver = pd.DataFrame(
        {
            "income": np.random.normal(52000, 10000, 100),
            "age": np.random.normal(41, 10, 100),
        }
    )

    results = compare_distributions(donor, receiver, ["income", "age"])

    # Check structure
    assert len(results) == 2
    assert all(results["Metric"] == "wasserstein_distance")
    assert set(results["Variable"]) == {"income", "age"}

    # Check that distances are positive
    assert all(results["Distance"] >= 0)


def test_compare_distributions_categorical_only() -> None:
    """Test compare_distributions with only categorical variables."""
    np.random.seed(42)

    # Create donor data
    donor = pd.DataFrame(
        {
            "region": np.random.choice(
                ["North", "South", "East", "West"], 100
            ),
            "status": np.random.choice(["Active", "Inactive"], 100),
        }
    )

    # Create receiver data with different distribution
    receiver = pd.DataFrame(
        {
            "region": np.random.choice(
                ["North", "South", "East", "West"], 100, p=[0.4, 0.3, 0.2, 0.1]
            ),
            "status": np.random.choice(
                ["Active", "Inactive"], 100, p=[0.7, 0.3]
            ),
        }
    )

    results = compare_distributions(donor, receiver, ["region", "status"])

    # Check structure
    assert len(results) == 2
    assert all(results["Metric"] == "kl_divergence")
    assert set(results["Variable"]) == {"region", "status"}

    # KL divergence should be non-negative (can be > 1)
    assert all(results["Distance"] >= 0)


def test_compare_distributions_mixed_types() -> None:
    """Test compare_distributions with mixed numerical and categorical variables."""
    np.random.seed(42)

    donor = pd.DataFrame(
        {
            "income": np.random.normal(50000, 10000, 100),
            "region": np.random.choice(["A", "B", "C"], 100),
            "age": np.random.normal(40, 10, 100),
            "employed": np.random.choice([True, False], 100),
        }
    )

    receiver = pd.DataFrame(
        {
            "income": np.random.normal(51000, 10000, 100),
            "region": np.random.choice(
                ["A", "B", "C"], 100, p=[0.5, 0.3, 0.2]
            ),
            "age": np.random.normal(40.5, 10, 100),
            "employed": np.random.choice([True, False], 100, p=[0.8, 0.2]),
        }
    )

    results = compare_distributions(
        donor, receiver, ["income", "region", "age", "employed"]
    )

    # Check structure
    assert len(results) == 4

    # Check correct metric assignment
    numerical_vars = results[results["Metric"] == "wasserstein_distance"][
        "Variable"
    ].tolist()
    categorical_vars = results[results["Metric"] == "kl_divergence"][
        "Variable"
    ].tolist()

    assert "income" in numerical_vars
    assert "age" in numerical_vars
    assert "region" in categorical_vars
    assert "employed" in categorical_vars


def test_compare_distributions_identical() -> None:
    """Test that identical distributions give zero distance."""
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "x": np.random.randn(50),
            "y": np.random.choice(["A", "B"], 50),
        }
    )

    results = compare_distributions(data, data, ["x", "y"])

    # Both distances should be very close to 0
    assert all(results["Distance"] < 1e-10)


def test_compare_distributions_missing_columns() -> None:
    """Test error handling for missing columns."""
    donor = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    receiver = pd.DataFrame({"a": [1, 2, 3], "c": [7, 8, 9]})

    # Should raise error for missing column in receiver
    with pytest.raises(ValueError, match="Missing columns"):
        compare_distributions(donor, receiver, ["a", "b"])

    # Should raise error for missing column in donor
    with pytest.raises(ValueError, match="Missing columns"):
        compare_distributions(donor, receiver, ["c"])


def test_compare_distributions_rejects_nulls_in_data() -> None:
    """Test that compare_distributions raises error when data contains nulls."""
    np.random.seed(42)

    donor_with_nulls = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, np.nan, np.nan],
            "y": ["A", "B", "A", "B", "A", None, "B"],
        }
    )

    receiver_ok = pd.DataFrame(
        {
            "x": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            "y": ["A", "A", "B", "B", "B", "A"],
        }
    )

    # Should raise error for nulls in donor
    with pytest.raises(ValueError, match="contains null values"):
        compare_distributions(donor_with_nulls, receiver_ok, ["x", "y"])


def test_compare_distributions_empty_data() -> None:
    """Test error when variables have no data."""
    donor = pd.DataFrame(
        {
            "x": pd.Series([], dtype=float),
            "y": pd.Series([], dtype=str),
        }
    )

    receiver = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": ["A", "B", "C"],
        }
    )

    # Should raise error when no valid comparisons can be computed
    with pytest.raises(ValueError, match="No valid distribution comparisons"):
        compare_distributions(donor, receiver, ["x", "y"])


def test_compare_distributions_return_format() -> None:
    """Test that compare_distributions returns correctly formatted DataFrame."""
    np.random.seed(42)

    donor = pd.DataFrame(
        {
            "num": np.random.randn(50),
            "cat": np.random.choice(["X", "Y"], 50),
        }
    )

    receiver = pd.DataFrame(
        {
            "num": np.random.randn(50),
            "cat": np.random.choice(["X", "Y"], 50),
        }
    )

    results = compare_distributions(donor, receiver, ["num", "cat"])

    # Check DataFrame structure
    assert isinstance(results, pd.DataFrame)
    assert list(results.columns) == ["Variable", "Metric", "Distance"]
    assert len(results) == 2

    # Check data types
    assert results["Variable"].dtype == "object"
    assert results["Metric"].dtype == "object"
    assert results["Distance"].dtype in ["float64", "float32"]


# === Weighted Distribution Comparison Tests ===


def test_kl_divergence_with_weights() -> None:
    """Test KL divergence with sample weights."""
    # Create data where weights matter
    donor = np.array(["A", "A", "B", "B"])
    receiver = np.array(["A", "A", "B", "B"])

    # Without weights: both distributions are 50% A, 50% B
    kl_unweighted = kl_divergence(donor, receiver)
    assert np.isclose(
        kl_unweighted, 0.0, atol=1e-10
    ), "Unweighted identical distributions should have KL=0"

    # With weights: donor becomes 80% A, 20% B; receiver stays 50% A, 50% B
    donor_weights = np.array(
        [4.0, 4.0, 1.0, 1.0]
    )  # A weighted 8, B weighted 2
    kl_weighted_donor = kl_divergence(
        donor, receiver, donor_weights=donor_weights
    )
    assert (
        kl_weighted_donor > 0
    ), "Weighted donor with different distribution should have KL > 0"

    # With receiver weights: donor 50/50, receiver becomes 80% A, 20% B
    receiver_weights = np.array([4.0, 4.0, 1.0, 1.0])
    kl_weighted_receiver = kl_divergence(
        donor, receiver, receiver_weights=receiver_weights
    )
    assert (
        kl_weighted_receiver > 0
    ), "Weighted receiver with different distribution should have KL > 0"


def test_kl_divergence_weights_symmetry() -> None:
    """Test that KL divergence with swapped weighted distributions gives same result."""
    donor = np.array(["A", "A", "B", "B"])
    receiver = np.array(["A", "A", "B", "B"])

    # Make donor 80% A via weights
    donor_weights = np.array([4.0, 4.0, 1.0, 1.0])

    # Make receiver 80% A via weights
    receiver_weights = np.array([4.0, 4.0, 1.0, 1.0])

    # Both weighted the same way should give KL = 0
    kl = kl_divergence(
        donor,
        receiver,
        donor_weights=donor_weights,
        receiver_weights=receiver_weights,
    )
    assert np.isclose(
        kl, 0.0, atol=1e-10
    ), "Identically weighted distributions should have KL=0"


def test_compare_distributions_with_weights() -> None:
    """Test compare_distributions with weight arrays."""
    np.random.seed(42)

    donor = pd.DataFrame(
        {
            "income": np.random.normal(50000, 10000, 100),
            "region": np.random.choice(["A", "B", "C"], 100),
        }
    )
    donor_weights = np.random.uniform(0.5, 2.0, 100)

    receiver = pd.DataFrame(
        {
            "income": np.random.normal(52000, 10000, 100),
            "region": np.random.choice(["A", "B", "C"], 100),
        }
    )
    receiver_weights = np.random.uniform(0.5, 2.0, 100)

    # Unweighted comparison
    results_unweighted = compare_distributions(
        donor, receiver, ["income", "region"]
    )

    # Weighted comparison
    results_weighted = compare_distributions(
        donor,
        receiver,
        ["income", "region"],
        donor_weights=donor_weights,
        receiver_weights=receiver_weights,
    )

    # Both should return valid results
    assert len(results_unweighted) == 2
    assert len(results_weighted) == 2

    # Results should be different (weights should affect the computation)
    # Get income distances
    income_unweighted = results_unweighted[
        results_unweighted["Variable"] == "income"
    ]["Distance"].values[0]
    income_weighted = results_weighted[
        results_weighted["Variable"] == "income"
    ]["Distance"].values[0]

    # With random weights, results should typically differ
    # (though not guaranteed, so we just check they're both valid)
    assert income_unweighted >= 0
    assert income_weighted >= 0


def test_compare_distributions_donor_weight_only() -> None:
    """Test compare_distributions with only donor weights."""
    np.random.seed(42)

    donor = pd.DataFrame(
        {
            "x": np.random.normal(0, 1, 50),
        }
    )
    donor_weights = np.random.uniform(1, 3, 50)

    receiver = pd.DataFrame(
        {
            "x": np.random.normal(0.5, 1, 50),
        }
    )

    # Should work with only donor weights
    results = compare_distributions(
        donor, receiver, ["x"], donor_weights=donor_weights
    )

    assert len(results) == 1
    assert results["Variable"].values[0] == "x"
    assert results["Distance"].values[0] >= 0


def test_compare_distributions_receiver_weight_only() -> None:
    """Test compare_distributions with only receiver weights."""
    np.random.seed(42)

    donor = pd.DataFrame(
        {
            "x": np.random.normal(0, 1, 50),
        }
    )

    receiver = pd.DataFrame(
        {
            "x": np.random.normal(0.5, 1, 50),
        }
    )
    receiver_weights = np.random.uniform(1, 3, 50)

    # Should work with only receiver weights
    results = compare_distributions(
        donor, receiver, ["x"], receiver_weights=receiver_weights
    )

    assert len(results) == 1
    assert results["Variable"].values[0] == "x"
    assert results["Distance"].values[0] >= 0


def test_compare_distributions_rejects_nulls() -> None:
    """Test that compare_distributions raises error when data contains nulls."""
    donor_with_null = pd.DataFrame(
        {
            "x": [1.0, 2.0, np.nan, 4.0, 5.0],
        }
    )
    donor_ok = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    receiver_with_null = pd.DataFrame(
        {
            "x": [1.5, np.nan, 3.5, 4.5, 5.5],
        }
    )
    receiver_ok = pd.DataFrame(
        {
            "x": [1.5, 2.5, 3.5, 4.5, 5.5],
        }
    )

    # Should raise error for null in donor
    with pytest.raises(ValueError, match="donor_data contains null values"):
        compare_distributions(donor_with_null, receiver_ok, ["x"])

    # Should raise error for null in receiver
    with pytest.raises(ValueError, match="receiver_data contains null values"):
        compare_distributions(donor_ok, receiver_with_null, ["x"])


def test_compare_distributions_weights_affect_wasserstein() -> None:
    """Test that weights actually affect Wasserstein distance calculation."""
    # Create two identical value arrays
    donor = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
        }
    )
    donor_weights = np.array([1.0, 1.0, 1.0, 1.0])

    receiver = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
        }
    )
    receiver_weights = np.array([1.0, 1.0, 1.0, 1.0])

    # Identical data and weights should give distance = 0
    results_identical = compare_distributions(
        donor,
        receiver,
        ["x"],
        donor_weights=donor_weights,
        receiver_weights=receiver_weights,
    )
    assert np.isclose(
        results_identical["Distance"].values[0], 0.0, atol=1e-10
    ), "Identical weighted distributions should have distance=0"

    # Now change receiver weights to shift distribution toward higher values
    receiver_shifted_weights = np.array(
        [0.1, 0.1, 1.0, 1.0]
    )  # More weight on higher values

    results_shifted = compare_distributions(
        donor,
        receiver,
        ["x"],
        donor_weights=donor_weights,
        receiver_weights=receiver_shifted_weights,
    )
    assert (
        results_shifted["Distance"].values[0] > 0
    ), "Different weighted distributions should have distance > 0"


def test_compare_distributions_weights_affect_kl() -> None:
    """Test that weights actually affect KL divergence calculation."""
    # Create identical categorical arrays
    donor = pd.DataFrame(
        {
            "cat": ["A", "A", "B", "B"],
        }
    )
    donor_weights = np.array([1.0, 1.0, 1.0, 1.0])

    receiver = pd.DataFrame(
        {
            "cat": ["A", "A", "B", "B"],
        }
    )
    receiver_weights = np.array([1.0, 1.0, 1.0, 1.0])

    # Identical data and weights should give KL = 0
    results_identical = compare_distributions(
        donor,
        receiver,
        ["cat"],
        donor_weights=donor_weights,
        receiver_weights=receiver_weights,
    )
    assert np.isclose(
        results_identical["Distance"].values[0], 0.0, atol=1e-10
    ), "Identical weighted distributions should have KL=0"

    # Now change weights to create different distributions
    # Donor: 50% A, 50% B (equal weights)
    # Receiver: 90% A, 10% B (by weights)
    receiver_shifted_weights = np.array(
        [4.5, 4.5, 0.5, 0.5]
    )  # 90% weight on A

    results_shifted = compare_distributions(
        donor,
        receiver,
        ["cat"],
        donor_weights=donor_weights,
        receiver_weights=receiver_shifted_weights,
    )
    assert (
        results_shifted["Distance"].values[0] > 0
    ), "Different weighted distributions should have KL > 0"


def test_compare_distributions_weight_length_mismatch() -> None:
    """Test error handling for weight length mismatch."""
    donor = pd.DataFrame({"x": [1, 2, 3]})
    receiver = pd.DataFrame({"x": [1, 2, 3]})

    # Should raise error for mismatched donor weights length
    with pytest.raises(ValueError, match="donor_weights length"):
        compare_distributions(
            donor,
            receiver,
            ["x"],
            donor_weights=np.array([1.0, 2.0]),  # Wrong length
        )

    # Should raise error for mismatched receiver weights length
    with pytest.raises(ValueError, match="receiver_weights length"):
        compare_distributions(
            donor,
            receiver,
            ["x"],
            receiver_weights=np.array([1.0, 2.0]),  # Wrong length
        )


def test_compare_distributions_with_series_weights() -> None:
    """Test that compare_distributions works with pandas Series as weights."""
    donor = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    donor_weights = pd.Series([1.0, 2.0, 3.0, 4.0])

    receiver = pd.DataFrame({"x": [1.5, 2.5, 3.5, 4.5]})
    receiver_weights = pd.Series([1.0, 1.0, 1.0, 1.0])

    # Should work with Series weights
    results = compare_distributions(
        donor,
        receiver,
        ["x"],
        donor_weights=donor_weights,
        receiver_weights=receiver_weights,
    )

    assert len(results) == 1
    assert results["Distance"].values[0] >= 0
