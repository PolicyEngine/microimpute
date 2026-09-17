"""Independent dtype, weight and prediction-contract regressions for PR 219."""

import importlib

import joblib
import numpy as np
import optuna
import pandas as pd
import pytest
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold

from microimpute import (
    autoimpute,
    compare_metrics,
    cross_validate_model,
    get_imputations,
)
from microimpute.comparisons.metrics import quantile_loss
from microimpute.models import OLS, QRF, QuantReg
from microimpute.models.matching import Matching


@pytest.fixture(autouse=True)
def threaded_cv():
    with joblib.parallel_backend("threading", n_jobs=2):
        yield


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "int8", "int16"])
def test_integer_pinball_loss_matches_real_arithmetic(dtype):
    limits = np.iinfo(dtype)
    truth = np.array([limits.min, limits.max], dtype=dtype)
    prediction = truth[::-1].copy()
    # At q=.25, errors of +/- the full integer range have these exact losses.
    width = int(limits.max) - int(limits.min)
    np.testing.assert_allclose(
        quantile_loss(0.25, truth, prediction), [0.75 * width, 0.25 * width]
    )


@pytest.mark.parametrize("model_class", [OLS, QRF, QuantReg])
def test_unsigned_constant_comparison_uses_numeric_loss(model_class):
    donor = pd.DataFrame({"x": np.arange(20.0), "y": np.full(20, 3, dtype="uint8")})
    truth = pd.DataFrame({"x": [21.0, 22.0], "y": np.array([1, 5], dtype="uint8")})
    predictions = get_imputations([model_class], donor, truth, ["x"], ["y"], [0.5])
    scores = compare_metrics(truth, predictions, ["y"])
    assert set(scores.Metric) == {"quantile_loss"}
    # Both errors are two, so the median pinball loss is one.
    np.testing.assert_allclose(scores.Loss, 1.0)


def test_unsigned_cv_matches_float_control():
    donor = pd.DataFrame(
        {"x": np.arange(8.0), "y": np.array([3] * 7 + [1], dtype="uint8")}
    )
    unsigned = cross_validate_model(
        OLS, donor, ["x"], ["y"], n_splits=2, quantiles=[0.5]
    )
    floating = cross_validate_model(
        OLS, donor.astype({"y": float}), ["x"], ["y"], n_splits=2, quantiles=[0.5]
    )
    pd.testing.assert_frame_equal(
        unsigned["quantile_loss"]["results"], floating["quantile_loss"]["results"]
    )


def _nearest_donor(
    receiver, donor, matching_variables, z_variables, dist_fun="Manhattan", **kwargs
):
    train = donor[matching_variables].to_numpy(dtype=float)
    test = receiver[matching_variables].to_numpy(dtype=float)
    if dist_fun == "Gower":
        scale = np.ptp(train, axis=0)
        scale[scale == 0] = 1
        distances = cdist(test / scale, train / scale, metric="cityblock")
    else:
        metric = {
            "Manhattan": "cityblock",
            "Euclidean": "euclidean",
            "minimax": "chebyshev",
            "Mahalanobis": "mahalanobis",
        }[dist_fun]
        options = (
            {"VI": np.linalg.inv(np.cov(train, rowvar=False))}
            if metric == "mahalanobis"
            else {}
        )
        distances = cdist(test, train, metric=metric, **options)
    nearest = np.argmin(distances, axis=1)
    result = receiver.copy()
    for variable in z_variables:
        result[variable] = donor[variable].to_numpy()[nearest]
    return result, result


def test_matching_tunes_with_numeric_donor_error(monkeypatch):
    rng = np.random.default_rng(0)
    features = rng.normal(size=(45, 2)) * [1, 5]
    donor = pd.DataFrame(
        {
            "x": features[:, 0],
            "z": features[:, 1],
            "y": ((features[:, 0] + rng.normal(size=45)) > -0.8).astype("uint8"),
        }
    )
    studies = []
    create_study = optuna.create_study

    def capture_study(*args, **kwargs):
        study = create_study(*args, **kwargs)
        studies.append(study)
        return study

    monkeypatch.setattr(optuna, "create_study", capture_study)
    _, params = Matching(matching_hotdeck=_nearest_donor, seed=42).fit(
        donor, ["x", "z"], ["y"], tune_hyperparameters=True
    )
    oracle = {}
    for distance in {trial.params["dist_fun"] for trial in studies[0].trials}:
        fold_losses = []
        for train, test in KFold(n_splits=3, shuffle=True, random_state=42).split(
            donor
        ):
            training, held_out = donor.iloc[train], donor.iloc[test]
            predicted, _ = _nearest_donor(
                held_out.drop(columns="y"),
                training,
                ["x", "z"],
                ["y"],
                dist_fun=distance,
            )
            errors = [
                abs(int(actual) - int(estimate))
                for actual, estimate in zip(held_out.y, predicted.y)
            ]
            fold_losses.append(np.mean(errors) / training.y.std(ddof=0))
        oracle[distance] = np.mean(fold_losses)
    assert params["dist_fun"] == min(oracle, key=oracle.get)
    for trial in studies[0].trials:
        assert trial.value == pytest.approx(oracle[trial.params["dist_fun"]])


@pytest.fixture
def boolean_donor():
    rng = np.random.default_rng(57)
    return pd.DataFrame({"x": np.linspace(-2, 2, 90), "y": rng.uniform(size=90) < 0.4})


@pytest.mark.parametrize("dtype", ["bool", "boolean"])
def test_explicit_numeric_boolean_agrees_across_public_paths(boolean_donor, dtype):
    donor = boolean_donor.astype({"y": dtype})
    floating = donor.astype({"y": float})
    receiver = donor.iloc[[1, 17, 61]]
    quantiles = [0.1, 0.5, 0.9]
    direct = (
        OLS()
        .fit(donor, ["x"], ["y"], target_types={"y": "numeric"})
        .predict(receiver, quantiles)
    )
    generated = get_imputations(
        [OLS], donor, receiver, ["x"], ["y"], quantiles, target_types={"y": "numeric"}
    )
    control = get_imputations([OLS], floating, receiver, ["x"], ["y"], quantiles)
    for q in quantiles:
        pd.testing.assert_frame_equal(direct[q], control["OLS"][q])
        pd.testing.assert_frame_equal(generated["OLS"][q], control["OLS"][q])
    scores = compare_metrics(receiver, generated, ["y"], target_types={"y": "numeric"})
    expected = compare_metrics(receiver.astype({"y": float}), control, ["y"])
    pd.testing.assert_frame_equal(scores, expected)
    assert set(scores.Metric) == {"quantile_loss"}


@pytest.mark.parametrize("dtype", ["bool", "boolean"])
def test_explicit_numeric_boolean_cv_matches_float(boolean_donor, dtype):
    donor = boolean_donor.astype({"y": dtype})
    result = cross_validate_model(
        OLS,
        donor,
        ["x"],
        ["y"],
        n_splits=2,
        quantiles=[0.5],
        target_types={"y": "numeric"},
    )
    control = cross_validate_model(
        OLS, donor.astype({"y": float}), ["x"], ["y"], n_splits=2, quantiles=[0.5]
    )
    assert result["quantile_loss"]["variables"] == ["y"]
    assert result["log_loss"]["variables"] == []
    pd.testing.assert_frame_equal(
        result["quantile_loss"]["results"], control["quantile_loss"]["results"]
    )


@pytest.mark.parametrize("declared_type", [None, "categorical", "bool"])
def test_boolean_classification_and_probabilities_remain_available(
    boolean_donor, declared_type
):
    declaration = None if declared_type is None else {"y": declared_type}
    truth = boolean_donor.iloc[:6]
    predictions = get_imputations(
        [OLS], boolean_donor, truth, ["x"], ["y"], [0.5], target_types=declaration
    )
    probabilities = predictions["OLS"]["probabilities"]["y"]
    assert probabilities["probabilities"].shape == (6, 2)
    np.testing.assert_allclose(probabilities["probabilities"].sum(axis=1), 1)
    scores = compare_metrics(truth, predictions, ["y"], target_types=declaration)
    assert set(scores.Metric) == {"log_loss"}
    cv = cross_validate_model(
        OLS,
        boolean_donor,
        ["x"],
        ["y"],
        n_splits=2,
        quantiles=[0.5],
        target_types=declaration,
    )
    assert cv["log_loss"]["variables"] == ["y"]
    assert cv["quantile_loss"]["variables"] == []


def test_weight_predictor_normalization_preserves_exact_linear_oracle():
    donor = pd.DataFrame({"w": np.arange(1.0, 61.0), "y": 1 + 3 * np.arange(1.0, 61.0)})
    receiver = pd.DataFrame({"w": [61.0, 65.0]}, index=[101, 107])
    result = autoimpute(
        donor,
        receiver,
        ["w"],
        ["y"],
        weight_col="w",
        models=[OLS],
        preprocessing={"w": "normalize"},
        train_size=1,
        k_folds=2,
    )
    np.testing.assert_allclose(result.receiver_data.y, [184, 196])
    assert result.cv_results["OLS"]["quantile_loss"]["mean_test"] < 1e-10
    replay = result.fitted_models["best_method"].predict(receiver, [0.5])[0.5]
    pd.testing.assert_frame_equal(replay, result.imputations["best_method"])


@pytest.mark.parametrize("transform", ["normalize", "log", "asinh"])
def test_transformed_weight_predictor_matches_separate_weights_after_sampling_and_filtering(
    transform,
):
    rng = np.random.default_rng(123)
    donor = pd.DataFrame(
        {"w": np.arange(1.0, 61.0), "y": rng.normal(size=60) + np.arange(60.0) ** 0.5},
        index=np.arange(60) * 7 + 3,
    )
    receiver = pd.DataFrame({"w": [3.0, 25.0, 65.0]}, index=[901, 902, 903])
    settings = dict(
        predictors=["w"],
        imputed_variables=["y"],
        models=[OLS],
        preprocessing={"w": transform},
        train_size=0.8,
        random_state=11,
        k_folds=2,
        imputation_quantiles=[0.1, 0.5, 0.9],
        hyperparameters={
            "OLS": {"row_filter": pd.Series(donor.index % 3 != 0, index=donor.index)}
        },
    )
    result = autoimpute(donor, receiver, weight_col="w", **settings)
    control = autoimpute(
        donor.assign(weight=donor.w), receiver, weight_col="weight", **settings
    )
    for q in [0.1, 0.5, 0.9]:
        pd.testing.assert_frame_equal(
            result.imputations["best_method"][q], control.imputations["best_method"][q]
        )
    pd.testing.assert_frame_equal(
        result.cv_results["OLS"]["quantile_loss"]["results"],
        control.cv_results["OLS"]["quantile_loss"]["results"],
    )
    sampled = donor.sample(frac=0.8, random_state=11)
    selected = sampled.loc[sampled.index % 3 != 0]
    np.testing.assert_allclose(
        result.fitted_models["best_method"].models["y"].model.model.weights,
        selected.w / selected.w.mean(),
    )


def test_cv_tuning_refit_preserves_raw_weight_predictor(monkeypatch):
    cv_module = importlib.import_module("microimpute.evaluations.cross_validation")
    original_fit = cv_module._fit_model_for_fold
    seen = []
    donor = pd.DataFrame(
        {"w": np.arange(1.0, 31.0), "y": np.sin(np.arange(30.0))},
        index=np.arange(30) * 7,
    )

    def capture_fit(model, model_class, data, predictors, targets, weight_col, *args):
        weights = data[weight_col] if isinstance(weight_col, str) else weight_col
        np.testing.assert_array_equal(weights, donor.loc[data.index, "w"])
        assert abs(data.w.mean()) < 1e-12
        seen.append(len(data))
        return original_fit(
            model, model_class, data, predictors, targets, weight_col, *args
        )

    monkeypatch.setattr(cv_module, "_fit_model_for_fold", capture_fit)
    # Bound the unrelated parameter search while retaining real QRF weighted fits.
    monkeypatch.setattr(
        QRF,
        "_tune_hyperparameters",
        lambda self, **kwargs: {"n_estimators": 5, "min_samples_leaf": 2},
    )
    _, params = cross_validate_model(
        QRF,
        donor,
        ["w"],
        ["y"],
        weight_col="w",
        preprocessing={"w": "normalize"},
        n_splits=2,
        quantiles=[0.5],
        tune_hyperparameters=True,
    )
    assert params == {"n_estimators": 5, "min_samples_leaf": 2}
    assert sorted(seen) == [15, 15, 30]


@pytest.mark.parametrize("model_class", [OLS, QRF])
@pytest.mark.parametrize("value", ["a", True])
def test_default_constant_category_has_default_frame_and_point_mass(model_class, value):
    donor = pd.DataFrame({"x": np.arange(20.0), "y": [value] * 20})
    receiver = pd.DataFrame({"x": [21.0, 22.0]}, index=[101, 109])
    predictions = get_imputations(
        [model_class], donor, receiver, ["x"], ["y"], quantiles=None
    )[model_class.__name__]
    assert set(predictions) == {0.5, "probabilities"}
    pd.testing.assert_frame_equal(
        predictions[0.5], pd.DataFrame({"y": [value, value]}, index=receiver.index)
    )
    info = predictions["probabilities"]["y"]
    np.testing.assert_array_equal(info["classes"], [value])
    np.testing.assert_array_equal(info["probabilities"], np.ones((2, 1)))


@pytest.mark.parametrize("model_class", [OLS, QRF, QuantReg])
def test_default_numeric_constant_return_contract_is_unchanged(model_class):
    donor = pd.DataFrame({"x": np.arange(20.0), "y": [3.0] * 20})
    receiver = pd.DataFrame({"x": [21.0, 22.0]})
    predictions = get_imputations(
        [model_class], donor, receiver, ["x"], ["y"], quantiles=None
    )[model_class.__name__]
    assert isinstance(predictions, pd.DataFrame)
    np.testing.assert_array_equal(predictions.y, [3, 3])


@pytest.mark.parametrize("dtype", ["Int64", "UInt8", "Float64"])
def test_ols_nullable_numeric_receiver_matches_native_float(dtype):
    rng = np.random.default_rng(456)
    donor = pd.DataFrame(
        {
            "children": pd.Series(np.tile(np.arange(5), 12), dtype=dtype),
            "y": rng.normal(size=60) + np.tile(np.arange(5), 12) * 3,
        }
    )
    receiver = donor[["children"]].iloc[[1, 7, 19]].copy()
    receiver.index = [101, 103, 109]
    fitted = OLS().fit(donor, ["children"], ["y"])
    actual = fitted.predict(receiver, [0.1, 0.5, 0.9])
    control = (
        OLS()
        .fit(donor.astype({"children": float}), ["children"], ["y"])
        .predict(receiver.astype(float), [0.1, 0.5, 0.9])
    )
    for q in actual:
        pd.testing.assert_frame_equal(actual[q], control[q])
        pd.testing.assert_index_equal(actual[q].index, receiver.index)
    # A single/homogeneous receiver must still get the fitted intercept.
    one = fitted.predict(receiver.iloc[[0]], [0.5])[0.5]
    pd.testing.assert_frame_equal(one, actual[0.5].iloc[[0]])


@pytest.mark.parametrize("bad", [pd.NA, np.inf])
def test_ols_nullable_numeric_receiver_rejects_nonfinite(bad):
    donor = pd.DataFrame(
        {"x": np.arange(20.0), "y": np.arange(20.0) + np.sin(np.arange(20.0))}
    )
    fitted = OLS().fit(donor, ["x"], ["y"])
    with pytest.raises(ValueError, match="finite|missing|NaN"):
        fitted.predict(pd.DataFrame({"x": pd.Series([bad], dtype="Float64")}), [0.5])
