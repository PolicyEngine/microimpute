"""Behavioral regressions for publication audit issues #202, #206, #209, #213."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss as sklearn_log_loss
from sklearn.model_selection import train_test_split

from microimpute.comparisons.autoimpute_helpers import prepare_data_for_imputation
from microimpute.comparisons.metrics import compare_metrics, compute_loss
from microimpute.comparisons.validation import validate_imputation_inputs
from microimpute.models import OLS
from microimpute.utils.data import preprocess_data


def test_receiver_uses_donor_normalization():
    donor = pd.DataFrame({"x": np.arange(20.0), "y": np.arange(20.0) * 2})
    receiver = pd.DataFrame({"x": [30.0]})
    train, test, _ = prepare_data_for_imputation(
        donor, receiver, ["x"], ["y"], None, 1.0, 0.0, preprocessing={"x": "normalize"}
    )
    assert test.x.iloc[0] == pytest.approx((30 - donor.x.mean()) / donor.x.std())
    assert train.x.mean() == pytest.approx(0)


def test_preprocess_split_fits_training_statistics_only():
    data = pd.DataFrame({"x": np.arange(30.0) ** 2})
    train, test, params = preprocess_data(data, normalize=["x"], random_state=3)
    raw_train, raw_test = train_test_split(
        data, train_size=0.8, test_size=0.2, random_state=3
    )
    assert params["normalization"]["x"]["mean"] == pytest.approx(raw_train.x.mean())
    np.testing.assert_allclose(
        test.x, (raw_test.x - raw_train.x.mean()) / raw_train.x.std()
    )


def test_count_target_stays_numeric_and_explicit_override_is_available():
    donor = pd.DataFrame({"x": np.arange(60.0), "y": np.tile(np.arange(6), 10)})
    model = OLS()
    model.fit(donor, ["x"], ["y"])
    assert model.numeric_targets == ["y"]
    model.fit(donor, ["x"], ["y"], target_types={"y": "categorical"})
    assert "y" in model.categorical_targets
    assert model.numeric_targets == []


def test_logloss_requires_real_probabilities():
    with pytest.raises(ValueError, match="probabilit"):
        compute_loss(np.array([0, 1]), np.array([0, 1]), "log_loss")
    actual = compute_loss(np.array([0, 1]), np.array([0.25, 0.75]), "log_loss")[1]
    assert actual == pytest.approx(-np.log(0.75))


def test_compare_metrics_uses_probabilities_and_custom_quantiles():
    y = pd.DataFrame({"label": ["a", "b"], "count": [2, 4]})
    probabilities = np.array([[0.8, 0.2], [0.3, 0.7]])
    predictions = {
        "OLS": {
            0.25: pd.DataFrame({"label": ["a", "b"], "count": [1, 3]}),
            "probabilities": {
                "label": {
                    "probabilities": probabilities,
                    "classes": np.array(["a", "b"]),
                }
            },
        }
    }
    result = compare_metrics(y, predictions, ["label", "count"])
    assert result.loc[result["Imputed Variable"] == "label", "Loss"].iloc[
        0
    ] == pytest.approx(sklearn_log_loss(y.label, probabilities))
    assert (
        result.loc[result["Imputed Variable"] == "count", "Percentile"].iloc[0] == 0.25
    )


@pytest.mark.parametrize(
    "predictors,targets,receiver,message",
    [
        (["x", "x"], ["y"], pd.DataFrame({"x": [1]}), "Duplicate"),
        (["x"], ["x"], pd.DataFrame({"x": [1]}), "overlap"),
        (["x"], ["y"], pd.DataFrame({"x": ["1"]}), "dtype"),
    ],
)
def test_invalid_imputation_inputs_are_explicit(predictors, targets, receiver, message):
    with pytest.raises(ValueError, match=message):
        validate_imputation_inputs(
            pd.DataFrame({"x": [1.0, 2.0], "y": [3, 4]}), receiver, predictors, targets
        )


def test_cv_preprocessing_uses_each_fold_and_scores_original_scale():
    from microimpute.evaluations.cross_validation import cross_validate_model

    data = pd.DataFrame({"x": np.arange(60.0), "y": 10 + np.arange(60.0) * 3})
    result = cross_validate_model(
        OLS,
        data,
        ["x"],
        ["y"],
        n_splits=3,
        quantiles=[0.5],
        preprocessing={"x": "normalize", "y": "normalize"},
    )
    assert result["quantile_loss"]["mean_test"] < 1e-10


def test_autoimpute_train_fraction_seed_custom_quantiles_and_mixed_transform():
    from microimpute import autoimpute

    data = pd.DataFrame({"x": np.arange(40.0), "y": 4 + np.arange(40.0) * 2})
    result = autoimpute(
        data,
        pd.DataFrame({"x": [50.0]}),
        ["x"],
        ["y"],
        models=[OLS],
        train_size=0.5,
        random_state=19,
        k_folds=2,
        imputation_quantiles=[0.25, 0.75],
        preprocessing={"x": "normalize", "y": "normalize"},
    )
    fitted = result.fitted_models["best_method"]
    assert fitted.models["y"].model.nobs == 20
    assert fitted.seed == 19
    assert {0.25, 0.5, 0.75} <= set(result.imputations["best_method"])
    assert result.receiver_data.y.iloc[0] == pytest.approx(104)


def test_cv_refits_tuning_on_all_rows_instead_of_selecting_test_fold(monkeypatch):
    import importlib
    import joblib

    cv = importlib.import_module("microimpute.evaluations.cross_validation")
    calls = []

    class ProbeResults:
        def predict(self, data, quantiles, **kwargs):
            return {
                q: pd.DataFrame({"y": np.zeros(len(data))}, index=data.index)
                for q in quantiles
            }

    def fake_fit(model, model_class, data, *args):
        calls.append(data.index.to_list())
        return ProbeResults(), {"training_rows": len(data)}

    monkeypatch.setattr(cv, "_fit_model_for_fold", fake_fit)
    data = pd.DataFrame({"x": np.arange(30.0), "y": np.arange(30.0)})
    with joblib.parallel_backend("threading"):
        _, params = cv.cross_validate_model(
            OLS,
            data,
            ["x"],
            ["y"],
            n_splits=3,
            quantiles=[0.25],
            tune_hyperparameters=True,
        )
    assert params == {"training_rows": 30}
    assert sorted(map(len, calls)) == [20, 20, 20, 30]


def test_explicit_categorical_target_is_stable_through_autoimpute():
    from microimpute import autoimpute

    data = pd.DataFrame(
        {"x": np.tile([0.2, 1.2, 2.2], 20), "y": np.tile([0, 1, 2], 20)}
    )
    result = autoimpute(
        data,
        data[["x"]].iloc[:3],
        ["x"],
        ["y"],
        models=[OLS],
        k_folds=2,
        target_types={"y": "categorical"},
        train_size=1,
    )
    assert result.cv_results["OLS"]["log_loss"]["variables"] == ["y"]
    assert result.cv_results["OLS"]["quantile_loss"]["variables"] == []


def test_predict_accepts_int_float_equivalence_and_rejects_strings():
    data = pd.DataFrame({"x": np.arange(20.0), "y": np.arange(20.0) * 2})
    fitted = OLS().fit(data, ["x"], ["y"])
    assert fitted.predict(pd.DataFrame({"x": [3]}), [0.5])[0.5].y.iloc[
        0
    ] == pytest.approx(6)
    with pytest.raises(ValueError, match="dtype"):
        fitted.predict(pd.DataFrame({"x": ["3"]}), [0.5])


def test_logloss_scores_constant_and_unseen_classes_without_fabricating_probs():
    probabilities = np.ones((2, 1))
    assert (
        compute_loss(
            np.array(["a", "a"]), probabilities, "log_loss", labels=np.array(["a"])
        )[1]
        == 0
    )
    actual = compute_loss(
        np.array(["a", "b"]), probabilities, "log_loss", labels=np.array(["a"])
    )[1]
    expected = sklearn_log_loss(
        ["a", "b"], np.array([[1.0, 0.0], [1.0, 0.0]]), labels=["a", "b"]
    )
    assert actual == pytest.approx(expected)


def test_returned_fitted_model_replays_donor_preprocessing_on_raw_receivers():
    from microimpute import autoimpute

    donor = pd.DataFrame({"x": np.arange(40.0), "y": 10 + np.arange(40.0) * 2})
    receiver = pd.DataFrame({"x": [50.0, 70.0]})
    result = autoimpute(
        donor,
        receiver,
        ["x"],
        ["y"],
        models=[OLS],
        train_size=1,
        k_folds=2,
        preprocessing={"x": "normalize", "y": "normalize"},
    )
    replay = result.fitted_models["best_method"].predict(receiver, quantiles=[0.5])[0.5]
    pd.testing.assert_frame_equal(replay, result.imputations["best_method"])
    np.testing.assert_allclose(replay.y, [110, 150])


def test_constant_categorical_get_imputations_has_exact_probabilities():
    from microimpute.comparisons import get_imputations

    donor = pd.DataFrame({"x": np.arange(20.0), "label": ["a"] * 20})
    receiver = pd.DataFrame({"x": [21.0, 22.0], "label": ["a", "b"]})
    predictions = get_imputations([OLS], donor, receiver, ["x"], ["label"], [0.5])
    assert predictions["OLS"]["probabilities"]["label"]["probabilities"].shape == (2, 1)
    result = compare_metrics(receiver[["label"]], predictions, ["label"])
    expected = sklearn_log_loss(["a", "b"], [[1.0, 0.0], [1.0, 0.0]], labels=["a", "b"])
    assert result.loc[result["Imputed Variable"] == "label", "Loss"].iloc[
        0
    ] == pytest.approx(expected)


def test_returned_model_preserves_probabilities_and_partial_target_transforms():
    from microimpute import autoimpute

    donor = pd.DataFrame(
        {
            "x": np.arange(40.0),
            "y": 10 + np.arange(40.0) * 2,
            "label": np.tile(["a", "b"], 20),
        }
    )
    receiver = pd.DataFrame({"x": [50.0, 70.0]})
    result = autoimpute(
        donor,
        receiver,
        ["x"],
        ["y", "label"],
        models=[OLS],
        train_size=1,
        k_folds=2,
        preprocessing={"x": "normalize", "y": "normalize"},
        imputation_quantiles=[0.5],
    )
    replay = result.fitted_models["best_method"].predict(
        receiver, quantiles=[0.5], return_probs=True
    )
    pd.testing.assert_frame_equal(replay[0.5], result.imputations["best_method"][0.5])
    np.testing.assert_allclose(
        replay["probabilities"]["label"]["probabilities"],
        result.imputations["best_method"]["probabilities"]["label"]["probabilities"],
    )


def test_qrf_comparison_quantiles_equal_independent_single_target_forecasts():
    from microimpute.comparisons import get_imputations
    from microimpute.models import QRF

    rng = np.random.default_rng(31)
    donor = pd.DataFrame({"x": rng.normal(size=160), "first": rng.normal(size=160)})
    donor["second"] = donor["first"] * 4 + rng.normal(size=160)
    receiver = pd.DataFrame({"x": np.linspace(-1, 1, 5)})
    quantiles = [0.1, 0.9]
    together = get_imputations(
        [QRF], donor, receiver, ["x"], ["first", "second"], quantiles
    )["QRF"]
    separate = get_imputations([QRF], donor, receiver, ["x"], ["second"], quantiles)[
        "QRF"
    ]
    for q in quantiles:
        np.testing.assert_allclose(together[q]["second"], separate[q]["second"])


def test_preprocessing_replay_ignores_receiver_target_placeholders():
    from microimpute import autoimpute

    donor = pd.DataFrame({"x": np.linspace(0, 2, 40)})
    donor["y"] = np.exp(1 + donor.x)
    receiver = pd.DataFrame({"x": [1.5], "y": [0.0]})
    result = autoimpute(
        donor,
        receiver,
        ["x"],
        ["y"],
        models=[OLS],
        train_size=1,
        k_folds=2,
        preprocessing={"y": "log"},
    )
    replay = result.fitted_models["best_method"].predict(receiver, quantiles=[0.5])[0.5]
    assert replay.y.iloc[0] == pytest.approx(np.exp(2.5))
    assert receiver.y.iloc[0] == 0
