"""Public regressions for the integration review of distributional APIs."""

import pickle

import numpy as np
import pandas as pd
import pytest

from microimpute.comparisons.autoimpute import _generate_imputations_for_all_models
from microimpute.comparisons.autoimpute_helpers import fit_and_predict_model
from microimpute.evaluations.predictor_analysis import (
    leave_one_out_analysis,
    progressive_predictor_inclusion,
)
from microimpute.models import OLS, QRF, Matching, QuantReg
from microimpute.models.zero_inflated import ZeroInflatedImputer


@pytest.fixture
def numeric_data():
    x = np.arange(40, dtype=float)
    return pd.DataFrame({"x": x, "y": x + np.sin(x), "z": x + np.cos(x)})


@pytest.mark.parametrize("seed", [2**32 - 1, np.int64(2**32 - 1)])
@pytest.mark.parametrize("sequential", [True, False])
def test_qrf_boundary_seed_fits_multiple_targets(numeric_data, seed, sequential):
    fitted = QRF(seed=seed, sequential=sequential).fit(
        numeric_data, ["x"], ["y", "z"], n_estimators=5
    )
    draws = fitted.predict(numeric_data[["x"]])
    assert draws.shape == (40, 2)
    assert np.isfinite(draws.to_numpy()).all()


def test_zero_inflated_boundary_component_seed(numeric_data):
    data = numeric_data.assign(y=np.where(numeric_data.x % 2, numeric_data.y, 0))
    fitted = ZeroInflatedImputer(base_imputer_class=QRF, seed=2**32 - 1).fit(
        data, ["x"], ["y"], n_estimators=5
    )
    predictions = fitted.predict(data[["x"]], quantiles=[0.2, 0.8])
    assert (predictions[0.2].y <= predictions[0.8].y).all()


@pytest.mark.parametrize("model_class", [OLS, QRF])
@pytest.mark.parametrize(
    "value, declaration",
    [("same", None), (True, None), (7, "categorical"), (True, "numeric")],
)
def test_constant_probability_public_and_helper(
    numeric_data, model_class, value, declaration
):
    data = numeric_data.assign(y=value)
    types = {"y": declaration} if declaration else None
    fitted, predictions = fit_and_predict_model(
        model_class,
        data,
        data[["x"]],
        ["x"],
        ["y"],
        None,
        0.5,
        target_types=types,
    )
    direct = fitted.predict(data[["x"]], quantiles=[0.5], return_probs=True)
    for result in (predictions, direct):
        if declaration == "numeric":
            assert "y" not in result.get("probabilities", {})
        else:
            info = result["probabilities"]["y"]
            np.testing.assert_array_equal(info["probabilities"], np.ones((40, 1)))
            np.testing.assert_array_equal(info["classes"], [value])


def test_zero_inflated_refit_replaces_type_state(numeric_data):
    imputer = ZeroInflatedImputer(base_imputer_class=OLS)
    imputer.fit(numeric_data, ["x"], ["y"])
    categorical = numeric_data.assign(y=np.where(numeric_data.x % 2, "a", "b"))
    fitted = imputer.fit(categorical, ["x"], ["y"], target_types={"y": "categorical"})
    assert set(fitted.predict(categorical[["x"]]).y) <= {"a", "b"}
    imputer.fit(numeric_data.assign(y=True), ["x"], ["y"])
    fitted = imputer.fit(numeric_data, ["x"], ["y"])
    assert fitted.predict(numeric_data[["x"]]).y.nunique() > 1


@pytest.mark.parametrize("model_class", [QRF, OLS])
def test_legacy_fitted_pickle_missing_new_state(numeric_data, model_class):
    model = model_class().fit(numeric_data, ["x"], ["y"])
    if model_class is QRF:
        del model.sequential
        for inner in model.models.values():
            del inner._weighted_leaves
            del inner._rng
    else:
        del model.rng
    restored = pickle.loads(pickle.dumps(model))
    draws = restored.predict(
        numeric_data[["x"]],
        **({"random_quantile_sample": True} if model_class is OLS else {}),
    )
    assert np.isfinite(draws.to_numpy()).all()
    again = restored.predict(
        numeric_data[["x"]],
        **({"random_quantile_sample": True} if model_class is OLS else {}),
    )
    assert not np.array_equal(draws.to_numpy(), again.to_numpy())


def test_quantreg_numeric_declaration_honored_before_guard(numeric_data):
    data = numeric_data.assign(y=numeric_data.x % 2 == 0)
    _, predictions = fit_and_predict_model(
        QuantReg,
        data,
        data[["x"]],
        ["x"],
        ["y"],
        None,
        0.5,
        target_types={"y": "numeric"},
    )
    assert np.isfinite(predictions[0.5].y).all()
    with pytest.raises(ValueError, match="categorical"):
        fit_and_predict_model(QuantReg, data, data[["x"]], ["x"], ["y"], None, 0.5)


@pytest.mark.parametrize(
    "analysis", [leave_one_out_analysis, progressive_predictor_inclusion]
)
def test_matching_distributional_predictor_analysis_rejected(numeric_data, analysis):
    with pytest.raises(NotImplementedError, match="Matching.*distribution"):
        analysis(numeric_data, ["x", "z"], ["y"], model_class=Matching)


def test_impute_all_includes_matching_draw(numeric_data, monkeypatch):
    import sys
    from types import SimpleNamespace

    def donor_draw(receiver, donor, matching_variables, z_variables, **kwargs):
        result = receiver.copy()
        for variable in z_variables:
            result[variable] = donor[variable].iloc[0]
        return result, result

    monkeypatch.setitem(
        sys.modules,
        "microimpute.utils.statmatch_hotdeck",
        SimpleNamespace(nnd_hotdeck_using_rpy2=donor_draw),
    )
    imputations, fitted = _generate_imputations_for_all_models(
        [Matching],
        "OLS",
        numeric_data,
        numeric_data[["x"]],
        ["x"],
        ["y"],
        None,
        0.5,
        0.8,
        None,
        "WARNING",
    )
    assert "Matching" in imputations and "Matching" in fitted
    np.testing.assert_array_equal(imputations["Matching"].y, np.zeros(40))


@pytest.mark.parametrize("model_class", [OLS, QRF])
def test_autoimpute_constant_probabilities_initial_and_replay(
    numeric_data, model_class
):
    from microimpute import autoimpute
    from microimpute.comparisons.metrics import compare_metrics

    data = numeric_data.assign(y="same")
    receiver = data[["x"]].iloc[:5]
    result = autoimpute(
        data,
        receiver,
        ["x"],
        ["y"],
        models=[model_class],
        k_folds=2,
        imputation_quantiles=[0.5],
        preprocessing={"x": "normalize"},
    )
    replay = result.fitted_models["best_method"].predict(
        receiver, quantiles=[0.5], return_probs=True
    )
    for predictions in (result.imputations["best_method"], replay):
        scores = compare_metrics(data.iloc[:5], {"model": predictions}, ["y"])
        assert scores.Loss.iloc[0] == pytest.approx(0)


@pytest.mark.parametrize("seed", [None, 0, np.int64(7)])
def test_qrf_valid_seed_controls(numeric_data, seed):
    fitted = QRF(seed=seed).fit(numeric_data, ["x"], ["y", "z"], n_estimators=3)
    assert np.isfinite(fitted.predict(numeric_data[["x"]]).to_numpy()).all()


@pytest.mark.parametrize("seed", [-1, 2**32, 1.5, True])
def test_invalid_seeds_are_not_wrapped(seed):
    with pytest.raises(ValueError, match="seed"):
        QRF(seed=seed)


@pytest.mark.parametrize("model_class", [OLS, QRF])
def test_constant_probabilities_without_explicit_quantiles(numeric_data, model_class):
    fitted = model_class().fit(numeric_data.assign(y="same"), ["x"], ["y"])
    predictions = fitted.predict(numeric_data[["x"]], return_probs=True)
    assert isinstance(predictions, dict)
    np.testing.assert_array_equal(
        predictions["probabilities"]["y"]["probabilities"], np.ones((40, 1))
    )
