"""Saved models preserve progressive streams and migrate missing RNG state."""

import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microimpute.models import OLS, QRF, Matching


@pytest.mark.parametrize("model_class", [OLS, QRF])
def test_current_roundtrip_preserves_next_draw(model_class):
    x = np.arange(50.0)
    data = pd.DataFrame({"x": x, "y": x + np.sin(x)})
    model = model_class(seed=73).fit(data, ["x"], ["y"])
    kwargs = {"random_quantile_sample": True} if model_class is OLS else {}
    model.predict(data[["x"]], **kwargs)
    restored = pickle.loads(pickle.dumps(model))
    pd.testing.assert_frame_equal(
        model.predict(data[["x"]], **kwargs), restored.predict(data[["x"]], **kwargs)
    )


def test_legacy_matching_roundtrip_recreates_native_draw_stream(monkeypatch):
    def seeded_draw(receiver, donor, matching_variables, z_variables, random_state):
        result = receiver.copy()
        result["y"] = np.random.default_rng(random_state).choice(donor.y, len(receiver))
        return result, result

    monkeypatch.setitem(
        sys.modules,
        "microimpute.utils.statmatch_hotdeck",
        SimpleNamespace(nnd_hotdeck_using_rpy2=seeded_draw),
    )
    data = pd.DataFrame({"x": np.arange(30.0), "y": np.arange(30.0)})
    model = Matching(seed=19).fit(data, ["x"], ["y"])
    del model._rng
    first = pickle.loads(pickle.dumps(model))
    second = pickle.loads(pickle.dumps(model))
    draw = first.predict(data[["x"]])
    pd.testing.assert_frame_equal(draw, second.predict(data[["x"]]))
    assert not draw.equals(first.predict(data[["x"]]))


def test_custom_matching_adapter_keeps_unseeded_contract():
    def custom_draw(receiver, donor, matching_variables, z_variables):
        result = receiver.copy()
        result["y"] = donor.y.iloc[0]
        return result, result

    data = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    model = Matching(matching_hotdeck=custom_draw, seed=19).fit(data, ["x"], ["y"])
    np.testing.assert_array_equal(model.predict(data[["x"]]).y, [3.0, 3.0])


def test_legacy_ols_constant_predictor_preserves_saved_design():
    import statsmodels.api as sm

    data = pd.DataFrame({"x": np.ones(40), "y": np.arange(40.0)})
    fitted = OLS().fit(data, ["x"], ["y"], not_numeric_categorical=["x"])
    # Historical add_constant skipped the intercept when x was constant.
    historical = sm.OLS(data.y, sm.add_constant(data[["x"]])).fit()
    fitted.models["y"].model = historical
    fitted.models["y"].scale = historical.scale
    del fitted.rng
    restored = pickle.loads(pickle.dumps(fitted))
    predictions = restored.predict(data[["x"]], quantiles=[0.5])[0.5]
    np.testing.assert_allclose(predictions.y, historical.predict(data[["x"]]))
    np.testing.assert_allclose(
        restored.models["y"].predict(data[["x"]]), historical.predict(data[["x"]])
    )
