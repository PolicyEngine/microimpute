"""Exercise shared preprocessing and scoring with real neural backends."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pytorch_tabular")

from microimpute.comparisons.autoimpute_helpers import (
    prepare_data_for_imputation,
    preprocessing_aware_model,
)
from microimpute.comparisons.metrics import compare_metrics
from microimpute.evaluations.cross_validation import cross_validate_model
from microimpute.models.mdn import MDN


@pytest.fixture
def small_mdn(tmp_path):
    class SmallMDN(MDN):
        def __init__(self, **kwargs):
            super().__init__(
                layers="8",
                max_epochs=2,
                batch_size=16,
                num_gaussian=2,
                model_dir=str(tmp_path / "models"),
                force_retrain=True,
                **kwargs,
            )

    return SmallMDN


@pytest.fixture
def mixed_data():
    rng = np.random.default_rng(19)
    x = rng.normal(size=48)
    return pd.DataFrame(
        {"x": x, "y": 5 + x + rng.normal(size=48), "label": np.where(x > 0, "a", "b")}
    )


def test_real_mdn_mixed_probabilities_and_strict_scoring(small_mdn, mixed_data):
    fitted = small_mdn().fit(mixed_data, ["x"], ["y", "label"])
    predictions = fitted.predict(
        mixed_data[["x"]], quantiles=[0.2, 0.8], return_probs=True
    )
    assert np.isfinite(predictions[0.2].y).all()
    info = predictions["probabilities"]["label"]
    np.testing.assert_allclose(info["probabilities"].sum(axis=1), 1, atol=1e-6)
    assert set(info["classes"]) == {"a", "b"}
    scores = compare_metrics(mixed_data, {"MDN": predictions}, ["y", "label"])
    assert np.isfinite(scores.Loss).all()


def test_real_mdn_cross_validation_uses_probabilities(small_mdn, mixed_data):
    results = cross_validate_model(
        small_mdn, mixed_data, ["x"], ["label"], n_splits=2, quantiles=[0.5]
    )
    assert np.isfinite(results["log_loss"]["mean_test"])
    assert results["log_loss"]["mean_test"] > 0


def test_real_mdn_preprocessing_replay_uses_donor_transform(small_mdn, mixed_data):
    receiver = pd.DataFrame({"x": [10.0, 20.0, 30.0]})
    train, transformed, params = prepare_data_for_imputation(
        mixed_data,
        receiver,
        ["x"],
        ["y"],
        None,
        1.0,
        0.0,
        preprocessing={"x": "normalize", "y": "normalize"},
    )
    fitted = small_mdn().fit(train, ["x"], ["y"])
    wrapped = preprocessing_aware_model(fitted, params)
    import torch

    torch.manual_seed(73)
    initial = fitted.predict(transformed, quantiles=[0.5])[0.5].y
    torch.manual_seed(73)
    replay = wrapped.predict(receiver, quantiles=[0.5])[0.5].y
    np.testing.assert_allclose(
        replay, initial * mixed_data.y.std() + mixed_data.y.mean(), rtol=1e-5
    )
    assert wrapped.transform_params["normalization"]["x"]["mean"] == pytest.approx(
        mixed_data.x.mean()
    )


def test_real_mdn_constant_categorical_probabilities(small_mdn, mixed_data):
    data = mixed_data.assign(label="same")
    fitted = small_mdn().fit(data, ["x"], ["label"])
    result = fitted.predict(data[["x"]], quantiles=[0.5], return_probs=True)
    np.testing.assert_array_equal(
        result["probabilities"]["label"]["probabilities"], np.ones((48, 1))
    )


@pytest.mark.parametrize("constant", [False, True])
def test_real_mdn_probabilities_without_explicit_quantiles(
    small_mdn, mixed_data, constant
):
    data = mixed_data.assign(label="same") if constant else mixed_data
    fitted = small_mdn().fit(data, ["x"], ["label"])
    result = fitted.predict(data[["x"]], return_probs=True)
    assert isinstance(result, dict)
    probabilities = result["probabilities"]["label"]["probabilities"]
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
