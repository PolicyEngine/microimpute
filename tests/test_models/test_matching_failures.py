"""Failure handling with a deterministic custom backend and real Optuna studies."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest


@pytest.fixture
def matching_class(monkeypatch):
    """Load Matching without installing or globally replacing its optional R adapter."""
    try:
        from microimpute.models.matching import Matching
    except ModuleNotFoundError as error:
        if not error.name.startswith("rpy2"):
            raise
        import microimpute.models

        adapter = types.ModuleType("microimpute.utils.statmatch_hotdeck")

        def unavailable_adapter(*args, **kwargs):
            raise AssertionError("These tests must use their custom matching backend")

        adapter.nnd_hotdeck_using_rpy2 = unavailable_adapter
        path = Path(microimpute.models.__file__).with_name("matching.py")
        spec = importlib.util.spec_from_file_location("_matching_failure_tests", path)
        module = importlib.util.module_from_spec(spec)
        with monkeypatch.context() as scoped:
            scoped.setitem(sys.modules, adapter.__name__, adapter)
            scoped.setitem(sys.modules, spec.name, module)
            spec.loader.exec_module(module)
        return module.Matching
    return Matching


def donor_data(size=30):
    x = np.arange(size, dtype=float)
    return pd.DataFrame({"x": x, "y": 2 * x + 1})


def exact_backend(receiver, donor, matching_variables, z_variables, **kwargs):
    """Predict the known linear relation, making the successful-trial score exact."""
    result = receiver.copy()
    for variable in z_variables:
        result[variable] = 2 * receiver["x"].to_numpy() + 1
    return result, result.copy()


@pytest.fixture
def studies(monkeypatch):
    """Queue one failing candidate followed by a successful candidate deterministically."""
    created = []
    create_study = optuna.create_study

    def tracked_study(**kwargs):
        study = create_study(**kwargs)
        study.enqueue_trial({"dist_fun": "Manhattan", "k": 1})
        study.enqueue_trial({"dist_fun": "Euclidean", "k": 1})
        created.append(study)
        return study

    monkeypatch.setattr(optuna, "create_study", tracked_study)
    return created


@pytest.mark.parametrize("size", [30, 3003])
def test_failed_tuning_candidate_cannot_win(matching_class, studies, size):
    """Prune failures, including a second failed chunk after the first succeeds."""
    successful_manhattan_chunks = []

    def backend(**kwargs):
        if kwargs["dist_fun"] == "Manhattan":
            if size == 30 or len(kwargs["receiver"]) == 1:
                raise RuntimeError("candidate cannot match")
            successful_manhattan_chunks.append(len(kwargs["receiver"]))
        return exact_backend(**kwargs)

    data = donor_data(size)
    fitted, params = matching_class(backend).fit(
        data, ["x"], ["y"], tune_hyperparameters=True
    )
    trials = studies[0].trials
    assert trials[0].state == optuna.trial.TrialState.PRUNED
    assert trials[1].state == optuna.trial.TrialState.COMPLETE
    assert trials[1].value == pytest.approx(0.0)
    assert params["dist_fun"] != "Manhattan"
    if size > 30:
        assert successful_manhattan_chunks and set(successful_manhattan_chunks) == {
            1000
        }
    prediction = fitted.predict(data[["x"]].iloc[:3], quantiles=[0.5])[0.5]
    np.testing.assert_array_equal(prediction.y, data.y.iloc[:3])


@pytest.mark.parametrize("size", [30, 3003])
def test_all_failed_trials_raise_without_a_model(matching_class, studies, size):
    """An all-pruned study must never fall back to a fitted mean predictor."""

    def backend(**kwargs):
        raise RuntimeError("no valid donor match")

    with pytest.raises(ValueError, match="No matching hyperparameter trial succeeded"):
        matching_class(backend).fit(
            donor_data(size), ["x"], ["y"], tune_hyperparameters=True
        )
    assert studies[0].trials
    assert all(
        trial.state == optuna.trial.TrialState.PRUNED for trial in studies[0].trials
    )


def test_failure_count_available_before_and_after_small_prediction(matching_class):
    """A new fitted model and its successful unchunked result report zero failures."""
    fitted = matching_class(exact_backend).fit(donor_data(), ["x"], ["y"])
    assert fitted.n_failed_records == 0
    prediction = fitted.predict(donor_data()[["x"]].iloc[:3], quantiles=[0.5])[0.5]
    assert fitted.n_failed_records == 0
    assert not prediction.isna().any().any()


@pytest.mark.parametrize("next_size", [3, 2001])
def test_partial_prediction_preserves_rows_and_resets_count(matching_class, next_size):
    """A later successful prediction replaces the previous failure count."""

    def backend(**kwargs):
        if len(kwargs["receiver"]) == 1:
            raise RuntimeError("last chunk cannot match")
        return exact_backend(**kwargs)

    fitted = matching_class(backend).fit(donor_data(), ["x"], ["y"])
    receiver = pd.DataFrame(
        {"x": np.arange(2001, dtype=float)}, index=np.arange(10000, 12001)
    )
    prediction = fitted.predict(receiver, quantiles=[0.5])[0.5]
    assert prediction.index.equals(receiver.index)
    np.testing.assert_array_equal(prediction.y.iloc[:-1], 2 * receiver.x.iloc[:-1] + 1)
    assert pd.isna(prediction.y.iloc[-1])
    assert fitted.n_failed_records == 1

    fitted.matching_hotdeck = exact_backend
    prediction = fitted.predict(receiver.iloc[:next_size], quantiles=[0.5])[0.5]
    assert not prediction.isna().any().any()
    assert fitted.n_failed_records == 0


def test_small_prediction_reports_missing_targets_only(matching_class, caplog):
    """Count a partially missing single-call result, not unrelated backend columns."""

    def backend(**kwargs):
        result, _ = exact_backend(**kwargs)
        result["unused"] = np.nan
        result.iloc[-1, result.columns.get_loc("y")] = np.nan
        return result, result.copy()

    fitted = matching_class(backend).fit(donor_data(), ["x"], ["y"])
    prediction = fitted.predict(donor_data()[["x"]].iloc[:3], quantiles=[0.5])[0.5]
    assert prediction.y.isna().sum() == 1
    assert fitted.n_failed_records == 1
    assert "1 of 3 records (33.3%) could not be matched" in caplog.text


@pytest.mark.parametrize("quantiles", [None, [0.25, 0.75]])
@pytest.mark.parametrize("failed_rows", [0, 1])
def test_prediction_frames_include_failure_metadata(
    matching_class, quantiles, failed_rows
):
    """Every returned frame exposes failures without changing values or row labels."""

    def backend(**kwargs):
        result, _ = exact_backend(**kwargs)
        if failed_rows:
            result.iloc[-1, result.columns.get_loc("y")] = np.nan
        return result, result.copy()

    fitted = matching_class(backend).fit(donor_data(), ["x"], ["y"])
    receiver = pd.DataFrame({"x": [1.0, 3.0, 5.0]}, index=[20, 10, 30])
    result = fitted.predict(receiver, quantiles=quantiles)
    frames = list(result.values()) if isinstance(result, dict) else [result]
    expected = 2 * receiver.x + 1
    if failed_rows:
        expected.iloc[-1] = np.nan
    for frame in frames:
        assert frame.attrs["n_failed_records"] == failed_rows
        assert frame.index.equals(receiver.index)
        np.testing.assert_array_equal(frame.y, expected)
    assert fitted.n_failed_records == failed_rows

    fitted.matching_hotdeck = exact_backend
    next_result = fitted.predict(receiver, quantiles=quantiles)
    next_frames = (
        list(next_result.values()) if isinstance(next_result, dict) else [next_result]
    )
    for frame in next_frames:
        assert frame.attrs["n_failed_records"] == 0
    assert fitted.n_failed_records == 0
    assert all(frame.attrs["n_failed_records"] == failed_rows for frame in frames)
