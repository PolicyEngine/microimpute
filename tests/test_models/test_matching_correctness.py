"""Matching API regression tests using an injected matcher, without optional R."""

import numpy as np
import pandas as pd
import pytest

from microimpute.models.matching import Matching


@pytest.fixture
def donor():
    return pd.DataFrame(
        {"x": np.arange(12.0), "y": np.arange(12.0) + 0.5, "w": np.arange(12.0) + 1}
    )


def first_donor(receiver, donor, matching_variables, z_variables, **kwargs):
    result = receiver.copy()
    for var in z_variables:
        result[var] = donor[var].iloc[0]
    return result, result


@pytest.mark.parametrize(
    "options", [{"quantiles": [0.1, 0.5, 0.9]}, {"return_probs": True}]
)
def test_matching_rejects_unestimated_distributions(donor, options):
    fitted = Matching(matching_hotdeck=first_donor).fit(donor, ["x"], ["y"])
    with pytest.raises(
        NotImplementedError, match="conditional quantiles|probabilities"
    ):
        fitted.predict(donor[["x"]], **options)


def test_tuned_matching_preserves_donor_weights_and_fixed_options(donor, monkeypatch):
    received = []

    def match(**kwargs):
        received.append(kwargs)
        return first_donor(**kwargs)

    def tune(self, **kwargs):
        assert np.array_equal(kwargs["matching_kwargs"]["donor_sample_weight"], donor.w)
        return {"dist_fun": "Euclidean"}

    monkeypatch.setattr(Matching, "_tune_hyperparameters", tune)
    fitted, params = Matching(matching_hotdeck=match).fit(
        donor, ["x"], ["y"], weight_col="w", tune_hyperparameters=True, keep_t=True
    )
    actual = fitted.predict(donor[["x"]])
    assert len(actual) == len(donor)
    assert params == {"dist_fun": "Euclidean"}
    assert received[0]["keep_t"] is True
    np.testing.assert_array_equal(received[0]["donor_sample_weight"], donor.w)


def test_matching_cv_slices_weights_and_never_scores_failed_trials(donor, monkeypatch):
    import optuna

    trials = []
    calls = []
    real_create_study = optuna.create_study

    def study_factory(*args, **kwargs):
        study = real_create_study(*args, **kwargs)
        trials.append(study)
        return study

    def match(**kwargs):
        calls.append(kwargs)
        np.testing.assert_array_equal(kwargs["donor_sample_weight"], kwargs["donor"].w)
        raise RuntimeError("injected matching failure")

    monkeypatch.setattr(optuna, "create_study", study_factory)
    model = Matching(matching_hotdeck=match)
    with pytest.raises(ValueError, match="No matching hyperparameter trial succeeded"):
        model.fit(donor, ["x"], ["y"], weight_col="w", tune_hyperparameters=True)
    assert calls
    assert all(t.state == optuna.trial.TrialState.PRUNED for t in trials[0].trials)


def test_chunk_failures_report_counts_and_reset_on_success(donor):
    def match(**kwargs):
        if kwargs["receiver"].x.iloc[0] == 2:
            raise RuntimeError("injected second-chunk failure")
        return first_donor(**kwargs)

    fitted = Matching(matching_hotdeck=match).fit(donor, ["x"], ["y"])
    receiver = donor[["x"]].iloc[:5]
    output = fitted._predict_chunked(receiver, quantiles=None, chunk_size=2)
    assert fitted.n_failed_records == 2
    assert output.attrs["n_failed_records"] == 2
    assert output.y.isna().sum() == 2
    clean = fitted.predict(receiver.iloc[:1])
    assert fitted.n_failed_records == 0
    assert clean.attrs["n_failed_records"] == 0


def test_r_matching_seed_stream_reproduces_and_advances(donor, monkeypatch):
    """Default bridge gets fresh child seeds while equal model seeds reproduce."""
    import sys
    from types import SimpleNamespace

    seeds = []

    def fake_bridge(**kwargs):
        seeds.append(kwargs.pop("random_state"))
        return first_donor(**kwargs)

    monkeypatch.setitem(
        sys.modules,
        "microimpute.utils.statmatch_hotdeck",
        SimpleNamespace(nnd_hotdeck_using_rpy2=fake_bridge),
    )
    receiver = donor[["x"]].iloc[:5]
    first = Matching(seed=17).fit(donor, ["x"], ["y"], weight_col="w")
    first.predict(receiver)
    first._predict_chunked(receiver, quantiles=None, chunk_size=2)
    initial = seeds.copy()
    assert len(initial) == 4
    assert len(set(initial)) == 4
    seeds.clear()
    second = Matching(seed=17).fit(donor, ["x"], ["y"], weight_col="w")
    second.predict(receiver)
    second._predict_chunked(receiver, quantiles=None, chunk_size=2)
    assert seeds == initial
    third = Matching(seed=18).fit(donor, ["x"], ["y"], weight_col="w")
    third.predict(receiver)
    assert seeds[-1] != initial[0]
