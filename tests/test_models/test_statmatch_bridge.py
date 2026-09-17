"""Validate Python/R argument contract against the documented StatMatch API.

These unit tests use a fake R package; real integration tests remain optional.
"""

import contextlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def bridge(monkeypatch):
    calls = []

    class Converter:
        def __add__(self, other):
            return self

    def run(name, **kwargs):
        calls.append((name, kwargs))
        return SimpleNamespace(rx2=lambda key: SimpleNamespace(ncol=2))

    def fused(**kwargs):
        out = kwargs["data_rec"].copy()
        out["y"] = kwargs["data_don"].y.iloc[0]
        return out

    package = SimpleNamespace(
        NND_hotdeck=lambda **kwargs: run("NND", **kwargs),
        RANDwNND_hotdeck=lambda **kwargs: run("RAND", **kwargs),
        create_fused=fused,
    )
    robjects = ModuleType("rpy2.robjects")
    robjects.StrVector = list
    robjects.FloatVector = list
    robjects.IntVector = list
    robjects.default_converter = Converter()
    robjects.numpy2ri = SimpleNamespace(converter=Converter())
    robjects.pandas2ri = SimpleNamespace(converter=Converter())
    conversion = ModuleType("rpy2.robjects.conversion")
    conversion.py2rpy = lambda x: x
    conversion.rpy2py = lambda x: x
    conversion.localconverter = lambda _: contextlib.nullcontext()
    robjects.conversion = conversion
    packages = ModuleType("rpy2.robjects.packages")
    packages.importr = lambda name: package
    for name, module in {
        "rpy2": ModuleType("rpy2"),
        "rpy2.robjects": robjects,
        "rpy2.robjects.conversion": conversion,
        "rpy2.robjects.packages": packages,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    module_path = Path(__file__).parents[2] / "microimpute/utils/statmatch_hotdeck.py"
    spec = importlib.util.spec_from_file_location(
        "statmatch_contract_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, calls


def test_weighted_matching_uses_rand_and_a_donor_column_name(bridge):
    # CRAN RANDwNND.hotdeck documents weight.don as a donor column NAME,
    # and cut.don="min" retains all donors tied at the nearest distance.
    module, calls = bridge
    donor = pd.DataFrame({"x": [1.0, 1.0], "y": [10.0, 20.0]})
    original = donor.copy()
    receiver = pd.DataFrame({"x": [1.0]})
    module.nnd_hotdeck_using_rpy2(
        receiver, donor, ["x"], ["y"], donor_sample_weight=np.array([1.0, 100.0])
    )
    name, kwargs = calls[0]
    assert name == "RAND"
    assert kwargs["cut_don"] == "min"
    weight_name = kwargs["weight_don"]
    assert isinstance(weight_name, str)
    np.testing.assert_array_equal(kwargs["data_don"][weight_name], [1.0, 100.0])
    pd.testing.assert_frame_equal(donor, original)


def test_unweighted_matching_keeps_nnd(bridge):
    module, calls = bridge
    donor = pd.DataFrame({"x": [1.0, 2.0], "y": [10.0, 20.0]})
    module.nnd_hotdeck_using_rpy2(donor[["x"]], donor, ["x"], ["y"])
    assert calls[0][0] == "NND"
    assert "weight_don" not in calls[0][1]


@pytest.mark.parametrize(
    "weights", [[1.0], [0.0, 0.0], [-1.0, 2.0], [1.0, np.nan], [np.inf, 1.0]]
)
def test_bridge_rejects_invalid_donor_weights(bridge, weights):
    module, calls = bridge
    donor = pd.DataFrame({"x": [1.0, 2.0], "y": [10.0, 20.0]})
    with pytest.raises(ValueError, match="weights"):
        module.nnd_hotdeck_using_rpy2(
            donor[["x"]], donor, ["x"], ["y"], donor_sample_weight=np.array(weights)
        )
    assert not calls


def test_bridge_rejects_weighted_constrained_matching(bridge):
    module, calls = bridge
    donor = pd.DataFrame({"x": [1.0, 2.0], "y": [10.0, 20.0]})
    with pytest.raises(ValueError, match="constrained"):
        module.nnd_hotdeck_using_rpy2(
            donor[["x"]],
            donor,
            ["x"],
            ["y"],
            donor_sample_weight=np.ones(2),
            constrained=True,
        )
    assert not calls


def test_fallback_matching_pairs_use_r_column_major_order(bridge):
    module, calls = bridge
    donor = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
    module._get_statmatch().NND_hotdeck = lambda **kwargs: SimpleNamespace(
        rx2=lambda key: np.array([3, 1, 2])
    )
    matrices = []

    def matrix(values, nrow, ncol):
        result = np.asarray(values).reshape(nrow, ncol, order="F")
        matrices.append(result)
        return result

    module.ro.IntVector = list
    module.ro.r = SimpleNamespace(matrix=matrix)
    module.nnd_hotdeck_using_rpy2(donor[["x"]], donor, ["x"], ["y"])
    np.testing.assert_array_equal(matrices[0], [[1, 3], [2, 1], [3, 2]])


def test_r_rng_state_is_restored_after_seeded_call(bridge):
    module, calls = bridge
    donor = pd.DataFrame({"x": [1.0, 2.0], "y": [10.0, 20.0]})
    original_state = np.array([1, 2, 3])
    module.ro.globalenv = {".Random.seed": original_state}
    observed_seeds = []

    def set_seed(seed):
        observed_seeds.append(seed)
        module.ro.globalenv[".Random.seed"] = np.array([seed])

    module.ro.r = {"set.seed": set_seed}
    module.nnd_hotdeck_using_rpy2(
        donor[["x"]],
        donor,
        ["x"],
        ["y"],
        donor_sample_weight=np.ones(2),
        random_state=73,
    )
    assert observed_seeds == [73]
    np.testing.assert_array_equal(module.ro.globalenv[".Random.seed"], original_state)
    assert "random_state" not in calls[0][1]


def test_seeded_r_call_restores_absent_rng_state_even_on_error(bridge):
    module, calls = bridge
    donor = pd.DataFrame({"x": [1.0, 2.0], "y": [10.0, 20.0]})
    module.ro.globalenv = {}
    module.ro.r = {
        "set.seed": lambda seed: module.ro.globalenv.update({".Random.seed": [seed]})
    }

    def fail(**kwargs):
        raise RuntimeError("injected R failure")

    module._get_statmatch().NND_hotdeck = fail
    with pytest.raises(RuntimeError, match="injected R failure"):
        module.nnd_hotdeck_using_rpy2(
            donor[["x"]], donor, ["x"], ["y"], random_state=73
        )
    assert ".Random.seed" not in module.ro.globalenv
