"""Compatibility checks for public imports used by published examples."""

import ast
import importlib
import json
from pathlib import Path

import pytest


def test_legacy_scf_years_remain_importable() -> None:
    """Existing SCF examples retain their historical supported-year list."""
    from microimpute.config import VALID_YEARS

    assert VALID_YEARS == list(range(1989, 2023, 3))


def test_legacy_model_parameters_remain_importable() -> None:
    """Preserve the public compatibility mapping, not learner defaults."""
    from microimpute.config import DEFAULT_MODEL_PARAMS

    assert DEFAULT_MODEL_PARAMS == {
        "qrf": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": 1.0,
        },
        "quantreg": {},
        "ols": {"l1_ratio": 0, "C": 1.0, "max_iter": 1000},
        "matching": {},
        "mdn": {
            "layers": "128-64-32",
            "activation": "ReLU",
            "dropout": 0.0,
            "use_batch_norm": False,
            "num_gaussian": 5,
            "softmax_temperature": 1.0,
            "n_samples": 100,
            "learning_rate": 1e-3,
            "max_epochs": 100,
            "early_stopping_patience": 10,
            "batch_size": 256,
        },
    }


@pytest.mark.parametrize(
    "notebook_path",
    [
        "docs/imputation-benchmarking/benchmarking-methods.ipynb",
        "paper/imputing-from-scf-to-cps.ipynb",
    ],
)
def test_published_notebook_config_imports(notebook_path: str) -> None:
    """Execute the examples' config imports without downloading their data."""
    repository = Path(__file__).resolve().parents[1]
    notebook = json.loads((repository / notebook_path).read_text())
    imports = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "from microimpute.config import" not in source:
            continue
        imports.extend(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom) and node.module == "microimpute.config"
        )

    assert imports, f"No config imports found in {notebook_path}"
    namespace = {}
    for node in imports:
        statement = ast.Module(body=[node], type_ignores=[])
        exec(compile(statement, notebook_path, "exec"), namespace)
    assert namespace["VALID_YEARS"] == list(range(1989, 2023, 3))


@pytest.mark.parametrize("module_name", ["microimpute", "microimpute.models"])
def test_zero_inflated_public_alias(module_name: str) -> None:
    """Both public exports expose the same usable wrapper class."""
    from microimpute.models import QRF
    from microimpute.models.zero_inflated import ZeroInflatedImputer

    exported = getattr(importlib.import_module(module_name), "ZeroInflatedImputer")
    assert exported is ZeroInflatedImputer
    assert isinstance(exported(base_imputer_class=QRF), ZeroInflatedImputer)
