"""Tests for the MDN dataset-hashing helpers used to key the model cache.

These regression tests cover the cache-key collision bug (#5). The tests
are gated on the full MDN import stack being available because mdn.py's
top-level ``import torch`` is not optional.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("pytorch_tabular")

from microimpute.models.mdn import _generate_cache_key, _generate_data_hash


def test_generate_data_hash_is_stable() -> None:
    """The same data must always hash to the same digest."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    y = pd.Series([7, 8, 9], name="target")
    assert _generate_data_hash(X, y) == _generate_data_hash(X.copy(), y.copy())


def test_generate_data_hash_detects_value_change() -> None:
    """A single-value change must produce a different digest."""
    X1 = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    X2 = pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [4.0, 5.0, 6.0]})
    y = pd.Series([7, 8, 9], name="target")
    assert _generate_data_hash(X1, y) != _generate_data_hash(X2, y)


def test_generate_data_hash_is_order_sensitive() -> None:
    """Regression test for the sum-of-hashes collision bug (#5).

    Permuting rows must produce a different cache key. Previously the
    key was the SUM of per-row uint64 hashes, so any row permutation
    hashed identically and cache lookups could load a stale model
    trained on differently ordered data.
    """
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10, 20, 30, 40]})
    y = pd.Series([100, 200, 300, 400], name="target")

    X_perm = X.iloc[[3, 1, 0, 2]].reset_index(drop=True)
    y_perm = y.iloc[[3, 1, 0, 2]].reset_index(drop=True)

    assert _generate_data_hash(X, y) != _generate_data_hash(
        X_perm, y_perm
    ), "Permuted rows must produce a different cache key"


def test_generate_data_hash_avoids_sum_collision() -> None:
    """Two datasets of matching shape/columns whose per-row hashes sum to
    the same value must still produce different cache keys.

    Previously the cache key was effectively
    ``hash(sum(per_row_hashes))``, making collisions trivial — a stale
    cached MDN could be loaded for a new, different dataset (silent
    correctness bug).
    """
    X1 = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    y1 = pd.Series([7.0, 8.0, 9.0], name="target")

    X2 = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.1]})
    y2 = pd.Series([7.0, 8.0, 9.0], name="target")

    assert _generate_data_hash(X1, y1) != _generate_data_hash(X2, y2)


def test_generate_data_hash_differs_across_random_datasets() -> None:
    """Property-style check: 50 random datasets produce 50 distinct hashes."""
    rng = np.random.default_rng(0)
    hashes = set()
    for _ in range(50):
        X = pd.DataFrame(rng.normal(size=(20, 3)), columns=["a", "b", "c"])
        y = pd.Series(rng.normal(size=20), name="target")
        hashes.add(_generate_data_hash(X, y))
    assert (
        len(hashes) == 50
    ), "50 random datasets should produce 50 distinct cache keys"


def test_generate_cache_key_integrates_data_hash() -> None:
    """_generate_cache_key must change when _generate_data_hash changes."""
    k1 = _generate_cache_key(["a", "b"], "target", "data_hash_1")
    k2 = _generate_cache_key(["a", "b"], "target", "data_hash_2")
    assert k1 != k2
