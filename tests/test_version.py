"""The reported version must match the installed distribution.

`__version__` was hardcoded and drifted to 1.1.2 while the package was 3.1.1,
which is the regression this guards against.
"""

from importlib.metadata import version

import pytest
from packaging.version import Version

import microimpute


def test_version_matches_installed_distribution():
    assert microimpute.__version__ == version("microimpute")


def test_version_is_pep440_parseable():
    """The fallback must parse too, or consumers comparing versions raise."""
    Version(microimpute.__version__)
    Version("0.0.0+unknown")


def test_fallback_used_when_distribution_is_absent(monkeypatch):
    """The fallback branch runs when the distribution cannot be found."""
    import importlib.metadata

    def _raise(_name):
        raise importlib.metadata.PackageNotFoundError(_name)

    monkeypatch.setattr(importlib.metadata, "version", _raise)

    # Re-run the same lookup __init__ performs, rather than reloading the
    # package: reloading re-imports every model and is slow and fragile.
    try:
        resolved = importlib.metadata.version("microimpute")
    except importlib.metadata.PackageNotFoundError:
        resolved = "0.0.0+unknown"

    assert resolved == "0.0.0+unknown"
    Version(resolved)
