"""The reported version must match the installed distribution.

`__init__.py` used to hard-code `__version__` as 1.1.2, which fell behind the
package's own version (3.1.1 by September 2026). These tests catch that drift.
"""

import subprocess
import sys
import textwrap
from importlib.metadata import version
from pathlib import Path

from packaging.version import Version

import microimpute

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_version_matches_installed_distribution():
    assert microimpute.__version__ == version("microimpute")


def test_version_is_pep440_parseable():
    """The fallback must parse too, or consumers comparing versions raise."""
    Version(microimpute.__version__)
    Version("0.0.0+unknown")


def test_fallback_used_when_distribution_is_absent():
    """Without a distribution to read, the package reports 0.0.0+unknown."""
    # A fresh interpreter makes the metadata lookup for microimpute fail and
    # then imports the package from this source tree, so the import runs the
    # fallback branch in microimpute/__init__.py itself.
    code = textwrap.dedent(
        """
        import importlib.metadata as metadata
        import sys

        real_version = metadata.version


        def missing(name):
            if name == "microimpute":
                raise metadata.PackageNotFoundError(name)
            return real_version(name)


        metadata.version = missing
        sys.path.insert(0, sys.argv[1])
        import microimpute

        print(microimpute.__version__)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(REPO_ROOT)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "0.0.0+unknown"
