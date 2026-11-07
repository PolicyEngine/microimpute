"""Cross-fit MOOP imputations on the 2026 CPS ASEC and compare SPM poverty.

Each SPM unit's reported medical out-of-pocket spending (MOOP) is
replaced by an imputation from models fit on the other folds (folds
group whole households), so every unit gets an out-of-fold imputation.
SPM poverty is then recomputed for every person under each method.

Usage (from the repository root):

    python -m experiments.moop_spm.run --data-dir ~/data/cps_asec_2026

The first run downloads the Census public-use zip (139 MB) into
``--data-dir`` and checks its SHA-256. Per-fold imputations are cached in
``<output>/cache`` so an interrupted run resumes. The cache key covers the
settings, the experiment's source files, every source file of microimpute
and of Microcosm's ``fit``, ``frame`` and ``graph`` packages, the versions
of Python and the numerical packages, the commit Microcosm came from, and
the model inputs with their row keys, so a change to any of them refits
every fold.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import time
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

import numpy as np
import pandas as pd

from .cps_asec import (
    REFERENCE_YEAR,
    SURVEY_YEAR,
    ZIP_SHA256,
    ZIP_URL,
    allocation_flags,
    build_units,
    check_spm_identity,
    design_matrix,
    load_persons,
    sha256_of,
)
from .evaluate import evaluate
from .methods import (
    CELL_LEVELS,
    DETERMINISTIC,
    MIN_CELL_DONORS,
    PARTICIPATION_CUTOFF,
    PMM_NEIGHBORS,
    FoldData,
    cell_keys,
    logit_family,
    microcosm_qrf,
    pmm,
    qrf,
    zi_qrf,
)

HERE = Path(__file__).resolve().parent
SOURCES = ["cps_asec.py", "methods.py", "evaluate.py", "run.py"]
# Distributions whose versions can change fitted models or draws.
PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit_learn": "scikit-learn",
    "quantile_forest": "quantile-forest",
    "microimpute": "microimpute",
}
# Microcosm's distributions and the packages they install. They install
# from git, and their version numbers do not change from commit to
# commit, so we record the commit and hash the source.
MICROCOSM_PACKAGES = {
    "microcosm_fit": ("microcosm-fit", "microcosm.fit"),
    "microcosm_frame": ("microcosm-frame", "microcosm.frame"),
    "microcosm_graph": ("microcosm-graph", "microcosm.graph"),
}
PUBLISHED = json.loads((HERE / "published_2025.json").read_text())


def assign_folds(households: pd.Series, n_folds: int, seed: int) -> np.ndarray:
    """Assign whole households to folds at random, as evenly as possible."""
    unique = np.sort(households.unique())
    permuted = np.random.default_rng(seed).permutation(unique)
    fold_of = pd.Series(np.arange(len(permuted)) % n_folds, index=permuted)
    return households.map(fold_of).to_numpy()


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def installed_source(dist: str) -> str | None:
    """Return a distribution's version and source, or None if it is absent.

    For an install from git, it reports the commit that pip or uv recorded
    in the distribution's ``direct_url.json`` (PEP 610). It raises, rather
    than returning None, when the metadata names no version.
    """
    try:
        found = distribution(dist)
    except PackageNotFoundError:
        return None
    # ``metadata.get`` reads a missing field as None without the warning,
    # and later KeyError, that ``Distribution.version`` gives.
    found_version = found.metadata.get("Version")
    if not found_version:
        raise RuntimeError(f"{dist}'s distribution metadata names no version.")
    text = found.read_text("direct_url.json")
    if not text:
        return found_version
    direct = json.loads(text)
    source = direct.get("vcs_info", {}).get("commit_id") or direct.get("url")
    return f"{found_version} @ {source}"


def environment() -> dict[str, str | None]:
    """Return the versions of Python, ``PACKAGES`` and Microcosm.

    Microcosm's entries carry the commit each package came from, or None
    for an absent package. It raises when a Microcosm package imports
    without distribution metadata, since the run would then hash the
    package's source into the cache key without recording where it came
    from.
    """
    versions: dict[str, str | None] = {
        name: version(dist) for name, dist in PACKAGES.items()
    }
    for name, (dist, module) in MICROCOSM_PACKAGES.items():
        source = installed_source(dist)
        if source is None and installed_package_sha256(module) is not None:
            raise RuntimeError(
                f"{module} imports, but no {dist} distribution metadata "
                "records its version or source; reinstall it from git "
                "(see README.md)."
            )
        versions[name] = source
    return {"python": platform.python_version(), **versions}


def package_sha256(package: str) -> str:
    """Hash every ``.py`` file of an imported package, in path order."""
    module = importlib.import_module(package)
    root = Path(module.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def installed_package_sha256(package: str) -> str | None:
    """Return ``package_sha256``, or None when Python finds no such package.

    When the package exists but fails to import, the import error
    propagates.
    """
    try:
        return package_sha256(package)
    except ModuleNotFoundError as error:
        parts = package.split(".")
        parents = {".".join(parts[:i]) for i in range(1, len(parts) + 1)}
        if error.name in parents:
            return None
        raise


def require_microcosm() -> None:
    """Fail before any fitting when Microcosm's ``fit`` package is absent."""
    if installed_source("microcosm-fit") is None:
        raise SystemExit(
            "The Microcosm methods need microcosm-fit, which installs from "
            "git (see README.md). Pass --skip-forests to run without the "
            "forest methods."
        )


def frame_sha256(*frames: pd.DataFrame) -> str:
    """Hash frames' column names, row keys and values."""
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(json.dumps([str(c) for c in frame.columns]).encode())
        hashed = pd.util.hash_pandas_object(frame, index=True)
        digest.update(hashed.to_numpy().tobytes())
    return digest.hexdigest()[:16]


def build_config(
    args: argparse.Namespace,
    units: pd.DataFrame,
    X: pd.DataFrame,
    cells: pd.DataFrame,
    folds: np.ndarray,
) -> dict:
    """Collect everything that can change the fold imputations."""
    fold_frame = pd.DataFrame({"fold": folds}, index=units.index)
    return {
        "zip_sha256": ZIP_SHA256,
        "folds": args.folds,
        "draws": args.draws,
        "seed": args.seed,
        "skip_forests": args.skip_forests,
        "max_households": args.max_households,
        "features": list(X.columns),
        "min_cell_donors": MIN_CELL_DONORS,
        "pmm_neighbors": PMM_NEIGHBORS,
        "cutoff": PARTICIPATION_CUTOFF,
        "cell_levels": [list(level) for level in CELL_LEVELS],
        "source_sha256": {name: sha256_of(HERE / name)[:16] for name in SOURCES},
        "microimpute_sha256": package_sha256("microimpute"),
        "microcosm_sha256": {
            module: installed_package_sha256(module)
            for _, module in MICROCOSM_PACKAGES.values()
        },
        "environment": environment(),
        "input_sha256": frame_sha256(units, X, cells, fold_frame),
    }


def collect_folds(
    cache: Path,
    tag: str,
    folds: np.ndarray,
    n_folds: int,
    n_draws: int,
    fit_fold: Callable[[int], dict[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Load each fold's imputations from the cache, or fit and store them.

    Returns each method's draws for every unit, ``(n_draws, n_units)``,
    and the fitting time of each method summed over folds.
    """
    imputations: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    for fold in range(n_folds):
        path = cache / f"{tag}_fold{fold}.npz"
        if path.exists():
            stored = dict(np.load(path, allow_pickle=False))
        else:
            start = time.time()
            stored = fit_fold(fold)
            np.savez_compressed(path, **stored)
            print(
                f"fold {fold} done in {time.time() - start:.0f}s",
                flush=True,
            )
        names = [str(n) for n in stored.pop("_timing_names")]
        for name, seconds in zip(names, stored.pop("_timings")[0]):
            timings[name] = timings.get(name, 0.0) + float(seconds)
        recipient = folds == fold
        for method, draws in stored.items():
            if method not in imputations:
                imputations[method] = np.full((n_draws, len(folds)), np.nan)
            imputations[method][:, recipient] = draws
    return imputations, timings


def run_fold(
    fold: int,
    folds: np.ndarray,
    units: pd.DataFrame,
    X: pd.DataFrame,
    cells: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    """Fit every method on the other folds and impute this fold."""
    donor = folds != fold
    recipient = folds == fold
    data = FoldData(
        X_donor=X.loc[donor],
        moop_donor=units["moop"].to_numpy()[donor],
        weight_donor=units["unit_weight"].to_numpy()[donor],
        cells_donor=cells.loc[donor],
        group_donor=units["group"].to_numpy()[donor],
        X_recipient=X.loc[recipient],
        cells_recipient=cells.loc[recipient],
        group_recipient=units["group"].to_numpy()[recipient],
    )
    rng = np.random.default_rng([args.seed, fold])
    timings: dict[str, float] = {}
    out: dict[str, np.ndarray] = {}

    start = time.time()
    out.update(logit_family(data, args.draws, rng))
    timings["logit_family"] = time.time() - start

    start = time.time()
    out["pmm"] = pmm(data, args.draws, rng)
    timings["pmm"] = time.time() - start

    if not args.skip_forests:
        start = time.time()
        out["qrf"] = qrf(data, args.draws, seed=args.seed + fold, n_jobs=args.n_jobs)
        timings["qrf"] = time.time() - start

        start = time.time()
        out["zi_qrf"] = zi_qrf(data, args.draws, seed=args.seed + fold)
        timings["zi_qrf"] = time.time() - start

        # Both Microcosm fits share a seed, so their draws use the same
        # uniforms and differ only through the weights.
        for method, weighted in (
            ("microcosm_qrf", True),
            ("microcosm_qrf_unweighted", False),
        ):
            start = time.time()
            out[method] = microcosm_qrf(
                data,
                args.draws,
                seed=args.seed + fold,
                weighted=weighted,
                n_jobs=args.n_jobs,
            )
            timings[method] = time.time() - start
    out["_timings"] = np.array([list(timings.values())])
    out["_timing_names"] = np.array(list(timings.keys()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--draws", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--skip-forests",
        action="store_true",
        help="Skip the forest methods, microimpute's and Microcosm's "
        "(for quick checks).",
    )
    parser.add_argument(
        "--max-households",
        type=int,
        default=None,
        help="Subsample households (for quick checks).",
    )
    args = parser.parse_args()
    if not args.skip_forests:
        require_microcosm()

    output = args.output
    cache = output / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    persons = load_persons(args.data_dir)
    identity = check_spm_identity(persons)
    if any(value != 0 for value in identity.values()):
        raise ValueError(f"SPM identities fail: {identity}")
    if args.max_households:
        keep = (
            pd.Series(persons["PH_SEQ"].unique())
            .sample(args.max_households, random_state=args.seed)
            .to_numpy()
        )
        persons = persons[persons["PH_SEQ"].isin(keep)].reset_index(drop=True)
    units = build_units(persons)
    X = design_matrix(units)
    cells = cell_keys(units)
    folds = assign_folds(units["household"], args.folds, args.seed)

    config = build_config(args, units, X, cells, folds)
    tag = config_hash(config)

    imputations, timings = collect_folds(
        cache,
        tag,
        folds,
        args.folds,
        args.draws,
        lambda fold: run_fold(fold, folds, units, X, cells, args),
    )
    for method, draws in imputations.items():
        if np.isnan(draws).any():
            raise ValueError(f"{method} left units without an imputation")
        if (draws < 0).any():
            raise ValueError(f"{method} produced negative MOOP")
        if method in DETERMINISTIC:
            # Deterministic methods: one draw carries all the information.
            imputations[method] = draws[:1]
    imputations["no_moop"] = np.zeros((1, len(units)))

    results = evaluate(persons, units, imputations)
    allocated = allocation_flags(persons)
    results["data"] = {
        "survey": f"CPS ASEC {SURVEY_YEAR} public-use file",
        "reference_year": REFERENCE_YEAR,
        "zip_url": ZIP_URL,
        "zip_sha256": ZIP_SHA256,
        "n_persons": len(persons),
        "n_units": len(units),
        "n_households": int(units["household"].nunique()),
        "population_millions": float(persons["weight"].sum() / 1e6),
        "n_units_by_fold": np.bincount(folds).tolist(),
        "unit_group_counts": units["group"].value_counts().to_dict(),
        "share_persons_moop_allocated": float(allocated["allocated_moop"].mean()),
        "share_persons_premium_allocated": float(allocated["allocated_premium"].mean()),
        "share_persons_any_allocated": float(allocated.max(axis=1).mean()),
        "share_units_any_allocated": float(units["any_allocated"].mean()),
        "share_units_reported_zero": float((units["moop"] == 0).mean()),
        "identity_checks": identity,
    }
    results["published"] = PUBLISHED
    results["config"] = {**config, "tag": tag, "n_jobs": args.n_jobs}
    results["timings_seconds"] = timings
    # The cache tag covers this environment, so it describes the cached
    # imputations as well as this run.
    results["environment"] = config["environment"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {output / 'results.json'}")


if __name__ == "__main__":
    main()
