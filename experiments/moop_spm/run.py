"""Cross-fit MOOP imputations on the 2026 CPS ASEC and compare SPM poverty.

Each SPM unit's reported medical out-of-pocket spending (MOOP) is
replaced by an imputation from models fit on the other folds (folds
group whole households), so every unit gets an out-of-fold imputation.
SPM poverty is then recomputed for every person under each method.

Usage (from the repository root):

    python -m experiments.moop_spm.run --data-dir ~/data/cps_asec_2026

The first run downloads the Census public-use zip (139 MB) into
``--data-dir`` and checks its SHA-256. Per-fold imputations are cached in
``<output>/cache`` so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .cps_asec import (
    REFERENCE_YEAR,
    SURVEY_YEAR,
    ZIP_SHA256,
    ZIP_URL,
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
    pmm,
    qrf,
    zi_qrf,
)

HERE = Path(__file__).resolve().parent
SOURCES = ["cps_asec.py", "methods.py", "evaluate.py", "run.py"]
PUBLISHED = json.loads((HERE / "published_2025.json").read_text())


def assign_folds(households: pd.Series, n_folds: int, seed: int) -> np.ndarray:
    """Assign whole households to folds at random, as evenly as possible."""
    unique = np.sort(households.unique())
    permuted = np.random.default_rng(seed).permutation(unique)
    fold_of = pd.Series(np.arange(len(permuted)) % n_folds, index=permuted)
    return households.map(fold_of).to_numpy()


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


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
        help="Skip QRF and ZI-QRF (for quick checks).",
    )
    parser.add_argument(
        "--max-households",
        type=int,
        default=None,
        help="Subsample households (for quick checks).",
    )
    args = parser.parse_args()

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

    config = {
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
    }
    tag = config_hash(config)

    imputations: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    for fold in range(args.folds):
        path = cache / f"{tag}_fold{fold}.npz"
        if path.exists():
            stored = dict(np.load(path, allow_pickle=False))
        else:
            start = time.time()
            stored = run_fold(fold, folds, units, X, cells, args)
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
                imputations[method] = np.full((args.draws, len(units)), np.nan)
            imputations[method][:, recipient] = draws
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
    allocated = persons["I_MOOP"].isin([1, 2, 3])
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
        "share_persons_moop_allocated": float(allocated.mean()),
        "share_units_any_allocated": float(units["any_allocated_moop"].mean()),
        "share_units_reported_zero": float((units["moop"] == 0).mean()),
        "identity_checks": identity,
    }
    results["published"] = PUBLISHED
    results["config"] = {**config, "tag": tag, "n_jobs": args.n_jobs}
    results["timings_seconds"] = timings
    results["environment"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "microimpute": _version("microimpute"),
        "quantile_forest": _version("quantile-forest"),
        "scikit_learn": _version("scikit-learn"),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {output / 'results.json'}")


def _version(package: str) -> str:
    from importlib.metadata import version

    return version(package)


if __name__ == "__main__":
    main()
