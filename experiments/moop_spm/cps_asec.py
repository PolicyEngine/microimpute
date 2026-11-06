"""Load the 2026 CPS ASEC public-use file into person and SPM-unit frames.

The 2026 CPS ASEC (fielded February to April 2026) reports calendar-year
2025 income and carries Census's Supplemental Poverty Measure (SPM)
variables for every person. This module downloads the public-use CSV
release, checks its SHA-256, and builds:

- a person frame with SPM resources, thresholds, poverty status, the
  person weight and Census's 160 replicate weights; and
- an SPM-unit frame with reported medical out-of-pocket spending (MOOP,
  ``SPM_MEDXPNS``) and the predictors an ACS-style recipient file carries.

Variable names and codes follow the Census data dictionary
``asec2026_ddl_pub_full.pdf`` shipped with the release.
"""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SURVEY_YEAR = 2026
REFERENCE_YEAR = 2025
ZIP_URL = (
    "https://www2.census.gov/programs-surveys/cps/datasets/2026/march/asecpub26csv.zip"
)
ZIP_SHA256 = "fe819d0d2fc4470c282e76c247b8aa8b6054811619730307e9f2fe6186707d93"
PERSON_CSV = "pppub26.csv"
HOUSEHOLD_CSV = "hhpub26.csv"
REPWGT_CSV = "asec_csv_repwgt_2026.csv"
N_REPLICATES = 160

PERSON_COLUMNS = [
    "PH_SEQ",
    "PPPOS",
    "A_AGE",
    "A_SEX",
    "PRDISFLG",
    "MARSUPWT",
    "I_MOOP",
    "MOOP",
    "PEMCPREM",
    "NOW_COV",
    "NOW_GRP",
    "NOW_DIR",
    "NOW_MCARE",
    "NOW_MCAID",
    "NOW_MIL",
    "NOW_VACARE",
    "NOW_CHAMPVA",
    "SPM_ID",
    "SPM_HEAD",
    "SPM_WEIGHT",
    "SPM_RESOURCES",
    "SPM_POVTHRESHOLD",
    "SPM_POOR",
    "SPM_MEDXPNS",
    "SPM_TOTVAL",
    "SPM_SNAPSUB",
    "SPM_CAPHOUSESUB",
    "SPM_SCHLUNCH",
    "SPM_ENGVAL",
    "SPM_WICVAL",
    "SPM_FEDTAX",
    "SPM_STTAX",
    "SPM_FICA",
    "SPM_CAPWKCCXPNS",
    "SPM_CHILDSUPPD",
    "SPM_NUMPER",
    "SPM_FAMTYPE",
    "SPM_TENMORTSTATUS",
]

# Current coverage at interview (NOW_*), the closest analogue of the
# ACS coverage question. Codes: 1 = yes, 2 = no.
COVERAGE_ORDER = [
    "medicare",
    "employer",
    "direct",
    "medicaid",
    "military",
    "uninsured",
]
COVERAGE_LABELS = {
    "medicare": "Medicare",
    "employer": "Employer",
    "direct": "Direct purchase",
    "medicaid": "Medicaid or other means-tested",
    "military": "TRICARE, VA or CHAMPVA",
    "uninsured": "Uninsured",
}
# Coverage groups for SPM units, used to define imputation cells and
# PMM groups. TRICARE/VA units are folded into the public group.
UNIT_GROUPS = ["medicare", "employer", "direct", "public", "uninsured"]

AGE_BANDS = [(0, 17), (18, 44), (45, 64), (65, 74), (75, 200)]


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_raw(data_dir: Path) -> Path:
    """Download and unpack the ASEC CSV release into ``data_dir``.

    The zip is fetched once, checked against the pinned SHA-256, and the
    three CSVs this experiment reads are extracted next to it.

    Args:
        data_dir: Directory that holds (or will hold) the raw files.

    Returns:
        The data directory.

    Raises:
        ValueError: If the zip's digest does not match ``ZIP_SHA256``.
    """
    data_dir = Path(data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "asecpub26csv.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(ZIP_URL, zip_path)
    digest = sha256_of(zip_path)
    if digest != ZIP_SHA256:
        raise ValueError(
            f"{zip_path} has SHA-256 {digest}, expected {ZIP_SHA256}. "
            "Census may have replaced the file; check before rerunning."
        )
    with zipfile.ZipFile(zip_path) as archive:
        for name in (PERSON_CSV, HOUSEHOLD_CSV, REPWGT_CSV):
            if not (data_dir / name).exists():
                archive.extract(name, data_dir)
    return data_dir


def coverage_category(persons: pd.DataFrame) -> pd.Series:
    """Assign each person one current-coverage category.

    The hierarchy is Medicare, employer, direct purchase, Medicaid or
    other means-tested, TRICARE/VA/CHAMPVA, then uninsured. A person with
    Medicare and Medicaid counts as Medicare.
    """
    military = (
        (persons["NOW_MIL"] == 1)
        | (persons["NOW_VACARE"] == 1)
        | (persons["NOW_CHAMPVA"] == 1)
    )
    conditions = [
        persons["NOW_MCARE"] == 1,
        persons["NOW_GRP"] == 1,
        persons["NOW_DIR"] == 1,
        persons["NOW_MCAID"] == 1,
        military,
        persons["NOW_COV"] == 2,
    ]
    category = np.select(conditions, COVERAGE_ORDER, default="unclassified")
    return pd.Series(category, index=persons.index, name="coverage")


def load_persons(data_dir: Path) -> pd.DataFrame:
    """Read person records, attach state and replicate weights.

    Returns one row per ASEC person record with the columns in
    ``PERSON_COLUMNS`` plus ``state`` (GESTFIPS), ``weight`` (MARSUPWT
    divided by 100, the person weight in people), ``coverage`` and
    ``repwgt_1`` to ``repwgt_160``.
    """
    data_dir = ensure_raw(data_dir)
    persons = pd.read_csv(data_dir / PERSON_CSV, usecols=PERSON_COLUMNS)
    households = pd.read_csv(
        data_dir / HOUSEHOLD_CSV, usecols=["H_SEQ", "GESTFIPS"]
    ).rename(columns={"H_SEQ": "PH_SEQ", "GESTFIPS": "state"})
    persons = persons.merge(households, on="PH_SEQ", how="left", validate="many_to_one")
    if persons["state"].isna().any():
        raise ValueError("Some persons have no household record.")

    replicate_columns = [f"pwwgt{i}" for i in range(1, N_REPLICATES + 1)]
    replicates = pd.read_csv(
        data_dir / REPWGT_CSV,
        usecols=["PPPOS", "h_seq", *replicate_columns],
    ).rename(columns={"h_seq": "PH_SEQ"})
    replicates = replicates.rename(
        columns={f"pwwgt{i}": f"repwgt_{i}" for i in range(1, N_REPLICATES + 1)}
    )
    persons = persons.merge(
        replicates, on=["PH_SEQ", "PPPOS"], how="left", validate="one_to_one"
    )
    if persons["repwgt_1"].isna().any():
        raise ValueError("Some persons have no replicate weights.")

    persons["weight"] = persons["MARSUPWT"] / 100.0
    persons["coverage"] = coverage_category(persons)
    if (persons["coverage"] == "unclassified").any():
        raise ValueError("Coverage hierarchy left persons unclassified.")
    return persons.reset_index(drop=True)


def check_spm_identity(persons: pd.DataFrame) -> dict[str, float]:
    """Check how SPM resources and MOOP are assembled in the file.

    Returns the maximum absolute gap for two identities that must hold
    for the imputation exercise to be valid:

    - ``resources_gap``: SPM_RESOURCES against cash income plus noncash
      benefits less taxes, capped work and child care expenses, MOOP and
      child support paid.
    - ``poor_mismatch``: share of persons whose SPM_POOR differs from
      ``SPM_RESOURCES < SPM_POVTHRESHOLD``.
    - ``moop_gap``: SPM_MEDXPNS against the unit sum of person MOOP plus
      the unit sum of PEMCPREM (edited Medicare premium).
    """
    rebuilt = (
        persons["SPM_TOTVAL"]
        + persons["SPM_SNAPSUB"]
        + persons["SPM_CAPHOUSESUB"]
        + persons["SPM_SCHLUNCH"]
        + persons["SPM_ENGVAL"]
        + persons["SPM_WICVAL"]
        - persons["SPM_FEDTAX"]
        - persons["SPM_STTAX"]
        - persons["SPM_FICA"]
        - persons["SPM_CAPWKCCXPNS"]
        - persons["SPM_MEDXPNS"]
        - persons["SPM_CHILDSUPPD"]
    )
    poor = (persons["SPM_RESOURCES"] < persons["SPM_POVTHRESHOLD"]).astype(int)
    by_unit = persons.groupby("SPM_ID").agg(
        moop=("MOOP", "sum"),
        premium=("PEMCPREM", "sum"),
        reported=("SPM_MEDXPNS", "first"),
    )
    return {
        "resources_gap": float((rebuilt - persons["SPM_RESOURCES"]).abs().max()),
        "poor_mismatch": float((poor != persons["SPM_POOR"]).mean()),
        "moop_gap": float(
            (by_unit["reported"] - by_unit["moop"] - by_unit["premium"]).abs().max()
        ),
    }


def _signed_log1p(values: pd.Series) -> pd.Series:
    return np.sign(values) * np.log1p(values.abs())


def build_units(persons: pd.DataFrame) -> pd.DataFrame:
    """Collapse persons to one row per SPM unit.

    The frame holds the imputation target ``moop`` (SPM_MEDXPNS), the
    unit's SPM resources and threshold, its weight (``SPM_WEIGHT`` / 100,
    which equals the head's person weight), and ACS-style predictors:
    age, sex, disability, current coverage, cash income, family size and
    type, tenure and state.
    """
    persons = persons.copy()
    persons["disabled"] = (persons["PRDISFLG"] == 1).astype(int)
    persons["female_adult"] = (
        (persons["A_SEX"] == 2) & (persons["A_AGE"] >= 18)
    ).astype(int)
    persons["allocated_moop"] = persons["I_MOOP"].isin([1, 2, 3]).astype(int)
    for low, high in AGE_BANDS:
        name = f"n_age_{low}_{high}" if high < 200 else f"n_age_{low}_plus"
        persons[name] = persons["A_AGE"].between(low, high).astype(int)
    military = (
        (persons["NOW_MIL"] == 1)
        | (persons["NOW_VACARE"] == 1)
        | (persons["NOW_CHAMPVA"] == 1)
    )
    # Coverage counts are not mutually exclusive: a person with two
    # coverage types counts toward both.
    any_coverage = {
        "n_any_medicare": persons["NOW_MCARE"] == 1,
        "n_any_employer": persons["NOW_GRP"] == 1,
        "n_any_direct": persons["NOW_DIR"] == 1,
        "n_any_medicaid": persons["NOW_MCAID"] == 1,
        "n_any_military": military,
        "n_uninsured": persons["NOW_COV"] == 2,
    }
    for name, flag in any_coverage.items():
        persons[name] = flag.astype(int)

    count_columns = [
        c for c in persons.columns if c.startswith("n_age_") or c in any_coverage
    ]
    count_columns += ["disabled", "female_adult", "allocated_moop"]
    sums = persons.groupby("SPM_ID")[count_columns].sum()
    sums = sums.rename(
        columns={
            "disabled": "n_disabled",
            "female_adult": "n_female_adult",
            "allocated_moop": "n_allocated_moop",
        }
    )
    heads = persons.loc[persons["SPM_HEAD"] == 1].set_index("SPM_ID")
    if not heads.index.is_unique or len(heads) != persons["SPM_ID"].nunique():
        raise ValueError("Expected exactly one SPM head per SPM unit.")
    units = pd.DataFrame(
        {
            "household": heads["PH_SEQ"],
            "unit_weight": heads["SPM_WEIGHT"] / 100.0,
            "moop": heads["SPM_MEDXPNS"].astype(float),
            "resources": heads["SPM_RESOURCES"].astype(float),
            "threshold": heads["SPM_POVTHRESHOLD"].astype(float),
            "cash_income": heads["SPM_TOTVAL"].astype(float),
            "n_persons": heads["SPM_NUMPER"].astype(int),
            "famtype": heads["SPM_FAMTYPE"].astype(int),
            "tenure": heads["SPM_TENMORTSTATUS"].astype(int),
            "state": heads["state"].astype(int),
            "head_age": heads["A_AGE"].astype(float),
            "head_female": (heads["A_SEX"] == 2).astype(int),
        }
    ).join(sums)
    units["pre_moop_ratio"] = (units["resources"] + units["moop"]) / units["threshold"]
    units["log_cash_income"] = _signed_log1p(units["cash_income"])
    units["cash_income_ratio"] = (units["cash_income"] / units["threshold"]).clip(
        -1.0, 10.0
    )
    units["group"] = unit_coverage_group(units)
    units["any_allocated_moop"] = (units["n_allocated_moop"] > 0).astype(int)
    return units.sort_index()


def unit_coverage_group(units: pd.DataFrame) -> pd.Series:
    """Classify SPM units by the coverage of their members.

    Any member with Medicare makes the unit ``medicare``; otherwise any
    employer coverage makes it ``employer``; then direct purchase; then
    Medicaid, other means-tested, TRICARE or VA (``public``); units with
    none of these are ``uninsured``.
    """
    conditions = [
        units["n_any_medicare"] > 0,
        units["n_any_employer"] > 0,
        units["n_any_direct"] > 0,
        (units["n_any_medicaid"] > 0) | (units["n_any_military"] > 0),
    ]
    group = np.select(conditions, UNIT_GROUPS[:4], default="uninsured")
    return pd.Series(group, index=units.index, name="group")


NUMERIC_FEATURES: list[str] = [
    "head_age",
    "head_female",
    "n_age_0_17",
    "n_age_18_44",
    "n_age_45_64",
    "n_age_65_74",
    "n_age_75_plus",
    "n_female_adult",
    "n_disabled",
    "n_any_medicare",
    "n_any_employer",
    "n_any_direct",
    "n_any_medicaid",
    "n_any_military",
    "n_uninsured",
    "log_cash_income",
    "cash_income_ratio",
]
CATEGORICAL_FEATURES: list[str] = ["famtype", "tenure", "state"]


def design_matrix(units: pd.DataFrame) -> pd.DataFrame:
    """Return the predictor matrix shared by every imputation method.

    Numeric predictors enter as-is; family type, tenure and state enter
    as 0/1 indicators with the lowest code dropped. Every method sees
    exactly these columns.
    """
    frames = [units[NUMERIC_FEATURES].astype(float)]
    for column in CATEGORICAL_FEATURES:
        dummies = pd.get_dummies(
            units[column].astype(int), prefix=column, drop_first=True
        ).astype(float)
        frames.append(dummies)
    matrix = pd.concat(frames, axis=1)
    matrix.columns = [str(c) for c in matrix.columns]
    return matrix
