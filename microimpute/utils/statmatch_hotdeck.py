"""Statistical matching hot deck imputation utilities.

This module provides an interface to R's StatMatch package for performing nearest neighbor
distance hot deck matching.
"""

import logging
from contextlib import contextmanager
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from pydantic import validate_call

from microimpute.config import VALIDATE_CONFIG

log = logging.getLogger(__name__)

"""
data.rec: A matrix or data frame that plays the role of recipient in the
statistical matching application.

data.don: A matrix or data frame that that plays the role of donor in the statistical matching application.

mtc.ids: A matrix with two columns. Each row must contain the name or the index of the recipient record (row) in data.don and the name or the index of the corresponding donor record (row) in data.don. Note that this type of matrix is returned by the functions NND.hotdeck, RANDwNND.hotdeck, rankNND.hotdeck, and mixed.mtc.

z.vars: A character vector with the names of the variables available only in data.don that should be "donated" to data.rec.
"""
import os

# Set env vars early, before rpy2 does anything
# RPY2_CFFI_MODE=ABI skips API mode and avoids the dlopen warning
os.environ["RPY2_CFFI_MODE"] = "ABI"

import rpy2.robjects as ro
from rpy2.robjects import conversion, default_converter, numpy2ri, pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

# Cache R package imports so they only happen once, avoiding repeated
# network calls to download CRAN mirror lists during hyperparameter tuning.
_statmatch_cache = {}


def _get_statmatch():
    """Return the cached StatMatch R package, importing it only once."""
    if "StatMatch" not in _statmatch_cache:
        _statmatch_cache["StatMatch"] = importr("StatMatch")
    return _statmatch_cache["StatMatch"]


@contextmanager
def _temporary_r_seed(seed):
    """Seed a bridge call without changing the caller's R random stream."""
    if seed is None:
        yield
        return
    had_state = ".Random.seed" in ro.globalenv
    original_state = (
        ro.IntVector(list(ro.globalenv[".Random.seed"])) if had_state else None
    )
    try:
        ro.r["set.seed"](int(seed))
        yield
    finally:
        if had_state:
            ro.globalenv[".Random.seed"] = original_state
        elif ".Random.seed" in ro.globalenv:
            del ro.globalenv[".Random.seed"]


@validate_call(config=VALIDATE_CONFIG)
def nnd_hotdeck_using_rpy2(
    receiver: pd.DataFrame,
    donor: pd.DataFrame,
    matching_variables: List[str],
    z_variables: List[str],
    **matching_kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform nearest neighbor distance hot deck matching using R's StatMatch package.

    Args:
        receiver: DataFrame containing recipient data.
        donor: DataFrame containing donor data.
        matching_variables: List of column names to use for matching.
        z_variables: List of column names to donate from donor to recipient.
        **matching_kwargs: Optional hyperparameters for matching.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two fused DataFrames:
            - First without duplication of matching variables
            - Second with duplication of matching variables


    Raises:
        ValueError: If any of the matching variables or z variables are not found in the respective DataFrames.
        RuntimeError: If there is an unexpected error during the statistical matching process.
    """

    StatMatch = _get_statmatch()

    try:
        missing_in_receiver = [
            v for v in matching_variables if v not in receiver.columns
        ]
        missing_in_donor = [v for v in matching_variables if v not in donor.columns]
        if missing_in_receiver:
            msg = f"Matching variables missing in receiver: {missing_in_receiver}"
            log.error(msg)
            raise ValueError(msg)
        if missing_in_donor:
            msg = f"Matching variables missing in donor: {missing_in_donor}"
            log.error(msg)
            raise ValueError(msg)
        missing_z = [v for v in z_variables if v not in donor.columns]
        if missing_z:
            msg = f"Z variables missing in donor: {missing_z}"
            log.error(msg)
            raise ValueError(msg)

        # NND.hotdeck has no weight argument. RANDwNND.hotdeck accepts
        # weight.don as the NAME of a column in data.don, not a vector.
        # cut.don="min" preserves nearest-distance matching and weights ties.
        # https://search.r-project.org/CRAN/refmans/StatMatch/html/RANDwNND.hotdeck.html
        r_kwargs = dict(matching_kwargs)
        random_state = r_kwargs.pop("random_state", None)
        donor_sample_weight = r_kwargs.pop("donor_sample_weight", None)
        donor_for_matching = donor
        matching_function = StatMatch.NND_hotdeck
        if donor_sample_weight is not None:
            weights = np.asarray(donor_sample_weight, dtype=float)
            if weights.ndim != 1 or len(weights) != len(donor):
                raise ValueError("Donor weights must contain one value per donor")
            if not np.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError("Donor weights must be positive and finite")
            if r_kwargs.pop("constrained", False):
                raise ValueError(
                    "Weighted constrained matching is not supported by RANDwNND.hotdeck"
                )
            r_kwargs.pop("constr_alg", None)
            if "k" in r_kwargs and "cut_don" not in r_kwargs:
                raise ValueError(
                    "Weighted matching with k requires an explicit cut_don rule"
                )
            weight_column = "__microimpute_donor_weight__"
            while weight_column in donor.columns:
                weight_column += "_"
            donor_for_matching = donor.copy()
            donor_for_matching[weight_column] = weights
            r_kwargs["weight_don"] = weight_column
            r_kwargs.setdefault("cut_don", "min")
            matching_function = StatMatch.RANDwNND_hotdeck

        with localconverter(
            default_converter + pandas2ri.converter + numpy2ri.converter
        ):
            r_receiver = conversion.py2rpy(receiver)
            r_donor = conversion.py2rpy(donor_for_matching)
            r_match = ro.StrVector(matching_variables)
            r_z = ro.StrVector(z_variables)

        with _temporary_r_seed(random_state):
            out_NND = matching_function(
                data_rec=r_receiver,
                data_don=r_donor,
                match_vars=r_match,
                **r_kwargs,
            )

        # Create the correct matching indices matrix for StatMatch.create_fused
        recipient_indices = np.arange(1, len(receiver) + 1)
        mtc_ids_r = out_NND.rx2("mtc.ids")
        log.debug(f"mtc_ids_r type: {type(mtc_ids_r)}")

        # Create the properly formatted 2-column matrix that create_fused expects
        if hasattr(mtc_ids_r, "ncol") and mtc_ids_r.ncol == 2:
            # Already a matrix with the right shape, use it directly
            log.debug("Using mtc_ids_r directly as 2-column matrix")
            mtc_ids = mtc_ids_r
        else:
            mtc_array = np.array(mtc_ids_r)
            log.debug(f"mtc_array shape: {mtc_array.shape}, dtype: {mtc_array.dtype}")
            log.debug(f"Receiver length: {len(receiver)}, Donor length: {len(donor)}")

            # If we have a 1D array with strings, convert to integers
            if mtc_array.dtype.kind in ["U", "S"]:
                mtc_array = np.array([int(x) for x in mtc_array])

            # Check if mtc_array is empty or has unexpected shape
            if mtc_array.size == 0:
                log.error("mtc_array is empty!")
                raise ValueError("No matching indices returned from NND_hotdeck")

            # If the mtc.ids array has 2 values per recipient
            # (recipient_idx, donor_idx pairs).
            if len(mtc_array) == 2 * len(receiver):
                donor_indices = mtc_array.reshape(-1, 2)[:, 1]
                # StatMatch uses 1-based indexing; valid donor indices are
                # in [1, len(donor)]. Previously we silently
                # modulo-wrapped out-of-range indices, masking real
                # StatMatch/R indexing bugs and silently assigning a
                # wrong donor. Raise loudly so the caller notices.
                out_of_range = (donor_indices < 1) | (donor_indices > len(donor))
                if out_of_range.any():
                    n_bad = int(out_of_range.sum())
                    raise ValueError(
                        f"StatMatch returned {n_bad} donor index/indices "
                        f"out of range [1, {len(donor)}]; this indicates "
                        "a bug in the StatMatch R call, not a recoverable "
                        "condition. Check input data for NaNs, infinite "
                        "values, or inconsistent dtypes between donor and "
                        "recipient."
                    )
                donor_indices_valid = donor_indices
            elif len(mtc_array) == len(receiver):
                # Flat 1-D array of donor indices, one per recipient.
                out_of_range = (mtc_array < 1) | (mtc_array > len(donor))
                if out_of_range.any():
                    n_bad = int(out_of_range.sum())
                    raise ValueError(
                        f"StatMatch returned {n_bad} donor index/indices "
                        f"out of range [1, {len(donor)}]"
                    )
                donor_indices_valid = mtc_array
            elif len(mtc_array) > len(receiver):
                # More indices than recipients is unexpected; refuse to
                # silently truncate (previously we kept just the first
                # len(receiver) items, which could drop a valid pairing).
                raise ValueError(
                    f"StatMatch returned {len(mtc_array)} match indices "
                    f"for {len(receiver)} recipients; expected exactly "
                    f"{len(receiver)} (1-D) or {2 * len(receiver)} "
                    "(paired). Refusing to silently truncate."
                )
            else:
                # Fewer matches than recipients. Previously the last
                # match was repeated for every missing recipient,
                # producing severe homogeneity bias in the imputed
                # column — entirely invisible to callers. Raise loudly
                # instead; the caller can implement a proper fallback
                # (e.g. random-donor sample with a warning) if
                # appropriate for their data.
                raise ValueError(
                    f"StatMatch returned only {len(mtc_array)} match "
                    f"indices for {len(receiver)} recipients. "
                    "Previously the last match was repeated to fill "
                    "the gap, producing severe homogeneity bias. This "
                    "is now rejected; investigate why StatMatch "
                    "under-matched (empty mtc.ids, dtype mismatch, "
                    "NaN predictors, etc.)."
                )
            # Create the final mtc.ids matrix required by create_fused
            mtc_matrix = np.column_stack((recipient_indices, donor_indices_valid))
            # Convert to R matrix
            mtc_ids = ro.r.matrix(
                ro.IntVector(mtc_matrix.flatten(order="F")),
                nrow=len(recipient_indices),
                ncol=2,
            )

        fused_0_r = StatMatch.create_fused(
            data_rec=r_receiver,
            data_don=r_donor,
            mtc_ids=mtc_ids,
            z_vars=r_z,
        )
        fused_1_r = StatMatch.create_fused(
            data_rec=r_receiver,
            data_don=r_donor,
            mtc_ids=mtc_ids,
            z_vars=r_z,
            dup_x=False,
            match_vars=r_match,
        )

        with localconverter(
            default_converter + pandas2ri.converter + numpy2ri.converter
        ):
            fused_0 = conversion.rpy2py(fused_0_r)
            fused_1 = conversion.rpy2py(fused_1_r)

        return fused_0, fused_1

    except ValueError:
        raise
    except IndexError as e:
        log.error(f"Index error in statistical matching: {e}")
        log.error(f"Receiver shape: {receiver.shape}, Donor shape: {donor.shape}")
        log.error(f"Matching variables: {matching_variables}")
        log.error(f"Z variables: {z_variables}")
        raise RuntimeError(f"Statistical matching failed with index error: {e}") from e
    except Exception as e:
        log.error(f"Unexpected error in statistical matching: {e}")
        log.error(f"Error type: {type(e).__name__}")
        raise RuntimeError(
            f"Statistical matching failed with unexpected error: {e}"
        ) from e
