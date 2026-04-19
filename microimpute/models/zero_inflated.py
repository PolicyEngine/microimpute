"""Regime-aware zero-inflation wrapper around base imputers.

Tabular microdata variables often fall into distinct *regimes* based on
which of {negative, zero, positive} values appear in the training data.
Imputing them with a single regressor mixes regimes together, causing
two recurring bugs in downstream ecosystems:

1. **Negative-dropping.** The common "fit QRF on ``y > 0``" pattern
   drops negative training rows along with zeros, so the imputer
   produces zero or positive values only. Variables like
   ``short_term_capital_gains`` lose their entire negative tail.

2. **Zero-crossing interpolation.** A QRF fit on all nonzero values
   (both signs) learns leaf distributions that interpolate between
   positive and negative training rows. Predictions for records that
   the gate marks "nonzero" can land in the interval between
   ``max(train_negatives)`` and ``min(train_positives)``, which is
   not a region any actual record occupies.

``ZeroInflatedImputer`` wraps any base ``Imputer`` and:

- Detects the regime automatically at fit time from the training
  distribution — no per-variable hand configuration required.
- Composes the base imputer with appropriate gate(s):
  - Three-sign: gate chooses ``{neg, 0, pos}``; separate base
    imputers on the positive and negative subsets.
  - ZI positive / ZI negative: binary gate (``0`` vs nonzero); base
    imputer on the nonzero-sign subset.
  - Sign-only (no zero): binary sign gate; two base imputers.
  - Single-sign or constant: no gate; direct base imputer or a
    constant imputer.
- At predict time, routes each record to the base imputer of its
  gate-assigned regime, guaranteeing no sign-interpolation leaks.

The wrapper is generic over the base imputer — ``QRF`` is the obvious
default, but ``MDN``, ``OLS``, or ``Matching`` all compose the same way.

Regime detection is parameterized by ``min_class_count`` and
``min_class_fraction``: a class with fewer observations than both
thresholds collapses into the closest adjacent regime. This avoids
fitting a full three-sign split on a variable whose negative tail is
five outlier rows — the cost-benefit flips toward the simpler
architecture.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from pydantic import SkipValidation, validate_call

from microimpute.config import RANDOM_STATE, VALIDATE_CONFIG
from microimpute.models.imputer import (
    Imputer,
    ImputerResults,
    _ConstantValueModel,
)
from microimpute.models.qrf import QRF


# Regime labels. Kept as module-level constants so downstream code can
# match on them without magic strings.
REGIME_THREE_SIGN = "THREE_SIGN"
REGIME_ZI_POSITIVE = "ZI_POSITIVE"
REGIME_ZI_NEGATIVE = "ZI_NEGATIVE"
REGIME_SIGN_ONLY = "SIGN_ONLY"
REGIME_POSITIVE_ONLY = "POSITIVE_ONLY"
REGIME_NEGATIVE_ONLY = "NEGATIVE_ONLY"
REGIME_DEGENERATE_ZERO = "DEGENERATE_ZERO"


def _make_classifier(kind: str, seed: int):
    """Build a sklearn classifier for the zero-gate.

    ``hist_gb`` (default): ``HistGradientBoostingClassifier``. On the
    isolated-log-loss benchmark over 26 zero-inflated PolicyEngine-US
    target variables this Pareto-dominated a 50-tree RF on log-loss
    (0.225 vs 0.310), Brier (0.071 vs 0.081), ECE (0.005 vs 0.039),
    and ROC-AUC (0.809 vs 0.737).
    """
    if kind == "hist_gb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(random_state=seed)
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=50, random_state=seed, n_jobs=-1
        )
    raise ValueError(
        f"Unknown classifier_type {kind!r}; expected 'hist_gb' or 'rf'."
    )


def _detect_regime(
    y: np.ndarray,
    *,
    min_class_count: int,
    min_class_fraction: float,
    zero_atol: float,
) -> str:
    """Classify the training distribution into one of seven regimes.

    A class (neg/zero/pos) counts as present iff its count is at least
    ``min_class_count`` AND its fraction of total rows is at least
    ``min_class_fraction``. Below both thresholds, the class collapses
    into its closest adjacent regime (minority negatives merge into
    zero → ZI_POSITIVE; minority zeros merge into the majority sign;
    etc.). This keeps the gate architecture stable in the presence of
    measurement-error outliers.
    """
    n = len(y)
    if n == 0:
        return REGIME_DEGENERATE_ZERO

    is_zero = np.abs(y) <= zero_atol
    is_pos = y > zero_atol
    is_neg = y < -zero_atol

    n_zero = int(is_zero.sum())
    n_pos = int(is_pos.sum())
    n_neg = int(is_neg.sum())

    # Apply both thresholds.
    def _meaningful(count: int) -> bool:
        return count >= min_class_count and (count / n) >= min_class_fraction

    has_zero = _meaningful(n_zero)
    has_pos = _meaningful(n_pos)
    has_neg = _meaningful(n_neg)

    if not (has_zero or has_pos or has_neg):
        # All three classes are below threshold. Pick the one with the
        # largest raw count as a degenerate fallback.
        counts = {"zero": n_zero, "pos": n_pos, "neg": n_neg}
        majority = max(counts, key=counts.get)
        if majority == "zero":
            return REGIME_DEGENERATE_ZERO
        return REGIME_POSITIVE_ONLY if majority == "pos" else REGIME_NEGATIVE_ONLY

    if has_pos and has_neg and has_zero:
        return REGIME_THREE_SIGN
    if has_pos and has_neg:
        return REGIME_SIGN_ONLY
    if has_pos and has_zero:
        return REGIME_ZI_POSITIVE
    if has_neg and has_zero:
        return REGIME_ZI_NEGATIVE
    if has_pos:
        return REGIME_POSITIVE_ONLY
    if has_neg:
        return REGIME_NEGATIVE_ONLY
    return REGIME_DEGENERATE_ZERO


class ZeroInflatedImputer(Imputer):
    """Imputer that wraps a base Imputer with regime-aware zero-gating.

    Args:
        base_imputer_class: ``Imputer`` subclass to use for the nonzero
            regression step. Defaults to ``QRF``.
        base_imputer_kwargs: Keyword arguments forwarded to the base
            imputer constructor. ``{}`` by default.
        min_class_count: Minimum raw count per class (neg/0/pos) for
            that class to be considered present. Below this, the class
            collapses into an adjacent regime. Defaults to 10.
        min_class_fraction: Minimum fraction of total rows per class
            for that class to be considered present. Defaults to 0.01.
        zero_atol: Absolute tolerance for "equals zero" in the regime
            detector. Defaults to 1e-6, matching the upstream
            ``_MultiSourceBase`` convention.
        classifier_type: Backend for the gate classifier;
            ``"hist_gb"`` (default) or ``"rf"``.
        seed: Random seed.
        log_level: Python logging level.
    """

    def __init__(
        self,
        base_imputer_class: Optional[Type[Imputer]] = None,
        base_imputer_kwargs: Optional[Dict[str, Any]] = None,
        min_class_count: int = 10,
        min_class_fraction: float = 0.01,
        zero_atol: float = 1e-6,
        classifier_type: str = "hist_gb",
        seed: Optional[int] = RANDOM_STATE,
        log_level: Optional[str] = "WARNING",
    ) -> None:
        super().__init__(seed=seed, log_level=log_level)
        self.base_imputer_class = base_imputer_class or QRF
        self.base_imputer_kwargs = dict(base_imputer_kwargs or {})
        self.min_class_count = int(min_class_count)
        self.min_class_fraction = float(min_class_fraction)
        self.zero_atol = float(zero_atol)
        self.classifier_type = classifier_type

        # Filled in during fit().
        self._regimes: Dict[str, str] = {}
        self._per_variable: Dict[str, Dict[str, Any]] = {}

    def _fit(self, *args: Any, **kwargs: Any) -> Any:
        """Abstract-method placeholder; this class overrides ``fit`` directly."""
        raise NotImplementedError(
            "ZeroInflatedImputer overrides `fit` directly; `_fit` is not used."
        )

    def get_regime(self, variable: str) -> str:
        """Return the detected regime label for a fitted variable."""
        if variable not in self._regimes:
            raise KeyError(
                f"Variable {variable!r} not fitted; call fit() first."
            )
        return self._regimes[variable]

    def fit(
        self,
        X_train: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        weight_col: Optional[Union[str, np.ndarray, pd.Series]] = None,
        skip_missing: bool = False,
        not_numeric_categorical: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Fit the regime-aware wrapper.

        Delegates non-numeric targets (categorical / boolean /
        constant) to a single base imputer instance. Numeric targets
        are handled per-variable: regime detection, then composition
        of gate + base imputer(s) as appropriate.

        Returns a ``ZeroInflatedImputerResults`` that routes
        predictions through each target's regime-specific pipeline.
        """
        self._validate_data(X_train, predictors + imputed_variables)

        # Classify target variables as numeric / categorical / boolean /
        # constant using the base Imputer's detector.
        self.identify_target_types(
            X_train,
            imputed_variables,
            not_numeric_categorical=not_numeric_categorical,
        )

        self.predictors = list(predictors)
        self.imputed_variables = list(imputed_variables)
        self._regimes = {}
        self._per_variable = {}

        # Per-variable fit for numeric targets. Constant numeric
        # targets (e.g. a column that is always 0 in training) are
        # also treated here so their regime lands in the wrapper's
        # ``_regimes`` map rather than being silently passed through.
        constant_numeric_targets = [
            v
            for v in imputed_variables
            if v in self.constant_targets
            and np.issubdtype(
                pd.Series(
                    [self.constant_targets[v]["value"]]
                ).dtype,
                np.number,
            )
        ]
        numeric_targets = [
            v
            for v in imputed_variables
            if v in self.numeric_targets or v in constant_numeric_targets
        ]
        for var in numeric_targets:
            y = X_train[var].to_numpy(dtype=float, copy=False)
            regime = _detect_regime(
                y,
                min_class_count=self.min_class_count,
                min_class_fraction=self.min_class_fraction,
                zero_atol=self.zero_atol,
            )
            self._regimes[var] = regime
            self._per_variable[var] = self._fit_single_numeric(
                X_train=X_train,
                predictors=predictors,
                variable=var,
                regime=regime,
                y=y,
            )

        # Non-numeric (categorical / boolean / constant) targets are
        # handled by a single auxiliary base imputer over their union.
        non_numeric = [v for v in imputed_variables if v not in numeric_targets]
        if non_numeric:
            aux = self.base_imputer_class(
                log_level="ERROR",
                **self.base_imputer_kwargs,
            )
            aux_result = aux.fit(
                X_train=X_train,
                predictors=predictors,
                imputed_variables=non_numeric,
                weight_col=weight_col,
                skip_missing=skip_missing,
                not_numeric_categorical=not_numeric_categorical,
                **kwargs,
            )
            aux_bundle = {"kind": "passthrough", "result": aux_result}
        else:
            aux_bundle = None

        return ZeroInflatedImputerResults(
            predictors=self.predictors,
            imputed_variables=self.imputed_variables,
            seed=self.seed,
            regimes=self._regimes,
            per_variable=self._per_variable,
            non_numeric_bundle=aux_bundle,
            log_level="WARNING",
        )

    # ------------------------------------------------------------------
    # Per-variable fit helpers
    # ------------------------------------------------------------------

    def _fit_single_numeric(
        self,
        *,
        X_train: pd.DataFrame,
        predictors: List[str],
        variable: str,
        regime: str,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """Fit the gate and base imputer(s) for one numeric target.

        Returns a bundle dict with the regime, the gate classifier
        (or None), and the base imputer(s) keyed by their role.
        """
        X_pred = X_train[predictors].to_numpy(dtype=float, copy=False)

        if regime == REGIME_DEGENERATE_ZERO:
            return {"kind": "constant", "value": 0.0}

        if regime in (REGIME_POSITIVE_ONLY, REGIME_NEGATIVE_ONLY):
            # No gate; single base imputer on the full training set.
            return {
                "kind": "single",
                "base": self._fit_base_single(X_train, predictors, variable),
            }

        if regime == REGIME_ZI_POSITIVE:
            labels = (y > self.zero_atol).astype(int)
            clf = _make_classifier(self.classifier_type, self.seed)
            clf.fit(X_pred, labels)
            pos_mask = y > self.zero_atol
            pos_base = self._fit_base_single(
                X_train.loc[pos_mask], predictors, variable
            )
            return {
                "kind": "zi_positive",
                "classifier": clf,
                "positive_base": pos_base,
            }

        if regime == REGIME_ZI_NEGATIVE:
            labels = (y < -self.zero_atol).astype(int)
            clf = _make_classifier(self.classifier_type, self.seed)
            clf.fit(X_pred, labels)
            neg_mask = y < -self.zero_atol
            neg_base = self._fit_base_single(
                X_train.loc[neg_mask], predictors, variable
            )
            return {
                "kind": "zi_negative",
                "classifier": clf,
                "negative_base": neg_base,
            }

        if regime == REGIME_SIGN_ONLY:
            # No zero class, but both signs present. Binary sign gate
            # plus a base imputer per sign.
            labels = (y > 0).astype(int)
            clf = _make_classifier(self.classifier_type, self.seed)
            clf.fit(X_pred, labels)
            pos_mask = y > 0
            neg_mask = ~pos_mask
            return {
                "kind": "sign_only",
                "classifier": clf,
                "positive_base": self._fit_base_single(
                    X_train.loc[pos_mask], predictors, variable
                ),
                "negative_base": self._fit_base_single(
                    X_train.loc[neg_mask], predictors, variable
                ),
            }

        if regime == REGIME_THREE_SIGN:
            # 0 / neg / pos three-way gate + two base imputers.
            labels = np.where(
                y > self.zero_atol,
                2,
                np.where(y < -self.zero_atol, 0, 1),
            )
            clf = _make_classifier(self.classifier_type, self.seed)
            clf.fit(X_pred, labels)
            pos_mask = y > self.zero_atol
            neg_mask = y < -self.zero_atol
            return {
                "kind": "three_sign",
                "classifier": clf,
                "positive_base": self._fit_base_single(
                    X_train.loc[pos_mask], predictors, variable
                ),
                "negative_base": self._fit_base_single(
                    X_train.loc[neg_mask], predictors, variable
                ),
            }

        raise ValueError(f"Unhandled regime {regime!r}")

    def _fit_base_single(
        self,
        X_train: pd.DataFrame,
        predictors: List[str],
        variable: str,
    ) -> ImputerResults:
        """Fit a single base Imputer on a (possibly filtered) slice."""
        imputer = self.base_imputer_class(
            log_level="ERROR",
            **self.base_imputer_kwargs,
        )
        return imputer.fit(
            X_train=X_train,
            predictors=predictors,
            imputed_variables=[variable],
        )


class ZeroInflatedImputerResults(ImputerResults):
    """Fitted regime-aware imputer ready for prediction."""

    def __init__(
        self,
        predictors: List[str],
        imputed_variables: List[str],
        seed: int,
        regimes: Dict[str, str],
        per_variable: Dict[str, Dict[str, Any]],
        non_numeric_bundle: Optional[Dict[str, Any]] = None,
        imputed_vars_dummy_info: Optional[Dict[str, Any]] = None,
        original_predictors: Optional[List[str]] = None,
        log_level: Optional[str] = "WARNING",
    ) -> None:
        super().__init__(
            predictors=predictors,
            imputed_variables=imputed_variables,
            seed=seed,
            imputed_vars_dummy_info=imputed_vars_dummy_info,
            original_predictors=original_predictors or predictors,
            log_level=log_level,
        )
        self._regimes = regimes
        self._per_variable = per_variable
        self._non_numeric_bundle = non_numeric_bundle
        self._rng = np.random.default_rng(seed)

    @validate_call(config=VALIDATE_CONFIG)
    def predict(
        self,
        X_test: pd.DataFrame,
        quantiles: Optional[List[float]] = None,
        return_probs: bool = False,
        **kwargs: Any,
    ) -> Union[pd.DataFrame, Dict[float, pd.DataFrame]]:
        """Predict imputed values, routing per-variable by regime.

        For numeric targets, the gate assigns each record to zero,
        positive, or negative regime (depending on the detected
        regime), and the base imputer for that regime produces the
        nonzero draw. Zeros are set exactly to 0.0 (no stochastic
        smearing).

        For non-numeric targets (categorical / boolean / constant),
        delegation is to the single auxiliary base imputer fit at
        training time.
        """
        if quantiles is not None:
            # Quantile grid not currently supported in the wrapper; the
            # regime routing only produces a single stochastic draw per
            # call. Deterministic-quantile support would require the
            # caller to specify quantile conditional on regime.
            return {
                q: self._predict_single_draw(X_test, quantile=q, **kwargs)
                for q in quantiles
            }
        return self._predict_single_draw(X_test, quantile=None, **kwargs)

    def _predict_single_draw(
        self,
        X_test: pd.DataFrame,
        quantile: Optional[float],
        **kwargs: Any,
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=X_test.index)

        for variable in self.imputed_variables:
            regime = self._regimes.get(variable)
            if regime is None:
                # Non-numeric target; handled by the auxiliary bundle.
                continue
            bundle = self._per_variable[variable]
            out[variable] = self._predict_single_variable(
                X_test, variable, bundle, quantile=quantile, **kwargs
            )

        # Merge in non-numeric target predictions from the auxiliary
        # single base imputer.
        if self._non_numeric_bundle is not None:
            aux_result = self._non_numeric_bundle["result"]
            if quantile is None:
                aux_preds = aux_result.predict(X_test)
            else:
                aux_dict = aux_result.predict(X_test, quantiles=[quantile])
                aux_preds = aux_dict[quantile]
            for col in aux_preds.columns:
                if col not in out.columns:
                    out[col] = aux_preds[col].values

        return out

    def _predict_single_variable(
        self,
        X_test: pd.DataFrame,
        variable: str,
        bundle: Dict[str, Any],
        quantile: Optional[float],
        **kwargs: Any,
    ) -> np.ndarray:
        n = len(X_test)
        kind = bundle["kind"]

        if kind == "constant":
            return np.full(n, bundle["value"], dtype=float)

        if kind == "single":
            preds = self._invoke_base(
                bundle["base"], X_test, quantile=quantile, **kwargs
            )
            return preds[variable].to_numpy(dtype=float)

        X_pred = X_test[self.predictors].to_numpy(dtype=float, copy=False)

        if kind == "zi_positive":
            clf = bundle["classifier"]
            draw = self._bernoulli_gate_draw(clf, X_pred)
            values = np.zeros(n, dtype=float)
            positive_mask = draw == 1
            if positive_mask.any():
                sub_preds = self._invoke_base(
                    bundle["positive_base"],
                    X_test.loc[positive_mask],
                    quantile=quantile,
                    **kwargs,
                )
                values[positive_mask] = sub_preds[variable].to_numpy(dtype=float)
            return values

        if kind == "zi_negative":
            clf = bundle["classifier"]
            draw = self._bernoulli_gate_draw(clf, X_pred)
            values = np.zeros(n, dtype=float)
            negative_mask = draw == 1
            if negative_mask.any():
                sub_preds = self._invoke_base(
                    bundle["negative_base"],
                    X_test.loc[negative_mask],
                    quantile=quantile,
                    **kwargs,
                )
                values[negative_mask] = sub_preds[variable].to_numpy(dtype=float)
            return values

        if kind == "sign_only":
            clf = bundle["classifier"]
            draw = self._bernoulli_gate_draw(clf, X_pred)
            positive_mask = draw == 1
            negative_mask = ~positive_mask
            values = np.zeros(n, dtype=float)
            if positive_mask.any():
                sub_preds = self._invoke_base(
                    bundle["positive_base"],
                    X_test.loc[positive_mask],
                    quantile=quantile,
                    **kwargs,
                )
                values[positive_mask] = sub_preds[variable].to_numpy(dtype=float)
            if negative_mask.any():
                sub_preds = self._invoke_base(
                    bundle["negative_base"],
                    X_test.loc[negative_mask],
                    quantile=quantile,
                    **kwargs,
                )
                values[negative_mask] = sub_preds[variable].to_numpy(dtype=float)
            return values

        if kind == "three_sign":
            clf = bundle["classifier"]
            probas = clf.predict_proba(X_pred)
            # Classes are [0=neg, 1=zero, 2=pos] per the fit encoding.
            cumulative = np.cumsum(probas, axis=1)
            u = self._rng.random(n)
            # Each row i is assigned to class argmax over k of (cumulative[i,k] >= u[i]).
            class_indices = (cumulative >= u[:, None]).argmax(axis=1)
            classes = clf.classes_[class_indices]
            values = np.zeros(n, dtype=float)
            positive_mask = classes == 2
            negative_mask = classes == 0
            if positive_mask.any():
                sub_preds = self._invoke_base(
                    bundle["positive_base"],
                    X_test.loc[positive_mask],
                    quantile=quantile,
                    **kwargs,
                )
                values[positive_mask] = sub_preds[variable].to_numpy(dtype=float)
            if negative_mask.any():
                sub_preds = self._invoke_base(
                    bundle["negative_base"],
                    X_test.loc[negative_mask],
                    quantile=quantile,
                    **kwargs,
                )
                values[negative_mask] = sub_preds[variable].to_numpy(dtype=float)
            return values

        raise ValueError(f"Unhandled bundle kind {kind!r}")

    def _invoke_base(
        self,
        base_result: ImputerResults,
        X_slice: pd.DataFrame,
        quantile: Optional[float],
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Call a base ImputerResults, returning a DataFrame."""
        if quantile is None:
            result = base_result.predict(X_slice, **kwargs)
            if isinstance(result, dict):
                # Some base imputers always return a dict even without
                # ``quantiles``; pick the first.
                result = next(iter(result.values()))
            return result
        result = base_result.predict(X_slice, quantiles=[quantile], **kwargs)
        if isinstance(result, dict):
            return result[quantile]
        return result

    def _bernoulli_gate_draw(
        self,
        classifier: Any,
        X_pred: np.ndarray,
    ) -> np.ndarray:
        """Stochastic draw from the binary classifier's predicted proba.

        Returns an array of 0/1 integers (length ``len(X_pred)``),
        matching classifier.classes_ encoding for class-1.
        """
        probas = classifier.predict_proba(X_pred)
        # Ensure we pull the probability for the "positive-class" index
        # (which is whichever class the classifier labeled 1 at fit time).
        classes = np.asarray(classifier.classes_)
        if 1 in classes:
            positive_idx = int(np.where(classes == 1)[0][0])
        else:
            positive_idx = probas.shape[1] - 1
        positive_prob = probas[:, positive_idx]
        u = self._rng.random(len(X_pred))
        return (u < positive_prob).astype(int)

    def _predict(self, *args: Any, **kwargs: Any) -> Any:
        """Abstract-method placeholder.

        ``ImputerResults._predict`` is abstract; this class overrides
        ``predict`` directly and never dispatches through ``_predict``,
        but the abstract method still must be satisfied.
        """
        raise NotImplementedError(
            "ZeroInflatedImputerResults overrides `predict` directly; "
            "`_predict` is not used."
        )


__all__ = [
    "REGIME_DEGENERATE_ZERO",
    "REGIME_NEGATIVE_ONLY",
    "REGIME_POSITIVE_ONLY",
    "REGIME_SIGN_ONLY",
    "REGIME_THREE_SIGN",
    "REGIME_ZI_NEGATIVE",
    "REGIME_ZI_POSITIVE",
    "ZeroInflatedImputer",
    "ZeroInflatedImputerResults",
]
