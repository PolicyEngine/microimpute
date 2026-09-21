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

Regime detection is based only on observed support. If the training data
contains negative, zero, and positive values, the imputer uses the
three-sign architecture. Callers do not provide sign/regime metadata.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd
from pydantic import validate_call

from microimpute.config import RANDOM_STATE, VALIDATE_CONFIG
from microimpute.models.imputer import Imputer, ImputerResults
from microimpute.models.qrf import QRF
from microimpute.utils.type_handling import declare_target_types

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

        return RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1)
    raise ValueError(f"Unknown classifier_type {kind!r}; expected 'hist_gb' or 'rf'.")


def _detect_regime(
    y: np.ndarray,
    *,
    zero_atol: float,
) -> str:
    """Classify the training distribution into one of seven regimes.

    A class (neg/zero/pos) counts as present when it appears at least
    once in the training data. Sign support is inferred from donor data;
    callers cannot force a variable to be positive-only, negative-only,
    or signed.
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

    has_zero = n_zero > 0
    has_pos = n_pos > 0
    has_neg = n_neg > 0

    if not (has_zero or has_pos or has_neg):
        return REGIME_DEGENERATE_ZERO

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
        zero_atol: float = 1e-6,
        classifier_type: str = "hist_gb",
        sequential: bool = True,
        seed: Optional[int] = RANDOM_STATE,
        log_level: Optional[str] = "WARNING",
    ) -> None:
        super().__init__(seed=seed, log_level=log_level)
        self.base_imputer_class = base_imputer_class or QRF
        self.base_imputer_kwargs = dict(base_imputer_kwargs or {})
        self.zero_atol = float(zero_atol)
        if not np.isfinite(self.zero_atol) or self.zero_atol < 0:
            raise ValueError("zero_atol must be finite and nonnegative")
        self.classifier_type = classifier_type
        self.sequential = bool(sequential)

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
            raise KeyError(f"Variable {variable!r} not fitted; call fit() first.")
        return self._regimes[variable]

    def fit(
        self,
        X_train: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        weight_col: Optional[Union[str, np.ndarray, pd.Series]] = None,
        skip_missing: bool = False,
        not_numeric_categorical: Optional[List[str]] = None,
        target_types: Optional[Dict[str, str]] = None,
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
        X_train = declare_target_types(X_train, imputed_variables, target_types)
        not_numeric_categorical = list(not_numeric_categorical or [])
        not_numeric_categorical.extend(
            name for name, kind in (target_types or {}).items() if kind == "numeric"
        )
        if skip_missing:
            imputed_variables = self._handle_missing_variables(
                X_train, imputed_variables
            )
        self._validate_data(X_train, predictors + imputed_variables)
        if set(predictors) & set(imputed_variables):
            raise ValueError("Predictors and imputed variables must be distinct")
        sample_weight = None
        if isinstance(weight_col, str):
            if weight_col not in X_train:
                raise ValueError(f"Weight column {weight_col!r} not found")
            sample_weight = X_train[weight_col].to_numpy(dtype=float)
        elif isinstance(weight_col, pd.Series):
            sample_weight = weight_col.reindex(X_train.index).to_numpy(dtype=float)
        elif weight_col is not None:
            sample_weight = np.asarray(weight_col, dtype=float)
        if sample_weight is not None:
            if sample_weight.shape != (len(X_train),):
                raise ValueError("Weights must have one value per training row")
            if not np.isfinite(sample_weight).all() or (sample_weight <= 0).any():
                raise ValueError("Weights must be positive and finite")

        self.categorical_targets = {}
        self.boolean_targets = {}
        self.numeric_targets = []
        self.constant_targets = {}

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
                pd.Series([self.constant_targets[v]["value"]]).dtype,
                np.number,
            )
        ]
        numeric_targets = [
            v
            for v in imputed_variables
            if v in self.numeric_targets or v in constant_numeric_targets
        ]
        nested_not_numeric_categorical = list(
            dict.fromkeys([*(not_numeric_categorical or []), *numeric_targets])
        )
        # Sequential (chained-equations) imputation: condition each numeric
        # target on the original predictors plus the previously-fit numeric
        # targets, so the imputed vector preserves cross-variable joint
        # structure. ``imputed_variables`` order is the chain order; a
        # single-target list is unaffected (no priors to chain on).
        for position, var in enumerate(numeric_targets):
            seq_predictors = (
                list(predictors) + numeric_targets[:position]
                if self.sequential
                else list(predictors)
            )
            y = X_train[var].to_numpy(dtype=float, copy=False)
            regime = _detect_regime(
                y,
                zero_atol=self.zero_atol,
            )
            self._regimes[var] = regime
            bundle = self._fit_single_numeric(
                X_train=X_train,
                predictors=seq_predictors,
                variable=var,
                regime=regime,
                y=y,
                not_numeric_categorical=nested_not_numeric_categorical,
                sample_weight=sample_weight,
                fit_kwargs=kwargs,
            )
            bundle["predictors"] = list(seq_predictors)
            self._per_variable[var] = bundle

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
        not_numeric_categorical: Optional[List[str]] = None,
        sample_weight: Optional[np.ndarray] = None,
        fit_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fit weighted regime probabilities and weighted sign components."""
        if not np.isfinite(y).all():
            raise ValueError(f"Numeric target {variable!r} must contain finite values")
        X_pred = X_train[predictors].to_numpy(dtype=float, copy=False)
        if not np.isfinite(X_pred).all():
            raise ValueError("Zero-inflated predictors must contain finite values")
        if regime == REGIME_DEGENERATE_ZERO:
            return {"kind": "constant", "value": 0.0}

        def fit_component(mask: np.ndarray, offset: int) -> ImputerResults:
            weights = sample_weight[mask] if sample_weight is not None else None
            seed = (
                None
                if self.seed is None
                else (
                    int(self.seed) + 3 * self.imputed_variables.index(variable) + offset
                )
                % (2**32)
            )
            return self._fit_base_single(
                X_train.loc[mask],
                predictors,
                variable,
                not_numeric_categorical=not_numeric_categorical,
                sample_weight=weights,
                seed=seed,
                fit_kwargs=fit_kwargs,
            )

        if regime in (REGIME_POSITIVE_ONLY, REGIME_NEGATIVE_ONLY):
            return {
                "kind": "single",
                "base": fit_component(np.ones(len(y), dtype=bool), 0),
            }

        positive = y > self.zero_atol
        negative = y < -self.zero_atol
        if regime == REGIME_THREE_SIGN:
            labels = np.where(positive, 2, np.where(negative, 0, 1))
            kind = "three_sign"
        elif regime == REGIME_SIGN_ONLY:
            labels, kind = positive.astype(int), "sign_only"
        elif regime == REGIME_ZI_POSITIVE:
            labels, kind = positive.astype(int), "zi_positive"
        elif regime == REGIME_ZI_NEGATIVE:
            labels, kind = negative.astype(int), "zi_negative"
        else:
            raise ValueError(f"Unhandled regime {regime!r}")
        classifier = _make_classifier(self.classifier_type, self.seed)
        classifier.fit(X_pred, labels, sample_weight=sample_weight)
        bundle = {"kind": kind, "classifier": classifier}
        if positive.any():
            bundle["positive_base"] = fit_component(positive, 1)
        if negative.any():
            bundle["negative_base"] = fit_component(negative, 2)
        return bundle

    def _fit_base_single(
        self,
        X_train: pd.DataFrame,
        predictors: List[str],
        variable: str,
        not_numeric_categorical: Optional[List[str]] = None,
        sample_weight: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        fit_kwargs: Optional[Dict[str, Any]] = None,
    ) -> ImputerResults:
        """Fit a component with its own seed and aligned conditional weights."""
        constructor_kwargs = {
            "log_level": "ERROR",
            "seed": seed,
            **self.base_imputer_kwargs,
        }
        imputer = self.base_imputer_class(**constructor_kwargs)
        result = imputer.fit(
            X_train=X_train,
            predictors=predictors,
            imputed_variables=[variable],
            weight_col=sample_weight,
            not_numeric_categorical=not_numeric_categorical,
            **(fit_kwargs or {}),
        )
        # The fit API returns (result, params) when tuning is requested.
        return result[0] if isinstance(result, tuple) else result


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

        Explicit quantiles invert the fitted signed mixture CDF, including
        its zero atom, deterministically. Without quantiles, the gate samples
        a regime and its component produces a stochastic draw. Quantiles
        across multiple sequential targets are unsupported because chaining
        conditional quantiles does not produce marginal quantiles.

        For non-numeric targets (categorical / boolean / constant),
        delegation is to the single auxiliary base imputer fit at
        training time.
        """
        self._validate_quantiles(quantiles)
        if return_probs:
            raise NotImplementedError(
                "ZeroInflatedImputer does not expose categorical probabilities"
            )
        if quantiles is not None:
            if not quantiles:
                raise ValueError("quantiles must not be empty")
            if any(
                set(bundle.get("predictors", [])) & set(self._regimes)
                for bundle in self._per_variable.values()
            ):
                raise NotImplementedError(
                    "Marginal quantiles for sequential multi-target imputation "
                    "are not available; fit with sequential=False or a single target"
                )
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

        # Carry imputed numeric targets forward as predictors for later
        # targets (chained-equations imputation), matching the sequential
        # conditioning used at fit time. ``X_aug`` accumulates imputed
        # columns in ``imputed_variables`` order so each target's gate/base
        # can condition on the ones already drawn.
        X_aug = X_test.copy()
        for variable in self.imputed_variables:
            regime = self._regimes.get(variable)
            if regime is None:
                # Non-numeric target; handled by the auxiliary bundle.
                continue
            bundle = self._per_variable[variable]
            values = self._predict_single_variable(
                X_aug, variable, bundle, quantile=quantile, **kwargs
            )
            out[variable] = values
            X_aug[variable] = np.asarray(values, dtype=float)

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

        X_pred = X_test[bundle.get("predictors", self.predictors)].to_numpy(
            dtype=float, copy=False
        )

        if quantile is not None:
            return self._mixture_quantile(
                X_test, X_pred, variable, bundle, quantile, **kwargs
            )

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

    def _mixture_quantile(
        self,
        X_test: pd.DataFrame,
        X_pred: np.ndarray,
        variable: str,
        bundle: Dict[str, Any],
        quantile: float,
        **kwargs: Any,
    ) -> np.ndarray:
        """Invert the ordered negative / zero / positive mixture CDF.

        Negative quantiles use q / p_neg; the zero atom occupies
        (p_neg, p_neg + p_zero]; positive quantiles use
        (q - p_neg - p_zero) / p_pos. Endpoints select the first/last
        nonempty component, including rows with degenerate gate probabilities.
        """
        classifier = bundle["classifier"]
        probabilities = classifier.predict_proba(X_pred)
        by_class = {
            label: probabilities[:, i] for i, label in enumerate(classifier.classes_)
        }
        zero = np.zeros(len(X_test))
        if bundle["kind"] == "three_sign":
            p_neg, p_zero, p_pos = (by_class.get(k, zero) for k in (0, 1, 2))
        elif bundle["kind"] == "sign_only":
            p_neg, p_zero, p_pos = by_class.get(0, zero), zero, by_class.get(1, zero)
        elif bundle["kind"] == "zi_positive":
            p_neg, p_zero, p_pos = zero, by_class.get(0, zero), by_class.get(1, zero)
        else:
            p_neg, p_zero, p_pos = by_class.get(1, zero), by_class.get(0, zero), zero
        values = np.zeros(len(X_test))
        negative = (p_neg > 0) & (quantile <= p_neg)
        if quantile == 1:
            negative &= (p_zero == 0) & (p_pos == 0)
        positive = (p_pos > 0) & (
            (quantile == 1)
            | (quantile > p_neg + p_zero)
            | ((quantile == 0) & (p_neg + p_zero == 0))
        )
        for sign, mask, offset, mass in (
            ("negative", negative, zero, p_neg),
            ("positive", positive, p_neg + p_zero, p_pos),
        ):
            if not mask.any():
                continue
            conditional_q = np.clip((quantile - offset[mask]) / mass[mask], 0, 1)
            if quantile == 1:
                conditional_q[:] = 1
            result = bundle[f"{sign}_base"]
            subset = X_test.loc[mask]
            if hasattr(result, "_predict_quantiles_per_row") and not kwargs:
                component = result._predict_quantiles_per_row(
                    subset, variable, conditional_q
                )
            else:
                component = np.empty(mask.sum())
                for q in np.unique(conditional_q):
                    selected = conditional_q == q
                    predictions = self._invoke_base(
                        result, subset.loc[selected], quantile=float(q), **kwargs
                    )
                    component[selected] = predictions[variable].to_numpy(dtype=float)
            if not np.isfinite(component).all():
                raise ValueError("Component quantiles must be finite")
            invalid = component >= 0 if sign == "negative" else component <= 0
            if invalid.any():
                raise ValueError(
                    f"The {sign} component predicts outside its sign support; "
                    "use a base imputer that preserves the component support"
                )
            values[mask] = component
        return values

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
