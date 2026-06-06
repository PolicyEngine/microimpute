**Renamed `ZeroInflatedImputer` to the canonical `microimpute.Imputer`** and made it the opinionated default. The previous abstract base class `Imputer` is now `BaseImputer` (still exported). `microimpute.Imputer` is the regime-gated, QRF-based, sequentially-chained imputer:

- **Sign-regime gating** (`{neg, 0, pos}`) on by default (`signregime=True`); pass `signregime=False` to impute each numeric target with the base model directly (no gate, the `REGIME_NO_GATE` path).
- **QRF base model** by default (`base_imputer_class=QRF`); swap for experiments.
- **Sequential chained-equations imputation is always on** — imputing a list of targets conditions each on the previously-imputed ones, preserving cross-variable joint structure. The old per-variable-independent path and its `sequential` flag are removed.
- The fitted result exposes fitted state sklearn-style — `regimes_`, `predictors_` (the chained predictor list per target), and `models_` (sub-estimators by role: single/gate/positive/negative). QRF base sub-estimators carry standard `feature_importances_`/`feature_names_in_`: `feature_importances_` is a `{fitted_feature: importance}` dict keyed by the forest's actual fitted columns (so names and values always align, even when a categorical predictor expands into dummy columns), and `feature_names_in_` reports the original input predictor names.

Migration: replace `from microimpute.models.zero_inflated import ZeroInflatedImputer` with `from microimpute import Imputer`; references to the old base class `Imputer` become `BaseImputer`.
