**Renamed `ZeroInflatedImputer` to the canonical `microimpute.Imputer`** and made it the opinionated default. The previous abstract base class `Imputer` is now `BaseImputer` (still exported). `microimpute.Imputer` is the regime-gated, QRF-based, sequentially-chained imputer:

- **Sign-regime gating** (`{neg, 0, pos}`) on by default (`signregime=True`); pass `signregime=False` to impute with the base model directly (no gate).
- **QRF base model** by default (`base_imputer_class=QRF`); swap for experiments.
- **Sequential chained-equations imputation is always on** — imputing a list of targets conditions each on the previously-imputed ones, preserving cross-variable joint structure. The old per-variable-independent path and its `sequential` flag are removed.
- The fitted result exposes fitted state sklearn-style — `regimes_`, `predictors_`, `models_` (sub-estimators by role), and the sub-estimators carry standard `feature_importances_`/`feature_names_in_`; full per-variable lineage is assembled by the caller (e.g. microplex), not microimpute.

Migration: replace `from microimpute.models.zero_inflated import ZeroInflatedImputer` with `from microimpute import Imputer`; references to the old base class `Imputer` become `BaseImputer`.
