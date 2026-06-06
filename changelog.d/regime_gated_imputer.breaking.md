**Renamed `ZeroInflatedImputer` to the canonical `microimpute.Imputer`** and made it the opinionated default. The previous abstract base class `Imputer` is now `BaseImputer` (still exported). `microimpute.Imputer` is the regime-gated, QRF-based, sequentially-chained imputer:

- **Sign-regime gating** (`{neg, 0, pos}`) on by default (`signregime=True`); pass `signregime=False` to impute with the base model directly (no gate).
- **QRF base model** by default (`base_imputer_class=QRF`); swap for experiments.
- **Sequential chained-equations imputation is always on** — imputing a list of targets conditions each on the previously-imputed ones, preserving cross-variable joint structure. The old per-variable-independent path and its `sequential` flag are removed.
- The fitted result exposes `.lineage()` → per-variable `VariableLineage`: regime, the chained predictor set, training-support counts, the fitted models, and feature importances where the underlying model provides them.

Migration: replace `from microimpute.models.zero_inflated import ZeroInflatedImputer` with `from microimpute import Imputer`; references to the old base class `Imputer` become `BaseImputer`.
