## [2.0.4] - 2026-04-28

### Changed

- Updated dashboard fonts from Roboto Serif to Inter/JetBrains Mono, added data preview on file upload, added reusable ChartLegend component, and improved chart styling across all visualization components. Updated paper formatting to Times New Roman 11pt with 1.5in margins, added Acknowledgements and Disclosures section with keywords, and refreshed figures and results with latest cross-validation run. Migrated versioning workflow from expired PAT to GitHub App token. Update versioning path to include CHANGELOG.

### Fixed

- Fixed towncrier fragment validation and package versioning workflow triggers, and dashboard CSV compatibility with `metric_std` columns, benchmark-loss error bars, and donor/receiver distribution rows.


## [2.0.3] - 2026-04-17

### Fixed

- Fixed `autoimpute` mutating the caller's `receiver_data` (#13). The final `receiver_data[var] = median_imputations[var]` assignment could write through to the caller's original DataFrame depending on whether intermediate pandas operations returned a copy or a view — a subtle side effect that silently added imputed columns to the user's input frame. `autoimpute` now takes a defensive `.copy()` of `receiver_data` at the top of the function so the caller's frame is always preserved, and the imputed columns are returned exclusively through `result.receiver_data`.
- Fixed `Imputer.fit(weight_col=...)` silently discarding weights (#4). Previously, weights were used only as bootstrap-resample probabilities over `X_train`, with the resampled data then fed unweighted into the underlying estimator; effective sample size shrank, rare donors were dropped, and variance was inflated relative to the correct weighted estimator. The base `Imputer.fit` now threads `sample_weight` through to each learner's native weighted-fit API: `RandomForestQuantileRegressor.fit(sample_weight=...)`, `sm.WLS` (instead of `sm.OLS`), `LogisticRegression.fit(sample_weight=...)`, `RandomForestClassifier.fit(sample_weight=...)`, and StatMatch's `NND.hotdeck` via `weight.don`. Models that do not support weighted fit (`QuantReg`, `MDN`) now raise `NotImplementedError` rather than silently ignoring weights. NaN weights are now rejected explicitly (previously `(weights <= 0).any()` returned `False` on NaN and let the NaN propagate into `.sample()` probabilities).
- Fixed asymmetric epsilon handling in `kl_divergence` (#12). The previous implementation clipped only `q_receiver` by `epsilon=1e-10`, so a category present in `p` but absent in `q` contributed `p * log(p / eps)` (a large finite value arbitrarily depending on `epsilon`) while the reverse contributed `rel_entr(0, q) = 0` (a free pass). Both distributions now receive the same epsilon floor and are renormalised back to probability vectors, so KL behaves consistently regardless of which side has the missing category.
- Fixed collision-prone MDN cache key (#5). `_generate_data_hash` previously used `pd.util.hash_pandas_object(X).sum()` which loses row ordering (any permutation hashes identically) and makes cross-dataset collisions trivial, so a different dataset with matching `(shape, columns, sum-of-hashes)` could silently load a stale MDN from disk (correctness bug). The function now uses `hashlib.sha256` over the raw bytes of per-row hashes, producing an order-sensitive content digest.
- Fixed three OLS imputation bugs: (#6) `_OLSModel.predict` previously used `se = sqrt(model.scale)` (residual std only) to scale normal quantiles, which under-dispersed imputations for test rows far from the training centroid; now uses `statsmodels`' `model.get_prediction(X).var_pred_mean + model.scale` to include the leverage term, producing a per-row prediction SE. The quantile is also clipped to `(1e-6, 1-1e-6)` so `q=0` / `q=1` no longer produce ±inf from `norm.ppf`. (#8) `_LogisticRegressionModel.fit` previously passed `l1_ratio` to `LogisticRegression` with the default `penalty="l2"`, which silently ignored the parameter; now sets `penalty="elasticnet"` and `solver="saga"` when a non-zero `l1_ratio` is supplied so the parameter actually takes effect. (followup) `OLSResults._predict_quantile` now returns a `pd.Series` indexed to `mean_preds.index` rather than a bare `ndarray`. When a mixed-type imputation built the output `DataFrame` by assigning a numeric column (ndarray) before a categorical column (indexed Series), pandas anchored the DataFrame to a default `RangeIndex` and the subsequent categorical assignment failed to align — silently producing all-NaN categorical columns, which later broke downstream `sklearn.log_loss`.
- Fixed three correctness bugs in `_QRFModel.predict`: (1) the random-number generator was re-initialised from `self.seed` on every `predict()` call, so repeated calls returned identical draws and multiple-imputation variance collapsed to zero — the RNG is now created once in `__init__` and advanced across calls; (2) the quantile grid `[0.091..0.909]` combined with `.astype(int)` truncation systematically biased stochastic median predictions low and truncated the tails — the implementation now rounds (rather than floors) a beta-distribution draw onto a fine symmetric grid so the empirical mean maps to the intended quantile; (3) when users passed explicit `quantiles=[q1, q2, q3]`, each quantile request drew its own per-row random index, producing crossed quantiles — `QRFResults._predict` now routes explicit quantiles through a deterministic `exact_quantile` path that guarantees row-level monotonicity.
- Removed dead `random_quantile` RNG code from `QuantReg._fit` (#11) — the RNG was initialised and never used while `q` was hardcoded to 0.5, and the log message "Fitting quantile regression for random quantile" was misleading. Also vectorised the per-row random-quantile path in `QuantRegResults._predict`: the previous implementation allocated an object-dtype DataFrame and wrote numeric predictions into it via `result_df.loc[idx, variable] = ...` inside a Python loop (quadratic in rows x variables, silently demoting numeric output to `object`, contributing to OOM pressure in issue #96). The new implementation uses `np.column_stack` + per-row index to select predictions in one pass, returning a numeric dtype.
- Fixed two silent-corruption paths in `nnd_hotdeck_using_rpy2` (#7). Out-of-range donor indices were previously modulo-wrapped (`np.remainder(donor_indices - 1, len(donor)) + 1`), which masked real StatMatch/R indexing bugs by silently assigning an arbitrary donor to recipients. And when StatMatch returned fewer matches than recipients, the function repeated the last match for every missing recipient — producing severe homogeneity bias in the imputed column that was entirely invisible to callers. Both paths now raise `ValueError` with a descriptive message so the caller is aware of the malformed R result and can investigate (NaN predictors, dtype mismatch, empty `mtc.ids`, etc.) rather than silently producing biased imputations.
- Fixed two variable-type bugs in `microimpute.utils.type_handling`: (#9) `VariableTypeDetector.is_boolean_variable` no longer treats a float column containing only `{0.0, 1.0}` as boolean, which previously silently routed probability / rescaled-indicator columns to a classifier instead of the regressor. Only genuine `bool` dtypes and integer columns with values in `{0, 1}` are recognised as boolean now. (#10) `DummyVariableProcessor.apply_dummy_encoding_to_test` now emits a `UserWarning` when test data contains categorical levels not seen at training time (previously those rows silently received the reference-level prediction via all-zero dummies). The processor also now records the full training level set (including the dropped reference) so the warning doesn't fire on genuine reference-level rows.


## [2.0.2] - 2026-04-03

No significant changes.


## [2.0.1] - 2026-04-03

No significant changes.


## [1.15.1] - 2026-03-09

### Changed

- Added Python 3.14 support, bumped dependency upper bounds (pandas, plotly, scipy, pytest, pytest-cov), and upgraded GitHub Actions versions.


## [1.15.0] - 2026-03-09

### Added

- Added `max_train_samples` parameter and `fit_predict()` method to QRF, with automatic zero-filling of missing output variables.


## [1.14.5] - 2026-03-06

No significant changes.


## [1.14.4] - 2026-03-06

### Changed

- Switch from black to ruff for code formatting


## [1.14.3] - 2026-02-27

### Fixed

- Updated paper and package documentation with latest changes. Fix pandas 2.x compatibility for Arrow string types and dtype checks.


## [1.14.2] - 2026-02-24

### Changed

- Migrated from changelog_entry.yaml to towncrier fragments to eliminate merge conflicts.


# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), 
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.14.1] - 2026-02-22 13:05:13

### Changed

- imputing-from-scf-to-cps.ipynb to test SSI policy reforms with different wealth imputations.
- updated paper/ and main.pdf to capture the new results and discussion from the notebook as well as general improvements in content and presentation.

## [1.14.0] - 2026-02-07 04:51:52

### Added

- Error bars and grid lines to visualizations.
- Notebook benchmarking models on additional datasets.

## [1.13.0] - 2026-01-20 05:25:02

## [1.12.0] - 2025-12-11 14:03:49

### Added

- Benchmarking experiments for wealth imputation paper draft.
- MDN model to experiments run in imputing-from-scf-to-cps.ipynb.
- Privacy & Terms to microimputation-dashboard.

## [1.11.0] - 2025-12-05 09:20:27

### Added

- Updates to documentation and Myst deployment.

## [1.10.0] - 2025-12-04 11:22:49

### Added

- Asinh transformation preprocessing option for numeric variables.
- Documentation for MDN model and preprocessing options.

## [1.9.0] - 2025-12-03 10:54:44

### Added

- Mixture Density Network (MDN) model for numeric variable imputation and Neural Classifier model for categorical variable imputation.

## [1.8.1] - 2025-11-06 04:49:24

### Fixed

- Fixed pyproject.toml to avoid dependency versions updating with package again, again.

## [1.8.0] - 2025-11-06 04:39:56

### Added

- Distribution comparison histogram of donor and receiver datasets for imputed variables (to dashboard).
- Log transformation option for numerical variables in data preprocessing.

## [1.7.0] - 2025-10-24 16:06:59

### Added

- Links to dashboard in README.md and documentation.
- First dashboard visualizations.

## [1.6.1] - 2025-10-21 03:20:28

### Fixed

- Add suspense to fix vercel deployment issue.

## [1.6.0] - 2025-10-20 08:34:22

### Added

- Created microimputation-dashboard directory with initial dashboard components.
- File upload component to load microimputation results.
- Designed structure required for micorimputation_results.csv to load onto dashboard.
- Created `format_csv` function to format results for dashboard compatibility.

## [1.5.2] - 2025-10-19 10:08:29

### Fixed

- Fixed QRF to encode categorical imputed variables that become predictors correctly.
- Added `not_numeric_categorical` parameter to control whether discrete numeric variables are treated as categorical.
- Replaced Total Variation Distance with KL-divergence.

## [1.5.1] - 2025-10-10 08:33:23

### Fixed

- Fixed pyproject.toml to avoid dependency versions updating with package again.

## [1.5.0] - 2025-10-10 08:09:07

### Added

- Predictor correlation and sensitivity analysis tools.
- Updated documentation with distributional similarity metrics and predictor analysis.

## [1.4.1] - 2025-10-04 02:27:16

### Fixed

- Fixed pyproject.toml to avoid dependency versions updating with package.

## [1.4.0] - 2025-10-04 01:32:02

### Added

- Wasserstein distance and Total Variation Distance metrics for distributional similarity.

### Fixed

- Bug in data preprocessing which attempted to normalize categorical variables.
- Bug in loss metric used in Matching hyperparameter tuning.

## [1.3.0] - 2025-10-02 12:11:14

### Fixed

- QRF hyperparameter tuning now correctly tunes QRF and RFC separately.
- QRF imputation now correctly handles categorical variables that become predictors after imputation.

## [1.2.3] - 2025-09-25 12:33:38

### Added

- Log loss metric for evaluating categorical variable imputation.
- Functionality for cross-validation and autoimpute to integrate log loss.
- Visualization utilities for categorical imputation performance.

### Changed

- Updated documentation to reflect new methods and log loss features.

## [1.2.2] - 2025-09-18 12:24:10

### Added

- RandomForestClassifier and LogisticRegression models for categorical variable imputation.

## [1.2.1] - 2025-09-14 03:57:42

### Added

- Improved test coverage.

### Changed

- Refactor the Imputer base class.
- Refactor cross-validation, comparison, visualization, and autoimputation functions.

## [1.2.0] - 2025-09-07 08:07:54

### Added

- Support for Python 3.12 alongside Python 3.13
- Python 3.12 to CI/CD test matrix for comprehensive testing
- Graceful handling of optional Matching module when R dependencies are unavailable

### Changed

- Python version requirement from ">=3.13,<3.14" to ">=3.12,<3.14"
- Black formatter target versions to include both py312 and py313
- GitHub Actions workflows to test against both Python 3.12 and 3.13
- Python 3.12 CI tests to run minimal smoke test only (QRF basic functionality)

### Fixed

- Issue where predict() returns DataFrame instead of Dict for single quantile in autoimpute
- Import errors when Matching module is not available due to missing R dependencies
- Unconditional import of rpy2-dependent modules in utils package causing test failures

## [1.1.6] - 2025-08-08 08:17:09

### Added

- Make models return a dataframe directly when no quantiles are specified.

## [1.1.5] - 2025-08-05 13:58:25

### Added

- Moved data loading utilities into utils and removed scf downloading functionality.
- Updated documentation to reflect the new structure and created a myst.yml file to deploy documentation with new jb v2.

## [1.1.4] - 2025-08-01 16:36:35

### Changed

- Add condition to not convert numeric columns to categorical if they have less than 10 unique values that are not evenly spaced.

## [1.1.3] - 2025-08-01 14:31:06

### Added

- Updated categorical dummy encoding to restore original columns when dummies are not generated (edge case for when there is a single category).

## [1.1.2] - 2025-07-31 23:56:29

### Added

- Added memory usage logging to QRF.
- Enabled imputation even if there are imputed variables missing when skip_missing is True.

## [1.1.1] - 2025-07-31 11:22:52

### Added

- Extended test coverage for QRF model including edge cases and internal class testing

### Changed

- Removed utils.QRF wrapper to use RandomForestQuantileRegressor directly for consistency with OLS/QuantReg patterns
- Removed duplicate categorical handling from QRF model as base Imputer class already handles this

## [1.1.0] - 2025-07-31 10:14:08

### Added

- Documentation explaining sequential imputation behavior in QRF

### Changed

- Upgraded to JupyterBook 2.0 (beta) for improved documentation builds

## [1.0.2] - 2025-07-30 23:45:19

### Added

- Made qrf impute sequentially when multiple imputed_variables are passed.

## [1.0.1] - 2025-07-26 16:22:56

### Fixed

- PyPI deployment workflow now properly defines Python version matrix.

## [1.0.0] - 2025-07-24 13:57:14

### Added

- Support for Python 3.13.
- Optional `images` extra for kaleido dependency (`pip install microimpute[images]`).

### Changed

- Require Python 3.13 (dropped support for Python 3.11 and 3.12).
- Updated CI/CD workflows to test only against Python 3.13.
- Updated Black formatter to target Python 3.13 only.
- Simplified documentation dependencies to let jupyter-book manage its own deps.
- Added furo theme as explicit dependency for documentation builds.
- Updated NumPy from 1.26.x to 2.x (major version upgrade).
- Updated SciPy from 1.14.x to 1.16.x.
- Updated joblib from 1.4.x to 1.5.x.
- Updated flake8 from 6.x to 7.x.
- Updated Black to require version 24.0.0 or newer.
- Updated isort to require version 5.13.0 or newer.
- Allowed statsmodels 0.15.x when released.
- Allowed optuna 4.x versions.
- Made kaleido an optional dependency (moved to `images` extra).

## [0.2.5] - 2025-07-24 12:42:09

### Changed

- Made kaleido an optional dependency (install with `pip install microimpute[images]`).
- Image export functionality now gracefully handles missing kaleido with informative warnings.

## [0.2.4] - 2025-07-24 11:46:11

### Changed

- Publishing job to run after versioning in Workflow.

## [0.2.3] - 2025-07-07 09:46:21

### Changed

- Default logging level.
- Autoimpute's output format for imputations

## [0.2.2] - 2025-06-23 14:45:18

### Changed

- Making autoimpute return imputations for all models.

## [0.2.1] - 2025-06-19 14:45:02

### Added

- Suppressed warnings.
- Handled edge case in categorical encoding for receiver data.

## [0.2.0] - 2025-06-18 13:45:52

### Added

- Fixed typo in qrf.py.

## [0.1.5] - 2025-06-18 10:44:16

### Added

- Initialized changelog



[1.14.1]: https://github.com/PolicyEngine/microimpute/compare/1.14.0...1.14.1
[1.14.0]: https://github.com/PolicyEngine/microimpute/compare/1.13.0...1.14.0
[1.13.0]: https://github.com/PolicyEngine/microimpute/compare/1.12.0...1.13.0
[1.12.0]: https://github.com/PolicyEngine/microimpute/compare/1.11.0...1.12.0
[1.11.0]: https://github.com/PolicyEngine/microimpute/compare/1.10.0...1.11.0
[1.10.0]: https://github.com/PolicyEngine/microimpute/compare/1.9.0...1.10.0
[1.9.0]: https://github.com/PolicyEngine/microimpute/compare/1.8.1...1.9.0
[1.8.1]: https://github.com/PolicyEngine/microimpute/compare/1.8.0...1.8.1
[1.8.0]: https://github.com/PolicyEngine/microimpute/compare/1.7.0...1.8.0
[1.7.0]: https://github.com/PolicyEngine/microimpute/compare/1.6.1...1.7.0
[1.6.1]: https://github.com/PolicyEngine/microimpute/compare/1.6.0...1.6.1
[1.6.0]: https://github.com/PolicyEngine/microimpute/compare/1.5.2...1.6.0
[1.5.2]: https://github.com/PolicyEngine/microimpute/compare/1.5.1...1.5.2
[1.5.1]: https://github.com/PolicyEngine/microimpute/compare/1.5.0...1.5.1
[1.5.0]: https://github.com/PolicyEngine/microimpute/compare/1.4.1...1.5.0
[1.4.1]: https://github.com/PolicyEngine/microimpute/compare/1.4.0...1.4.1
[1.4.0]: https://github.com/PolicyEngine/microimpute/compare/1.3.0...1.4.0
[1.3.0]: https://github.com/PolicyEngine/microimpute/compare/1.2.3...1.3.0
[1.2.3]: https://github.com/PolicyEngine/microimpute/compare/1.2.2...1.2.3
[1.2.2]: https://github.com/PolicyEngine/microimpute/compare/1.2.1...1.2.2
[1.2.1]: https://github.com/PolicyEngine/microimpute/compare/1.2.0...1.2.1
[1.2.0]: https://github.com/PolicyEngine/microimpute/compare/1.1.6...1.2.0
[1.1.6]: https://github.com/PolicyEngine/microimpute/compare/1.1.5...1.1.6
[1.1.5]: https://github.com/PolicyEngine/microimpute/compare/1.1.4...1.1.5
[1.1.4]: https://github.com/PolicyEngine/microimpute/compare/1.1.3...1.1.4
[1.1.3]: https://github.com/PolicyEngine/microimpute/compare/1.1.2...1.1.3
[1.1.2]: https://github.com/PolicyEngine/microimpute/compare/1.1.1...1.1.2
[1.1.1]: https://github.com/PolicyEngine/microimpute/compare/1.1.0...1.1.1
[1.1.0]: https://github.com/PolicyEngine/microimpute/compare/1.0.2...1.1.0
[1.0.2]: https://github.com/PolicyEngine/microimpute/compare/1.0.1...1.0.2
[1.0.1]: https://github.com/PolicyEngine/microimpute/compare/1.0.0...1.0.1
[1.0.0]: https://github.com/PolicyEngine/microimpute/compare/0.2.5...1.0.0
[0.2.5]: https://github.com/PolicyEngine/microimpute/compare/0.2.4...0.2.5
[0.2.4]: https://github.com/PolicyEngine/microimpute/compare/0.2.3...0.2.4
[0.2.3]: https://github.com/PolicyEngine/microimpute/compare/0.2.2...0.2.3
[0.2.2]: https://github.com/PolicyEngine/microimpute/compare/0.2.1...0.2.2
[0.2.1]: https://github.com/PolicyEngine/microimpute/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/PolicyEngine/microimpute/compare/0.1.5...0.2.0
