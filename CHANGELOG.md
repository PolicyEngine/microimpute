# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), 
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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



[1.0.2]: https://github.com/PolicyEngine/microimpute/compare/1.0.1...1.0.2
[1.0.1]: https://github.com/PolicyEngine/microimpute/compare/1.0.0...1.0.1
[1.0.0]: https://github.com/PolicyEngine/microimpute/compare/0.2.5...1.0.0
[0.2.5]: https://github.com/PolicyEngine/microimpute/compare/0.2.4...0.2.5
[0.2.4]: https://github.com/PolicyEngine/microimpute/compare/0.2.3...0.2.4
[0.2.3]: https://github.com/PolicyEngine/microimpute/compare/0.2.2...0.2.3
[0.2.2]: https://github.com/PolicyEngine/microimpute/compare/0.2.1...0.2.2
[0.2.1]: https://github.com/PolicyEngine/microimpute/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/PolicyEngine/microimpute/compare/0.1.5...0.2.0
