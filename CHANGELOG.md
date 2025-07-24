# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), 
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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



[0.2.5]: https://github.com/PolicyEngine/microimpute/compare/0.2.4...0.2.5
[0.2.4]: https://github.com/PolicyEngine/microimpute/compare/0.2.3...0.2.4
[0.2.3]: https://github.com/PolicyEngine/microimpute/compare/0.2.2...0.2.3
[0.2.2]: https://github.com/PolicyEngine/microimpute/compare/0.2.1...0.2.2
[0.2.1]: https://github.com/PolicyEngine/microimpute/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/PolicyEngine/microimpute/compare/0.1.5...0.2.0
