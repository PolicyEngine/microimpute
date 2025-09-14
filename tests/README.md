# Microimpute tests

This directory contains tests for the Microimpute package.

## Test structure

The test suite is organized as follows:

- **test_basic.py**: Basic functionality tests for the package
- **test_quantile_comparison.py**: Tests the end-to-end quantile loss comparison workflow
  - Verifies the data preparation functions
  - Tests imputation with multiple model types
  - Compares quantile loss metrics across models
  - Tests visualization of loss comparisons
- **test_models/**: Model-specific tests (see [test_models/README.md](test_models/README.md) for details)
  - Tests for each model implementation
  - Tests for the common Imputer interface
  - Examples of model usage

## Running tests

To run the tests, use the following command from the project root:

```bash
python -m pytest tests/
```

For more verbose output:

```bash
python -m pytest tests/ -v
```
