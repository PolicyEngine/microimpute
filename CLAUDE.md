# MicroImpute - Developer Guide

## Code Style Guidelines

### Formatting & Organization
- Use 4 spaces for indentation
- Maximum line length: 79 characters (Black default)
- Format code with Black: `black microimpute/`
- Sort imports with isort: `isort microimpute/`
- **IMPORTANT**: All files must end with a newline character
  - This is enforced by the `linecheck` tool in CI
  - Run `make check-format` locally before pushing to catch these errors

### Naming & Types
- Use snake_case for variables, functions, and modules
- Use CamelCase for classes
- Constants should be UPPERCASE
- Add type hints to all function parameters and return values
- Document functions with ReStructuredText-style docstrings

### Imports
- Group imports: standard library, third-party, local modules
- Import specific functions/classes rather than entire modules when practical

### Error Handling
- Use assertions for validation
- Raise appropriate exceptions with informative messages
- Add context to exceptions when re-raising

## Python Version Upgrades
When upgrading Python versions:
1. Update `pyproject.toml`:
   - Update `requires-python` to include new versions
   - Update Black's `target-version` to include all supported versions
2. Update GitHub Actions workflows:
   - Add new Python versions to the test matrix in `.github/workflows/pr.yaml` and `.github/workflows/main.yml`
   - Update single-version jobs to use the latest Python version
3. Run `uv lock` to update the lockfile with new dependencies
4. Create a changelog entry with a minor version bump
5. Run `make check-format` locally before pushing to ensure all files have trailing newlines
6. If CI fails on linting, run `make format` to fix formatting issues automatically

## Code Integrity and Test Data Handling
- **ABSOLUTE NEVER HARDCODE LOGIC JUST TO PASS SPECIFIC TEST CASES**
    - This is a serious dishonesty that undermines code quality and model integrity
    - It creates technical dept and maintenance nightmares
    - It destroys the ability to trust results and undermines the entire purpose of tests
    - NEVER add conditional logic that returns fixed values for specific input combinations