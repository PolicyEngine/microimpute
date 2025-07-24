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

## Development Workflow

### Before Pushing Code

1. **Always run formatting tools**: `make format`
   - This runs Black, isort, and linecheck
   - Fixes most formatting issues automatically
2. **Check your formatting**: `make check-format`
   - Ensures your code will pass CI linting checks
3. **Run tests locally**: `make test`
   - Catch test failures before CI
4. **Check for trailing newlines**: All files must end with a newline
   - The linecheck tool will catch this

### Common CI Failures

- **Linting failures**: Run `make format` to fix
- **Missing trailing newlines**: Run `make format` or add manually
- **Import order issues**: Run `isort microimpute/`
- **Line length issues**: Run `black microimpute/ --line-length 79`

## Code Integrity and Test Data Handling

- **NEVER HARDCODE LOGIC JUST TO PASS SPECIFIC TEST CASES**
  - This undermines code quality and model integrity
  - It creates technical debt and maintenance nightmares
  - It destroys the ability to trust results
  - Never add conditional logic that returns fixed values for specific input combinations
