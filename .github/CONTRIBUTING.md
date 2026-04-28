## Updating the code

Please make sure that introduced changes are consistent with the testing api and add additional tests if relevant.

## Updating the versioning

Add a towncrier fragment under `changelog.d/` with the format
`<short-description>.<type>.md`, where `<type>` is one of `added`,
`changed`, `fixed`, `removed`, or `breaking`.

Do not edit `CHANGELOG.md` directly in feature PRs. After the PR is merged,
the versioning workflow runs `make changelog`, deletes the consumed fragments,
updates `CHANGELOG.md`, bumps `pyproject.toml`, and commits the result as
`Update package version`.
