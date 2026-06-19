# Versioning

`md_to_rag` uses explicit package, artifact, and release versioning. The public
contract covers the CLI, Python API, MCP tool metadata, and md_to_rag-owned
artifact schemas.

## Package Version

The package version is recorded in `pyproject.toml` under
`project.version`. Runtime imports expose the same value as
`md_to_rag.__version__`.

`tests/test_public_shells.py` verifies that `pyproject.toml` and
`src/md_to_rag/__init__.py` stay synchronized. A release PR must update both
files together until the project adopts a generated single-source version.

Stable package releases follow `MAJOR.MINOR.PATCH`:

- `MAJOR`: breaking public CLI, Python API, MCP, or artifact contract changes.
- `MINOR`: backwards-compatible public behavior, commands, options, schemas, or
  optional backend capabilities.
- `PATCH`: backwards-compatible bug fixes, documentation fixes, dependency
  constraint fixes, and internal hardening.

Pre-release and local development versions should use valid PEP 440 syntax.

## Artifact Versions

Artifact compatibility is versioned separately from the package release:

- `schema_version` identifies the JSON/JSONL row or manifest schema.
- `index_version` identifies the local index format.
- `md_to_rag_version` records the package version that produced a project
  manifest.

Changing an artifact schema or index format requires a package version bump.
Backwards-compatible readers may continue accepting older artifact versions, but
public writers should emit the current owned schema versions documented in the
contract tests.

## Changelog

`CHANGELOG.md` is the durable release history. New work should add entries under
`Unreleased`. A release PR moves those entries into a dated version section.

Use these headings when helpful:

- Added
- Changed
- Fixed
- Deprecated
- Removed
- Security

## Release Checklist

1. Start from the latest `main` on a task branch.
2. Update `pyproject.toml` and `src/md_to_rag/__init__.py` to the target
   package version.
3. Move `CHANGELOG.md` entries from `Unreleased` to a dated version section.
4. Update public docs or contract docs for any user-visible behavior changes.
5. Run the required local validation and the Pre-PR Codex Review Gate.
6. Merge the release PR after CI and review comments are resolved.
7. Tag the merge commit on `main` with `vX.Y.Z`.
8. Push the tag and create a GitHub Release using the changelog entry.

Do not tag a release from an unmerged task branch.
