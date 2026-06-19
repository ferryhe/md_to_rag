# Changelog

All notable project changes are tracked here. This project uses PEP 440
package versions, with stable releases following `MAJOR.MINOR.PATCH`.

See [Versioning](docs/versioning.md) for the release policy, artifact schema
policy, and release checklist.

## [Unreleased]

- Added the versioning and release management policy.
- Added this changelog as the durable release history entry point.

## [0.1.0] - 2026-06-19

- Established `md-to-rag` as the public CLI for Markdown to portable RAG
  artifacts.
- Added the public Python API surface under `md_to_rag.api`.
- Added MCP tool metadata for the public command set.
- Implemented `init`, `ingest`, `chunk`, `embed`, `index`, `query`, `inspect`,
  `diff`, and `rebuild`.
- Added deterministic local embeddings, index artifacts, local query results,
  drift inspection, and rebuild orchestration.
- Added the optional internal RAG-Anything adapter boundary while keeping
  upstream objects out of public CLI/API/MCP/artifact contracts.
- Added GitHub Actions CI for Python 3.11 and 3.12.
