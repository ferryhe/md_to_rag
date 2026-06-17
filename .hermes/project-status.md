# md_to_rag Project Status

Last updated: 2026-06-17

## Scope

- Repo: `md_to_rag` (local checkout path varies by worker)
- Active branch: `codex/package-interface-shells`
- Active PR lane: PR2, controller verification after review fixes
- Sibling repos: off-limits unless a future task explicitly names them
- Current task: Python package, CLI/API/MCP skeleton, owned schemas, and focused tests only

## Current Contract Decisions

- Keep the current public CLI/API/MCP/artifact contract.
- Public CLI command: `md-to-rag`.
- Public v1 commands: `init`, `ingest`, `chunk`, `embed`, `index`, `query`, and `inspect`.
- Later compatible commands: `diff` and `rebuild`.
- RAG-Anything is optional internal adapter/backend only.
- Optional dependency target: `raganything>=1.3.1,<2.0`.
- Internal adapter touchpoints to document: `RAGAnythingConfig`, `insert_content_list(...)`, and `aquery(...)`.
- RAG-Anything objects must not be exposed through public CLI, API, MCP, or artifact manifests.

## Managed PR Ledger

| PR | Branch | Status | Scope |
| --- | --- | --- | --- |
| PR1 | `codex/contract-freeze-raganything-v1` | Merged (#3) | Freeze public contract, RAG-Anything boundary, managed-PR policy, and controller ledger. |
| PR2 | `codex/package-interface-shells` | Pre-publish verification | Python package and CLI/API/MCP skeleton with owned schemas and tests. |
| PR3 | TBD | Queued | `init` and `inspect`. |
| PR4 | TBD | Queued | `ingest`. |
| PR5 | TBD | Queued | `chunk`. |
| PR6 | TBD | Queued | `embed` and cache/profile behavior. |
| PR7 | TBD | Queued | Native `index` and `query`. |
| PR8 | TBD | Queued | Compatible `diff` and `rebuild`. |
| PR9 | TBD | Queued | Optional internal RAG-Anything backend. |

## Completion Policy

For each PR, required verification includes focused/full checks for touched files plus the Pre-PR Codex Review Gate. For this managed-PR program, the user explicitly authorized on 2026-06-17 that once checks pass and valid Copilot/remote comments are resolved, the controller may merge the PR to `main` and delete that PR's remote/local task branch.

Branch deletion outside this scoped managed-PR program, force-push, history rewrite, broad cleanup, removing unrelated files, or deleting work outside the managed PR completion flow still requires fresh explicit approval.

## PR1 Verification

- Passed: `git diff --check`
- Not present: repo-configured markdown/static checks; no `package.json`, `pyproject.toml`, markdownlint, prettier, or remark command/config was found
- Passed: Pre-PR Codex Review Gate via npm Codex CLI fallback because the WindowsApps `codex`/`codex.exe` launcher fails with `Access is denied`
- Fixed: Copilot comments on the OS-specific status path and plan grammar issue; reran `git diff --check` and the Pre-PR Codex Review Gate

Command used:

```bash
git fetch origin main
npx --yes @openai/codex -c 'model="gpt-5.5"' review --base origin/main
```

## PR2 Verification

- Scope: Python 3.11+ package `md-to-rag` version `0.1.0`; Typer CLI commands `init`, `ingest`, `chunk`, `embed`, `index`, `query`, and `inspect`; owned Pydantic schemas; API facade; MCP metadata skeleton; focused tests.
- Passed: `pytest` (9 tests)
- Passed: `python -m md_to_rag --help`
- Passed: installed `md-to-rag --help` and every command `--help`
- Passed: idempotent JSON skeleton smoke with `md-to-rag query 'What artifacts exist?' --json`
- Passed: every command `--json` skeleton output parses as JSON
- Passed: `git diff --check` with a CRLF conversion warning for `.hermes/project-status.md`
- Passed: Pre-PR Codex Review Gate via `npx.cmd --yes @openai/codex -c 'model="gpt-5.5"' review --base origin/main`; native `codex.exe` failed with `Access is denied`, and `npx.ps1` was blocked by PowerShell execution policy.
- Fixed after code-quality review: MCP tool metadata now uses per-command input schemas, and public request/response payloads are constrained to JSON-compatible values.
- Fixed after Codex review: dependency lower bounds now require `pydantic>=2.5,<3` for `JsonValue` and `typer>=0.16,<1` for the CLI annotations used by the skeleton.

## Notes

- PR2 must not implement real runtime `init`, `ingest`, `chunk`, `embed`, `index`, `query`, or `inspect` behavior beyond stable typed skeleton responses and help surfaces.
- RAG-Anything remains optional internal adapter/backend only and is not a default dependency.
