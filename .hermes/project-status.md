# md_to_rag Project Status

Last updated: 2026-06-18

## Scope

- Repo: `md_to_rag` (local checkout path varies by worker)
- Active branch: `codex/ingest-hardening-followup`
- Active PR lane: PR4a open as #7; Copilot comments addressed locally, follow-up verification passed
- Sibling repos: off-limits unless a future task explicitly names them
- Current task: Real `ingest` behavior only; `chunk`, `embed`, `index`, and `query` remain typed skeletons

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
| PR2 | `codex/package-interface-shells` | Merged (#4) | Python package and CLI/API/MCP skeleton with owned schemas and tests. |
| PR3 | `codex/manifest-init-inspect` | Merged (#5) | Real `init` and `inspect`. |
| PR4 | `codex/ingest-documents` | Merged (#6) | `ingest`. |
| PR4a | `codex/ingest-hardening-followup` | Open (#7); Copilot comments addressed locally and verified | `ingest` hardening for portable paths and artifact write boundaries. |
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
- Fixed after Copilot review: public skeleton messages no longer mention PR2, and the version test compares against `pyproject.toml`.

## Notes

- PR4 scope is limited to real `ingest`: read Markdown files/directories or doc_to_md JSON/JSONL manifests inside an initialized project and emit portable `source/source_manifest.jsonl` and `documents/documents.jsonl` artifacts.
- PR4 must preserve typed skeleton responses for `chunk`, `embed`, `index`, and `query`; do not implement their runtime behavior.
- RAG-Anything remains optional internal adapter/backend only and is not a default dependency.

## PR4 Verification Plan

- Required: `pytest`
- Required: `python -m md_to_rag --help`
- Required: installed `md-to-rag --help`
- Required: every command `--help`
- Required: CLI smoke with `md-to-rag init <tmp> --json`, write Markdown sources, `md-to-rag ingest --json` twice, and `md-to-rag inspect --json`
- Required: `git diff --check`
- Required: Pre-PR Codex Review Gate via native `codex` or `npx.cmd --yes @openai/codex` fallback if native remains blocked

## PR3 Verification

- Scope: real `init` creates `corpus_manifest.json` plus `source/`, `documents/`, `chunks/`, `embeddings/`, `indexes/`, and `reports/`; real `inspect` reads md_to_rag-owned manifest/status schemas and returns typed missing/invalid artifact responses; non-PR3 commands keep typed skeleton responses.
- Passed: `pytest` (21 tests)
- Passed: `python -m md_to_rag --help`
- Passed: installed `md-to-rag --help`
- Passed: every command `--help`
- Passed: installed CLI smoke with `md-to-rag init <tmp> --json`, rerun same command, `md-to-rag inspect <tmp> --json`, and `md-to-rag inspect <missing> --json`
- Passed: `git diff --check` with CRLF conversion warnings for touched text files only
- Native `codex.exe` review gate attempt failed with `Access is denied`
- Passed: Pre-PR Codex Review Gate via `npx.cmd --yes @openai/codex -c 'model="gpt-5.5"' review --base origin/main`
- Resolved local Codex review findings: untracked manifest helper included in diff via intent-to-add, typed init filesystem/write errors, target-anchored missing-artifact lookup, manifest schema marker validation, invalid-manifest reporting, and `inspect` marked implemented in generated manifests.
- Resolved controller code-quality review findings: MCP init/inspect output schemas now require the real response envelope, init error data uses an owned empty payload schema, and `init.changed` reports repaired artifact directories.
- Resolved PR #5 Copilot comments: repair/upgrade init runs now report `Project updated.`, and backfilled manifest status rows use the normalization timestamp instead of the project creation timestamp.
- Merged: PR #5 squash-merged to `main` at `fe86424`; remote branch and local task branch were deleted.

## PR4 Verification

- Scope: real `ingest` reads Markdown files/directories and doc_to_md JSON/JSONL manifests inside initialized projects, writes portable `source/source_manifest.jsonl` and `documents/documents.jsonl`, updates ingest manifest status, and leaves `chunk`, `embed`, `index`, and `query` as typed skeletons.
- Passed: `pytest` (44 passed, 8 skipped)
- Passed: `python -m md_to_rag --help`
- Passed: installed `md-to-rag --help`
- Passed: every command `--help`
- Passed: installed CLI smoke with `md-to-rag init <tmp> --json`, Markdown source creation, `md-to-rag ingest --source <tmp>/source --json` twice, and `md-to-rag inspect <tmp> --json`
- Passed: `git diff --check` with CRLF conversion warnings for touched text files only
- Resolved controller/reviewer findings: doc_to_md upstream paths are validated portably across POSIX/Windows absolute path forms; doc_to_md duplicate Markdown rows get unique document IDs; manifest row reordering no longer changes stable source/document IDs or stable source hashes; stale ingest manifest status is repaired even when artifact bytes are unchanged.
- Resolved Codex review findings: doc_to_md row Markdown paths now reject traversal, absolute paths, non-Markdown files, and generated artifact targets before reading; duplicate doc_to_md identities now return a typed error instead of emitting duplicate IDs.
- Resolved final Codex review finding: out-of-project resolved Markdown paths encountered during directory traversal now raise typed `source_outside_project` errors instead of leaking sort-key `TypeError` tracebacks.
- Resolved final path portability finding: Windows drive-relative doc_to_md Markdown and upstream paths such as `C:source/doc.md` and `C:raw/report.pdf` are rejected.
- Resolved final provenance finding: doc_to_md upstream document IDs used for identity are now preserved in visible provenance/source rows and included in stable source hashes.
- Passed: Pre-PR Codex Review Gate via `npx.cmd --yes @openai/codex -c 'model="gpt-5.5"' review --base origin/main`; native `codex.exe` remains blocked by `Access is denied`.
- Published: PR #6 at `https://github.com/ferryhe/md_to_rag/pull/6`.
- Merged: PR #6 squash-merged to `main` at `b305029`.
- Resolved post-publish local Codex review finding: generated artifact directories such as `documents/`, `chunks/`, `embeddings/`, `indexes/`, and `reports/` are rejected as ingest sources without overwriting existing artifacts.
- Resolved post-publish local Codex review findings: upstream URI provenance such as `https://example.com/a.pdf` is preserved without path normalization, and generated artifact directories are rejected even when reached through project-root traversal or doc_to_md manifest rows.
- Resolved post-publish local Codex review findings: Windows drive-looking upstream paths such as `C://raw/report.pdf` are rejected before URI acceptance, and doc_to_md `metadata.title` is preserved when no top-level title is supplied.
- Resolved post-publish local Codex review finding: netloc-less upstream URIs such as `file:///tmp/a.pdf` are preserved without path normalization.
- Resolved Copilot PR #6 feedback: the stale syntax-error comment was disproven by import/test validation, and doc_to_md manifest rows now sort by stable identity for same-Markdown multi-upstream rows.
- Resolved post-publish local Codex review finding: explicit symlink sources are anchored to the lexical source project before resolved-path boundary checks, preventing cross-project artifact writes.
- Resolved post-publish local Codex review finding: linked source directories are not treated as lexical manifest roots, so linked project directories are rejected by boundary checks instead of mutating the target project.
- Resolved post-publish local Codex review finding: lexical manifest search skips linked source ancestors, so nested paths under linked project directories are also rejected by boundary checks.
- Resolved post-publish local Codex review findings: malformed upstream URIs return typed `manifest_path_not_portable` errors, and non-finite manifest values are rejected before portable JSONL artifacts are written.
- Resolved post-publish local Codex review finding: generated artifact output paths are resolved and checked under `project_root` before reading or writing, rejecting symlinked output directories outside the project.
- Resolved post-publish local Codex review finding: manifest Markdown and upstream paths containing embedded NUL characters return typed portable-path errors instead of escaping CLI JSON handling.
- Resolved post-publish local Codex review finding: malformed scheme-like upstream values such as `http:/example.com/a.pdf` and `https:\example.com\a.pdf` return typed portable-path errors instead of being persisted as provenance.
- Resolved post-publish local Codex review finding: manifest Markdown path components with Windows-reserved characters such as `:` and `?` are rejected to avoid alternate-stream or non-portable source paths.
- Resolved post-publish local Codex review finding: symlinked project roots remain valid manifest roots, while linked nested source directories still resolve through the lexical parent project and boundary checks.
- Resolved post-publish local Codex review findings: direct Markdown paths now use the same portable-component validation as manifest Markdown paths, and directory ingest rejects `.md` symlinks that resolve to non-Markdown targets.
- Resolved post-publish local Codex review finding: non-URI upstream paths now reject Windows-reserved characters such as `?` and alternate-stream `:` before provenance is persisted.
- Resolved post-publish local Codex review finding: generated artifact output files cannot be symlinks or linked paths, preventing writes from corrupting in-project manifest/source files.
- Resolved post-publish local Codex review findings: generated artifact output files cannot be hard links, and Windows-reserved path components such as `CON.md`, trailing dot/space names, and control characters are rejected.
- Resolved post-publish local Codex review findings: overflowing JSON floats such as `1e999` are rejected, and unresolvable requested source paths return typed `source_path_unresolvable` errors.
- Resolved post-publish local Codex review findings: dot-dot source paths now select the nearest initialized project, and manifest Markdown symlink targets are revalidated after resolution for generated artifact and non-Markdown targets.
- Resolved post-publish local Codex review findings: doc_to_md manifest provenance paths are validated before persistence, and literal backslashes in resolved source paths are rejected instead of being rewritten.
- PR4a validation passed on latest `origin/main`: `python -m pytest tests/test_ingest_documents.py` (30 passed, 11 skipped), `python -m pytest` (51 passed, 11 skipped), CLI help checks, installed CLI init/ingest/inspect smoke, and `git diff --check` with CRLF warnings only.
- Resolved PR4a local Codex review finding: unresolvable generated artifact output paths such as symlink loops return typed `artifact_path_collision` errors.
- Resolved PR4a local Codex review finding: nested linked projects inside a symlinked project root use the outer linked manifest fallback and are rejected by boundary checks instead of writing target project artifacts.
- Resolved PR4a local Codex review findings: real nested initialized projects under a symlinked project root keep their nearest manifest, and unresolvable generated artifact directories return typed `artifact_path_collision` errors before source-artifact checks can crash.
- Resolved PR4a local Codex review finding: linked paths from a parent project into an initialized nested project now return typed `source_nested_project` errors instead of writing nested content to parent artifacts.
- Resolved PR4a local Codex review finding: valid project roots reached through multiple symlink/junction components are accepted when no parent project manifest is present.
- Resolved PR4a local Codex review finding: direct filesystem Markdown paths with leading whitespace are preserved in source/document provenance instead of being trimmed.
- Resolved PR4a local Codex review findings: Markdown paths discovered through directory traversal or doc_to_md manifests are checked for nested initialized projects before row emission, and no-source ingest now uses lexical manifest lookup for linked working directories.
- Passed: PR4a Pre-PR Codex Review Gate via `npx.cmd --yes @openai/codex -c 'model="gpt-5.5"' review --base origin/main`; native `codex.exe` remains blocked by `Access is denied`.
- Published: PR #7 at `https://github.com/ferryhe/md_to_rag/pull/7`.
- Resolved PR #7 Copilot comments locally: hoisted Windows reserved basenames to a module constant and simplified non-finite manifest JSON test construction.
- Passed: PR #7 Copilot/review follow-up validation with `python -m pytest tests/test_ingest_documents.py` (32 passed, 11 skipped), `python -m pytest` (53 passed, 11 skipped), CLI smoke, `git diff --check`, and Pre-PR Codex Review Gate.
- Resolved PR4a local Codex review finding: directory ingest now checks linked directory entries before Markdown file filtering, so POSIX directory symlinks into nested initialized projects return typed errors instead of producing incomplete parent artifacts.
- Resolved PR4a local Codex review findings: directory ingest now walks with linked-directory pruning before recursion, and Windows reserved basenames include `CONIN$` and `CONOUT$`.
- Required next: push the Copilot fixes, then recheck PR #7 for remote feedback.
