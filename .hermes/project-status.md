# md_to_rag Project Status

Last updated: 2026-06-17

## Scope

- Repo: `md_to_rag` (local checkout path varies by worker)
- Active branch: `codex/contract-freeze-raganything-v1`
- Active PR lane: PR1, pre-publish verification in progress
- Sibling repos: off-limits unless a future task explicitly names them
- Current task: contract freeze and managed-PR status only

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
| PR1 | `codex/contract-freeze-raganything-v1` | Fixing Copilot comments | Freeze public contract, RAG-Anything boundary, managed-PR policy, and controller ledger. |
| PR2 | TBD | Queued | Python package and CLI skeleton. |
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

Command used:

```bash
git fetch origin main
npx --yes @openai/codex -c 'model="gpt-5.5"' review --base origin/main
```

## Notes

- PR1 must not implement runtime Python package or CLI behavior.
- Keep changes limited to the allowed documentation and ledger files.
