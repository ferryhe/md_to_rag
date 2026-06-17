# md_to_rag CLI and RAG Artifact v1 Contract Plan

> **For Hermes/Codex:** Use the project-isolated Codex worker pattern from `AGENTS.md`. Read `AGENTS.md`, read `.hermes/project-status.md` if present, run `git status --short --branch`, restate repo/branch/scope, and only then edit. After required verification and the Pre-PR Codex Review Gate pass, routine commit, push, and PR creation/update are authorized. For the managed-PR program explicitly authorized on 2026-06-17, after checks and valid Copilot/remote comments are resolved, the controller may merge to `main` and delete that PR's remote/local task branch. Outside this scoped program, force-push, history rewrite, branch deletion, broad cleanup, or deleting unrelated work still requires fresh explicit approval.

**Goal:** Freeze the public md_to_rag contract before runtime implementation: a domain-neutral Markdown corpus to retrievable RAG artifact CLI, API, MCP surface, and artifact layout.

**Architecture:** CLI-first, JSONL-first, manifest-first. Core layers are source, documents, chunks, embeddings, index, query, and inspect. The v1 public CLI is `md-to-rag` with `init`, `ingest`, `chunk`, `embed`, `index`, `query`, and `inspect`; later `diff` and `rebuild` must be compatible additions. Native artifacts remain the public contract. RAG-Anything is optional internal adapter/backend only.

**Tech Stack:** Project-native stack plus CLI-first JSON/JSONL manifests. Python projects should use Typer/Pydantic where already present. Optional RAG-Anything integration targets `raganything>=1.3.1,<2.0` behind internal adapter boundaries.

---

## Context

This repository is one module in the broader agent-operated knowledge pipeline:

```text
web_listening -> doc_to_md -> md_to_rag -> rag_to_agent/domain adapters -> ai_interface
```

Current project role: Markdown -> RAG artifact CLI for `ingest`, `chunk`, `embed`, `index`, `query`, and `inspect`.

Current planning scope: contract freeze first, then implementation on later PRs. Keep the current public CLI/API/MCP/artifact contract and do not implement runtime Python package or CLI behavior in PR1.

## Non-Negotiable Contracts

1. Public CLI command is `md-to-rag`.
2. Public v1 commands are `init`, `ingest`, `chunk`, `embed`, `index`, `query`, and `inspect`.
3. Later `diff` and `rebuild` commands must be compatible additions, not breaking replacements.
4. CLI outputs must be machine-readable and stable (`--json` where applicable).
5. Public API and MCP surfaces expose md_to_rag-owned schemas and artifact paths only.
6. Artifacts must be path-portable and manifest-driven.
7. Reruns must be idempotent.
8. Every derived artifact must preserve provenance back to its input.
9. Secrets/API keys must never be written into manifests or committed files.
10. Cross-repo integration happens through files/manifests/tool specs, not hidden imports.
11. RAG-Anything is optional internal adapter/backend only. Do not expose upstream RAG-Anything objects through public CLI, API, or MCP.
12. The only upstream RAG-Anything adapter touchpoints to document for internal use are `RAGAnythingConfig`, `insert_content_list(...)`, and `aquery(...)`.

## Frozen Public Contract v1

See `docs/contracts/public-contract-v1.md` for the reviewable contract freeze. In summary:

- CLI: `md-to-rag init|ingest|chunk|embed|index|query|inspect`.
- Compatible later additions: `md-to-rag diff` and `md-to-rag rebuild`.
- API/MCP: expose md_to_rag request/response models, manifest paths, artifact status, and normalized query/citation results.
- Artifacts: JSON/JSONL manifests owned by md_to_rag with stable IDs, hashes, provenance, and profiles.
- Backend boundary: native artifacts are public; RAG-Anything can be used only behind an internal adapter.

## Internal RAG-Anything Boundary

RAG-Anything may be added as an optional backend dependency with target range `raganything>=1.3.1,<2.0`. Its objects and response classes are not part of the public contract. Any adapter must normalize all upstream calls into md_to_rag-owned manifests, status payloads, query results, and MCP responses.

The internal adapter may reference:

- `RAGAnythingConfig`
- `insert_content_list(...)`
- `aquery(...)`

No public CLI option, Python API return type, MCP tool result, or artifact manifest may require callers to import or understand RAG-Anything objects.

## Managed PR Sequence

### PR1: Contract freeze and controller ledger

**Objective:** Freeze the public CLI/API/MCP/artifact contract, document RAG-Anything as optional internal-only backend, resolve stale approval contradictions, and create `.hermes/project-status.md`.

**Files:**
- `.hermes/project-status.md`
- `.codex/README.md`
- `docs/plans/2026-05-05-agent-cli-contract-plan.md`
- `docs/contracts/*`
- `docs/adr/*`

**Steps:**
1. Keep changes documentation-only.
2. Record the v1 public contract.
3. Record the internal-only RAG-Anything boundary.
4. Update the managed-PR completion policy.
5. Run documentation checks and the Pre-PR Codex Review Gate before commit/push/PR.

**Verification:** `git diff --check` and any available markdown/static checks. If no markdown/static checks exist, record that explicitly.

### PR2: Python package and CLI skeleton

**Objective:** Create the Python package skeleton, CLI entry point, owned schemas, manifest helpers, and test fixture layout without changing the frozen contract.

**Verification:** `md-to-rag --help`, schema fixture checks, and focused tests.

### PR3: `init` and `inspect`

**Objective:** Implement `md-to-rag init PROJECT` and `md-to-rag inspect --json` against manifest/status files.

**Verification:** Repeat `init` is idempotent; `inspect --json` matches fixtures.

### PR4: `ingest`

**Objective:** Read Markdown folders or doc_to_md manifests and generate source/document artifacts.

**Verification:** Stable `doc_id`, hash, path portability, manifest provenance, and idempotent reruns.

### PR5: `chunk`

**Objective:** Implement semantic Markdown chunking and emit chunk artifacts/profiles.

**Verification:** `heading_path`, line ranges, content hashes, and schema fixtures.

### PR6: `embed`

**Objective:** Implement embedding provider abstraction and cache/profile behavior without storing secrets.

**Verification:** Same chunk/profile reuses cache; changed profile triggers recomputation; secret fields stay out of manifests.

### PR7: `index` and `query`

**Objective:** Implement native index/query flow, initially FAISS, with normalized citations.

**Verification:** Top-k, threshold, missing index error, citation provenance, and `--json` output fixtures.

### PR8: Compatible `diff` and `rebuild`

**Objective:** Add compatible incremental diff/rebuild commands based on source, chunk, embedding, and index hashes.

**Verification:** Editing one Markdown input only recomputes affected documents/chunks/embeddings/index rows.

### PR9: Optional internal RAG-Anything backend

**Objective:** Add optional internal adapter/backend support for `raganything>=1.3.1,<2.0` while preserving md_to_rag public CLI/API/MCP/artifact contracts.

**Verification:** Public API/MCP returns md_to_rag-owned schemas only; no RAG-Anything objects leak; adapter tests cover `RAGAnythingConfig`, `insert_content_list(...)`, and `aquery(...)` normalization.

## Completion Policy

For each PR, required verification includes focused/full checks for touched files plus the Pre-PR Codex Review Gate. For this managed-PR program, the user explicitly authorized on 2026-06-17 that once checks pass and valid Copilot/remote comments are resolved, the controller may merge the PR to `main` and delete that PR's remote/local task branch. Outside this scoped program, destructive actions including branch deletion, force-push, history rewrite, broad cleanup, or deleting unrelated work still require fresh explicit approval.

---

## Acceptance Criteria

- A Codex worker can understand this repo's boundary from `AGENTS.md`.
- A future implementation branch can start from this plan without needing cross-chat context.
- The module's input/output contract is explicit enough for the next module in the chain.
- All new behavior is testable through CLI commands and fixture manifests.

## Recommended First PR

PR1 is documentation/contracts/status only. Do not implement runtime Python package or CLI behavior in PR1. The first PR should make the intended contract reviewable before code follows.
