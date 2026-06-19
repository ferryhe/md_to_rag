# md_to_rag Public Contract v1

Status: active
Date: 2026-06-19

## Scope

This contract describes the public md_to_rag surface after the Markdown-to-RAG
artifact pipeline implementation. Runtime internals may evolve, but the public
CLI, Python API, MCP metadata, artifacts, and response envelopes are owned by
md_to_rag and must remain backend-neutral.

md_to_rag owns the Markdown-to-RAG artifact contract for this pipeline:

```text
web_listening -> doc_to_md -> md_to_rag -> rag_to_agent/domain adapters -> ai_interface
```

Sibling repositories integrate through files, manifests, and tool specs only.

## Public CLI

The public command is `md-to-rag`.

Supported v1 commands:

- `md-to-rag init`
- `md-to-rag ingest`
- `md-to-rag chunk`
- `md-to-rag embed`
- `md-to-rag index`
- `md-to-rag query`
- `md-to-rag inspect`
- `md-to-rag diff`
- `md-to-rag rebuild`

All commands support stable `--json` responses. Command behavior, artifact
paths, manifest semantics, and JSON output contracts must remain compatible
across future changes.

## Public API and MCP

The public Python API is `md_to_rag.api`. MCP metadata is exposed through
`md_to_rag.mcp.list_tools()`.

The public Python API and MCP tool schemas expose only md_to_rag-owned
request/response schemas, artifact paths, manifest metadata, status payloads,
query results, and citations.

Public surfaces must not require callers to import, instantiate, serialize, or inspect objects from optional backend packages.

## Artifact Contract

Artifacts are manifest-driven, JSON/JSONL-first, path-portable, and provenance-preserving.

Required invariants:

- Stable IDs and hashes for source documents, chunks, embeddings, and indexes.
- Idempotent reruns for unchanged inputs and profiles.
- Every derived row can trace back to input source material.
- Provider and engine profiles are recorded without secrets.
- Secrets, credentials, local `.env` values, and generated keys are never written into manifests.

## Backend Boundary

Native md_to_rag artifacts are the public contract. Backends are implementation details.

RAG-Anything may be used only as an optional internal adapter/backend with
target dependency:

```text
raganything>=1.3.1,<2.0
```

The documented internal adapter touchpoints are:

- `RAGAnythingConfig`
- `insert_content_list(...)`
- `aquery(...)`

No RAG-Anything object, response class, config object, or exception type may be exposed through public CLI output, Python API, MCP result schemas, or artifact manifests. The adapter must normalize upstream behavior into md_to_rag-owned schemas.
