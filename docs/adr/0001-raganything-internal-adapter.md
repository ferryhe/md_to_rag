# ADR 0001: Keep RAG-Anything Internal

Status: accepted
Date: 2026-06-19

## Context

md_to_rag needs a stable public contract for Markdown ingestion, chunking,
embedding, indexing, querying, inspection, diff, and rebuild behavior.
HKUDS/RAG-Anything is supported only as a possible optional backend, with target
dependency `raganything>=1.3.1,<2.0`.

The project must avoid coupling downstream CLI, API, MCP, or artifact consumers to upstream backend objects.

## Decision

RAG-Anything is an optional internal adapter/backend only.

The internal adapter may document and use these upstream touchpoints:

- `RAGAnythingConfig`
- `insert_content_list(...)`
- `aquery(...)`

The public md_to_rag CLI, Python API, MCP tools, and artifacts expose only md_to_rag-owned schemas, paths, status payloads, query results, and citation records.

## Consequences

- Public consumers can use md_to_rag without installing or understanding RAG-Anything.
- Optional backend changes are isolated behind adapter tests.
- Any upstream result must be normalized before crossing a public md_to_rag boundary.
- The native md_to_rag artifact pipeline remains the default CLI/API/MCP path.
