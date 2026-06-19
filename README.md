# md_to_rag

`md_to_rag` is a Python CLI and library for turning Markdown corpora into
portable RAG artifacts. It owns the project manifest, source/document/chunk
JSONL artifacts, deterministic local embeddings, local vector index, query
results, drift inspection, and rebuild orchestration.

The stable public surfaces are:

- CLI: `md-to-rag`
- Python API: `md_to_rag.api`
- MCP tool metadata: `md_to_rag.mcp.list_tools()`

RAG-Anything support is optional and internal. The default CLI/API/MCP contract
uses md_to_rag-owned schemas and artifacts; it does not expose upstream
RAG-Anything objects, configs, exceptions, or response classes.

## Install

```bash
python -m pip install -e .
```

Install the optional internal adapter dependency only when developing or testing
the RAG-Anything backend boundary:

```bash
python -m pip install -e ".[raganything]"
```

## CLI Quickstart

```bash
md-to-rag init ./rag-artifacts --json

# Create Markdown files under ./rag-artifacts/source, then run the pipeline
# from the initialized project directory.
cd ./rag-artifacts
md-to-rag ingest --source source --json
md-to-rag chunk --manifest documents/documents.jsonl --json
md-to-rag embed --chunks chunks/chunks.jsonl --json
md-to-rag index --embeddings embeddings/embeddings.jsonl --json
md-to-rag query "What does this corpus say?" --json
```

Useful maintenance commands:

```bash
md-to-rag inspect . --json
md-to-rag diff . --json
md-to-rag rebuild . --json
```

All commands support `--json` and return stable md_to_rag-owned response
envelopes with `command`, `status`, `message`, optional `artifact_path`,
optional typed `error`, and typed `data`.

## Artifact Layout

An initialized project contains:

```text
corpus_manifest.json
source/
  source_manifest.jsonl
documents/
  documents.jsonl
chunks/
  chunks.jsonl
embeddings/
  embeddings.jsonl
indexes/
  index_manifest.json
  vectors.jsonl
reports/
```

Key guarantees:

- Artifacts are JSON/JSONL-first, path-portable, and manifest-driven.
- Reruns are idempotent when inputs and profiles have not changed.
- Derived rows preserve provenance back to source Markdown.
- Provider and engine profiles are recorded without secrets.
- `diff` is read-only; `rebuild` runs `ingest -> chunk -> embed -> index` and
  stops on typed errors.

## Python API

```python
from md_to_rag import api

api.init("rag-artifacts")
api.ingest("rag-artifacts/source")
api.chunk("rag-artifacts/documents/documents.jsonl")
api.embed("rag-artifacts/chunks/chunks.jsonl")
api.index("rag-artifacts/embeddings/embeddings.jsonl")
```

`api.query("...")` resolves the active project from the current working
directory, matching the CLI query behavior.

## MCP Metadata

```python
from md_to_rag import mcp

tools = mcp.list_tools()
```

`list_tools()` returns one md_to_rag-owned tool definition per command, with
owned input and output JSON schemas for `init`, `ingest`, `chunk`, `embed`,
`index`, `query`, `inspect`, `diff`, and `rebuild`.

## CI

GitHub Actions runs on pushes to `main` and on pull requests. The workflow
installs the package, runs the test suite, and checks the CLI help surface on
Python 3.11 and 3.12.
