# md_to_rag

Python package and CLI skeleton for Markdown-to-RAG artifact workflows.

```bash
md-to-rag --help
```

Initialize a local artifact project and inspect its manifest:

```bash
md-to-rag init ./rag-artifacts --json
md-to-rag inspect ./rag-artifacts --json
```

Ingest Markdown files from an initialized project:

```bash
md-to-rag init ./rag-artifacts --json
# write Markdown files under ./rag-artifacts/source
md-to-rag ingest --source ./rag-artifacts/source --json
```

`ingest` writes portable JSONL artifacts to `source/source_manifest.jsonl` and `documents/documents.jsonl`.
