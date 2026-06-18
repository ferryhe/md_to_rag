from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from md_to_rag import api
from md_to_rag.cli import app
from md_to_rag.schemas import CommandStatus


runner = CliRunner()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _link_directory_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except (NotImplementedError, OSError) as symlink_error:
        if os.name != "nt":
            pytest.skip(f"symlink creation unavailable: {symlink_error}")
    link_text = str(link).replace("'", "''")
    target_text = str(target).replace("'", "''")
    try:
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"New-Item -ItemType Junction -Path '{link_text}' -Target '{target_text}' | Out-Null",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as junction_error:
        pytest.skip(f"directory link creation unavailable: {junction_error}")


def _prepare_embedded_project(project: Path) -> Path:
    api.init(project)
    (project / "source" / "alpha.md").write_text(
        "# Alpha Guide\n\nAlpha retrieval target.\n\nShared words live here.\n",
        encoding="utf-8",
    )
    (project / "source" / "beta.md").write_text(
        "# Beta Notes\n\nBeta material only.\n",
        encoding="utf-8",
    )
    assert api.ingest(source=project / "source").status is CommandStatus.OK
    assert api.chunk(manifest=project / "documents" / "documents.jsonl").status is CommandStatus.OK
    embed_response = api.embed(chunks=project / "chunks" / "chunks.jsonl")
    assert embed_response.status is CommandStatus.OK
    return project / "embeddings" / "embeddings.jsonl"


def test_index_writes_portable_artifacts_and_updates_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    embeddings_path = _prepare_embedded_project(project)
    embedding_rows = _jsonl(embeddings_path)
    monkeypatch.chdir(project / "source")

    response = api.index()

    assert response.__class__.__name__ == "IndexResponse"
    assert response.status is CommandStatus.OK
    assert response.message == "Index artifacts generated."
    assert response.artifact_path == str((project / "indexes" / "index_manifest.json").resolve())
    assert response.data.project_root == str(project.resolve())
    assert response.data.embeddings_path == "embeddings/embeddings.jsonl"
    assert response.data.index_manifest_path == "indexes/index_manifest.json"
    assert response.data.index_path == "indexes/vectors.jsonl"
    assert response.data.changed is True
    assert response.data.embedding_count == len(embedding_rows)
    assert response.data.vector_count == len(embedding_rows)
    assert response.data.index_engine == "md_to_rag.local_vector"

    index_manifest = json.loads(
        (project / "indexes" / "index_manifest.json").read_text(encoding="utf-8")
    )
    assert index_manifest["schema_name"] == "md_to_rag.index"
    assert index_manifest["schema_version"] == "1.0"
    assert index_manifest["embeddings_path"] == "embeddings/embeddings.jsonl"
    assert index_manifest["index_path"] == "indexes/vectors.jsonl"
    assert index_manifest["embeddings_hash"] == response.data.embeddings_hash
    assert index_manifest["index_hash"] == response.data.index_hash
    assert "raganything" not in json.dumps(index_manifest).lower()

    index_rows = _jsonl(project / "indexes" / "vectors.jsonl")
    assert len(index_rows) == len(embedding_rows)
    first_index = index_rows[0]
    first_embedding = embedding_rows[0]
    assert first_index["schema_name"] == "md_to_rag.index_vector"
    assert first_index["schema_version"] == "1.0"
    assert first_index["index_id"].startswith("idx_")
    assert first_index["embedding_id"] == first_embedding["embedding_id"]
    assert first_index["chunk_id"] == first_embedding["chunk_id"]
    assert first_index["source_path"] == first_embedding["source_path"]
    assert first_index["embedding_hash"] == first_embedding["embedding_hash"]
    assert first_index["vector"] == first_embedding["embedding"]
    assert isinstance(first_index["vector_norm"], float)
    assert isinstance(first_index["line_start"], int)
    assert isinstance(first_index["line_end"], int)
    assert isinstance(first_index["heading_path"], list)
    assert first_index["provenance"]["embeddings_path"] == "embeddings/embeddings.jsonl"

    manifest = json.loads((project / "corpus_manifest.json").read_text(encoding="utf-8"))
    index_status = next(
        status for status in manifest["command_status"] if status["command"] == "index"
    )
    assert index_status["status"] == "ok"
    assert index_status["artifact_path"] == "indexes/index_manifest.json"
    assert index_status["data"]["embedding_count"] == len(embedding_rows)
    assert index_status["data"]["vector_count"] == len(embedding_rows)
    assert index_status["data"]["embeddings_path"] == "embeddings/embeddings.jsonl"
    assert index_status["data"]["index_manifest_path"] == "indexes/index_manifest.json"
    assert index_status["data"]["index_path"] == "indexes/vectors.jsonl"
    assert index_status["data"]["index_hash"] == response.data.index_hash
    assert "raganything" not in json.dumps(manifest).lower()


def test_index_rerun_reuses_unchanged_artifacts_and_manifest(tmp_path: Path) -> None:
    project = tmp_path / "project"
    embeddings_path = _prepare_embedded_project(project)

    first = api.index(embeddings=embeddings_path)
    index_manifest_path = project / "indexes" / "index_manifest.json"
    index_path = project / "indexes" / "vectors.jsonl"
    first_index_manifest_bytes = index_manifest_path.read_bytes()
    first_index_bytes = index_path.read_bytes()
    first_manifest_bytes = (project / "corpus_manifest.json").read_bytes()
    first_index_mtime = index_path.stat().st_mtime_ns

    second = api.index(embeddings=embeddings_path)

    assert first.status is CommandStatus.OK
    assert second.status is CommandStatus.OK
    assert second.message == "Index artifacts unchanged."
    assert second.data.changed is False
    assert index_manifest_path.read_bytes() == first_index_manifest_bytes
    assert index_path.read_bytes() == first_index_bytes
    assert index_path.stat().st_mtime_ns == first_index_mtime
    assert (project / "corpus_manifest.json").read_bytes() == first_manifest_bytes


def test_query_returns_deterministic_local_results_from_index(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    _prepare_embedded_project(project)
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK
    monkeypatch.chdir(project)

    first = api.query("alpha retrieval")
    second = api.query("alpha retrieval")

    assert first.__class__.__name__ == "QueryResponse"
    assert first.status is CommandStatus.OK
    assert first.message == "Query results generated."
    assert first.data.question == "alpha retrieval"
    assert first.data.index_manifest_path == "indexes/index_manifest.json"
    assert first.data.index_path == "indexes/vectors.jsonl"
    assert first.data.embeddings_path == "embeddings/embeddings.jsonl"
    assert first.data.result_count == len(first.data.results)
    assert first.data.result_count > 0
    assert first.model_dump(mode="json") == second.model_dump(mode="json")

    top = first.data.results[0]
    assert top.rank == 1
    assert top.score > 0
    assert top.chunk_id.startswith("chk_")
    assert top.embedding_id.startswith("emb_")
    assert top.source_path == "source/alpha.md"
    assert "Alpha" in top.content
    assert "raganything" not in first.model_dump_json().lower()


def test_query_updates_manifest_status_for_inspect(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    _prepare_embedded_project(project)
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK
    monkeypatch.chdir(project)

    response = api.query("alpha retrieval")

    assert response.status is CommandStatus.OK
    manifest = json.loads((project / "corpus_manifest.json").read_text(encoding="utf-8"))
    query_status = next(
        status for status in manifest["command_status"] if status["command"] == "query"
    )
    assert query_status["status"] == "ok"
    assert query_status["message"] == "Query results generated."
    assert query_status["artifact_path"] == "indexes/index_manifest.json"
    assert query_status["data"]["index_manifest_path"] == "indexes/index_manifest.json"
    assert query_status["data"]["index_path"] == "indexes/vectors.jsonl"
    assert query_status["data"]["embeddings_path"] == "embeddings/embeddings.jsonl"
    assert query_status["data"]["result_count"] == response.data.result_count
    assert "question" not in query_status["data"]

    inspect_response = api.inspect(project)
    inspect_status = next(
        status
        for status in inspect_response.data.manifest.command_status
        if status.command.value == "query"
    )
    assert inspect_status.status is CommandStatus.OK
    assert inspect_status.artifact_path == "indexes/index_manifest.json"


def test_query_matches_unicode_only_question_and_markdown(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "aaa.md").write_text(
        "# English Notes\n\nASCII fallback document only.\n",
        encoding="utf-8",
    )
    (project / "source" / "zzz.md").write_text(
        "# 中文指南\n\n苹果检索目标。\n",
        encoding="utf-8",
    )
    assert api.ingest(source=project / "source").status is CommandStatus.OK
    assert api.chunk(manifest=project / "documents" / "documents.jsonl").status is CommandStatus.OK
    assert api.embed(chunks=project / "chunks" / "chunks.jsonl").status is CommandStatus.OK
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK
    monkeypatch.chdir(project)

    response = api.query("苹果检索")

    assert response.status is CommandStatus.OK
    top = response.data.results[0]
    assert top.source_path == "source/zzz.md"
    assert top.score > 0
    assert "苹果检索" in top.content


def test_query_matches_unicode_metadata_when_body_is_ascii(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from md_to_rag.chunk import _hash_text

    project = tmp_path / "project"
    api.init(project)
    content = "# English\n\nOnly ascii body.\n"
    documents_path = project / "documents" / "custom.jsonl"
    _write_jsonl(
        documents_path,
        [
            {
                "schema_name": "md_to_rag.document",
                "schema_version": "1.0",
                "doc_id": "doc_unicode_meta",
                "source_id": "src_unicode_meta",
                "source_path": "source/english.md",
                "source_hash": "sha256:" + "1" * 64,
                "content_hash": _hash_text(content),
                "content": content,
                "line_count": len(content.splitlines()),
                "metadata": {"title": "苹果"},
                "provenance": {},
            }
        ],
    )
    assert api.chunk(manifest=documents_path).status is CommandStatus.OK
    assert api.embed(chunks=project / "chunks" / "chunks.jsonl").status is CommandStatus.OK
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK
    monkeypatch.chdir(project)

    response = api.query("苹果")

    assert response.status is CommandStatus.OK
    assert response.data.result_count >= 1
    assert response.data.results[0].metadata["title"] == "苹果"


@pytest.mark.parametrize("question", ["missingterm", "!!!"])
def test_query_returns_empty_results_for_unmatched_question(
    tmp_path: Path,
    monkeypatch,
    question: str,
) -> None:
    project = tmp_path / "project"
    _prepare_embedded_project(project)
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK
    monkeypatch.chdir(project)

    response = api.query(question)

    assert response.status is CommandStatus.OK
    assert response.data.result_count == 0
    assert response.data.results == []


def test_query_allows_empty_indexed_corpus(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    assert api.ingest(source=project / "source").status is CommandStatus.OK
    assert api.chunk(manifest=project / "documents" / "documents.jsonl").status is CommandStatus.OK
    assert api.embed(chunks=project / "chunks" / "chunks.jsonl").status is CommandStatus.OK
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK
    monkeypatch.chdir(project)

    response = api.query("anything")

    assert response.status is CommandStatus.OK
    assert response.data.result_count == 0
    assert response.data.results == []


def test_query_rejects_non_utf8_question_before_json_serialization(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    _prepare_embedded_project(project)
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK
    monkeypatch.chdir(project)

    response = api.query("bad\ud800")

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "query_invalid_question"
    response.model_dump_json()


def test_query_rejects_non_utf8_status_artifact_path_before_echoing_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    manifest_path = project / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for status in manifest["command_status"]:
        if status["command"] == "index":
            status["status"] = "ok"
            status["message"] = "Index artifacts generated."
            status["artifact_path"] = "indexes/bad\ud800.json"
            break
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    response = api.query("anything")

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "index_path_not_portable"
    assert response.data.index_manifest_path is None
    response.model_dump_json()


def test_query_json_cli_reports_missing_index_without_traceback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    monkeypatch.chdir(project)

    result = runner.invoke(
        app,
        ["query", "What is indexed?", "--json"],
        prog_name="md-to-rag",
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["command"] == "query"
    assert payload["status"] == "missing_artifact"
    assert payload["error"]["code"] == "index_not_found"
    assert "traceback" not in result.output.lower()


def test_query_json_cli_uses_real_response_when_index_exists(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    _prepare_embedded_project(project)
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK
    monkeypatch.chdir(project)

    result = runner.invoke(
        app,
        ["query", "beta material", "--json"],
        prog_name="md-to-rag",
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["command"] == "query"
    assert payload["status"] == "ok"
    assert payload["data"]["result_count"] > 0
    assert payload["data"]["results"][0]["source_path"] == "source/beta.md"
    assert "raganything" not in result.output.lower()


@pytest.mark.parametrize(
    "relative_path",
    [
        "indexes/index_manifest.json",
        "indexes/vectors.jsonl",
        "embeddings/embeddings.jsonl",
        "chunks/chunks.jsonl",
    ],
)
def test_query_rejects_linked_artifact_reads(
    tmp_path: Path,
    monkeypatch,
    relative_path: str,
) -> None:
    project = tmp_path / "project"
    _prepare_embedded_project(project)
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK
    artifact_path = project / relative_path
    outside_artifact = tmp_path / "outside" / artifact_path.name
    outside_artifact.parent.mkdir(parents=True, exist_ok=True)
    outside_artifact.write_bytes(artifact_path.read_bytes())
    artifact_path.unlink()
    try:
        os.link(outside_artifact, artifact_path)
    except OSError as error:
        pytest.skip(f"hard link creation unavailable: {error}")
    monkeypatch.chdir(project)

    response = api.query("alpha retrieval")

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "artifact_path_collision"


def test_index_rejects_linked_embeddings_input(tmp_path: Path) -> None:
    project = tmp_path / "project"
    embeddings_path = _prepare_embedded_project(project)
    linked_embeddings = project / "embeddings" / "linked.jsonl"
    try:
        os.symlink(embeddings_path, linked_embeddings)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlink creation unavailable: {error}")

    response = api.index(embeddings=linked_embeddings)

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "artifact_path_collision"
    assert not (project / "indexes" / "index_manifest.json").exists()


def test_index_allows_explicit_embeddings_under_linked_project_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    linked_project = tmp_path / "linked-project"
    project.mkdir()
    _link_directory_or_skip(linked_project, project)
    api.init(linked_project)
    (linked_project / "source" / "alpha.md").write_text(
        "# Alpha\n\nAlpha retrieval target.\n",
        encoding="utf-8",
    )
    assert api.ingest(source=linked_project / "source").status is CommandStatus.OK
    assert api.chunk(manifest=linked_project / "documents" / "documents.jsonl").status is CommandStatus.OK
    assert api.embed(chunks=linked_project / "chunks" / "chunks.jsonl").status is CommandStatus.OK
    monkeypatch.chdir(linked_project)

    response = api.index(embeddings=Path("embeddings") / "embeddings.jsonl")

    assert response.status is CommandStatus.OK
    assert response.error is None
    assert response.data.project_root == str(project.resolve())
    assert response.data.embeddings_path == "embeddings/embeddings.jsonl"


def test_query_rejects_stale_chunk_content_after_indexing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from md_to_rag.chunk import _hash_text

    project = tmp_path / "project"
    _prepare_embedded_project(project)
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK
    chunks_path = project / "chunks" / "chunks.jsonl"
    rows = _jsonl(chunks_path)
    rows[0]["content"] = "Stale forged content after indexing."
    rows[0]["content_hash"] = _hash_text(rows[0]["content"])
    _write_jsonl(chunks_path, rows)
    monkeypatch.chdir(project)

    response = api.query("stale forged")

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "index_chunk_mismatch"


def test_index_rejects_stale_chunk_content_before_writing_index(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _prepare_embedded_project(project)
    chunks_path = project / "chunks" / "chunks.jsonl"
    rows = _jsonl(chunks_path)
    rows[0]["content"] = "Changed after embed with a stale content hash."
    _write_jsonl(chunks_path, rows)

    response = api.index(embeddings=project / "embeddings" / "embeddings.jsonl")

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "chunks_invalid_jsonl"
    assert not (project / "indexes" / "index_manifest.json").exists()


@pytest.mark.parametrize("field", ["metadata", "provenance"])
def test_index_rejects_chunk_metadata_or_provenance_drift_before_writing_index(
    tmp_path: Path,
    field: str,
) -> None:
    project = tmp_path / "project"
    _prepare_embedded_project(project)
    chunks_path = project / "chunks" / "chunks.jsonl"
    rows = _jsonl(chunks_path)
    rows[0][field] = {"changed": True}
    _write_jsonl(chunks_path, rows)

    response = api.index(embeddings=project / "embeddings" / "embeddings.jsonl")

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "embedding_chunk_mismatch"
    assert not (project / "indexes" / "index_manifest.json").exists()


@pytest.mark.parametrize("reserved_key", ["embedding_id", "embeddings_path"])
def test_index_rejects_reserved_index_provenance_keys_before_writing_index(
    tmp_path: Path,
    reserved_key: str,
) -> None:
    from md_to_rag.chunk import _hash_text

    project = tmp_path / "project"
    api.init(project)
    content = "# Reserved\n\nAlpha reserved provenance.\n"
    documents_path = project / "documents" / "custom.jsonl"
    _write_jsonl(
        documents_path,
        [
            {
                "schema_name": "md_to_rag.document",
                "schema_version": "1.0",
                "doc_id": "doc_reserved",
                "source_id": "src_reserved",
                "source_path": "source/reserved.md",
                "source_hash": "sha256:" + "1" * 64,
                "content_hash": _hash_text(content),
                "content": content,
                "line_count": len(content.splitlines()),
                "metadata": {},
                "provenance": {reserved_key: "upstream"},
            }
        ],
    )
    assert api.chunk(manifest=documents_path).status is CommandStatus.OK
    assert api.embed(chunks=project / "chunks" / "chunks.jsonl").status is CommandStatus.OK

    response = api.index(embeddings=project / "embeddings" / "embeddings.jsonl")

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "embedding_chunk_mismatch"
    assert not (project / "indexes" / "index_manifest.json").exists()


def test_index_rejects_missing_recorded_chunks_before_writing_index(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _prepare_embedded_project(project)
    (project / "chunks" / "chunks.jsonl").write_text("", encoding="utf-8")

    response = api.index(embeddings=project / "embeddings" / "embeddings.jsonl")

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "embedding_chunk_mismatch"
    assert not (project / "indexes" / "index_manifest.json").exists()


def test_index_rejects_embedding_rows_without_chunk_provenance(tmp_path: Path) -> None:
    project = tmp_path / "project"
    embeddings_path = _prepare_embedded_project(project)
    rows = _jsonl(embeddings_path)
    for row in rows:
        row["provenance"].pop("chunks_path", None)
    _write_jsonl(embeddings_path, rows)

    response = api.index(embeddings=embeddings_path)

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "embedding_schema_invalid"
    assert not (project / "indexes" / "index_manifest.json").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chunk_id", "forged_chunk"),
        ("chunk_content_hash", "sha256:" + "f" * 64),
        ("profile_hash", "sha256:" + "e" * 64),
    ],
)
def test_index_rejects_forged_embedding_provenance_identity_fields(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    project = tmp_path / "project"
    embeddings_path = _prepare_embedded_project(project)
    rows = _jsonl(embeddings_path)
    rows[0]["provenance"][field] = value
    _write_jsonl(embeddings_path, rows)

    response = api.index(embeddings=embeddings_path)

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "embedding_schema_invalid"
    assert not (project / "indexes" / "index_manifest.json").exists()


def test_index_recovers_empty_embedding_chunks_path_from_manifest(tmp_path: Path) -> None:
    project = tmp_path / "project"
    api.init(project)
    assert api.ingest(source=project / "source").status is CommandStatus.OK
    assert api.chunk(manifest=project / "documents" / "documents.jsonl").status is CommandStatus.OK
    assert api.embed(chunks=project / "chunks" / "chunks.jsonl").status is CommandStatus.OK

    response = api.index(embeddings=project / "embeddings" / "embeddings.jsonl")

    assert response.status is CommandStatus.OK
    assert response.data.embedding_count == 0
    assert response.data.chunks_path == "chunks/chunks.jsonl"
    assert response.data.dimensions == 8
    assert response.data.profile == {
        "dimensions": 8,
        "model": "deterministic-hash-v1",
        "provider": "md_to_rag.local_hash",
        "version": "1.0",
    }
    index_manifest = json.loads(
        (project / "indexes" / "index_manifest.json").read_text(encoding="utf-8")
    )
    assert index_manifest["chunks_path"] == "chunks/chunks.jsonl"
    assert index_manifest["dimensions"] == 8
    assert index_manifest["profile"] == response.data.profile


def test_index_rejects_empty_embeddings_when_recorded_chunks_later_change(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    assert api.ingest(source=project / "source").status is CommandStatus.OK
    assert api.chunk(manifest=project / "documents" / "documents.jsonl").status is CommandStatus.OK
    assert api.embed(chunks=project / "chunks" / "chunks.jsonl").status is CommandStatus.OK
    (project / "source" / "later.md").write_text("# Later\n\nNew chunk content.\n", encoding="utf-8")
    assert api.ingest(source=project / "source").status is CommandStatus.OK
    assert api.chunk(manifest=project / "documents" / "documents.jsonl").status is CommandStatus.OK

    response = api.index(embeddings=project / "embeddings" / "embeddings.jsonl")

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "embedding_chunk_mismatch"
    assert not (project / "indexes" / "index_manifest.json").exists()


def test_index_rejects_truncated_empty_artifacts_against_nonempty_embed_status(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    embeddings_path = _prepare_embedded_project(project)
    embeddings_path.write_text("", encoding="utf-8")
    (project / "chunks" / "chunks.jsonl").write_text("", encoding="utf-8")

    response = api.index(embeddings=embeddings_path)

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "embedding_chunk_mismatch"
    assert not (project / "indexes" / "index_manifest.json").exists()


def test_index_rejects_non_utf8_chunk_ids_before_mismatch_messages(tmp_path: Path) -> None:
    from md_to_rag.chunk import _hash_text

    project = tmp_path / "project"
    _prepare_embedded_project(project)
    chunks_path = project / "chunks" / "chunks.jsonl"
    rows = _jsonl(chunks_path)
    extra_content = "Extra chunk with a malformed id."
    rows.append(
        rows[0]
        | {
            "chunk_id": "bad\ud800",
            "chunk_index": max(row["chunk_index"] for row in rows) + 1,
            "content": extra_content,
            "content_hash": _hash_text(extra_content),
        }
    )
    _write_jsonl(chunks_path, rows)

    response = api.index(embeddings=project / "embeddings" / "embeddings.jsonl")

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "chunks_invalid_jsonl"
    response.model_dump_json()
    assert not (project / "indexes" / "index_manifest.json").exists()


def test_index_reports_chunk_error_for_nonportable_chunk_source_path(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _prepare_embedded_project(project)
    chunks_path = project / "chunks" / "chunks.jsonl"
    rows = _jsonl(chunks_path)
    rows[0]["source_path"] = "../source/alpha.md"
    _write_jsonl(chunks_path, rows)

    response = api.index(embeddings=project / "embeddings" / "embeddings.jsonl")

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "chunks_invalid_jsonl"
    assert "Chunks artifact" in response.error.message
    assert not (project / "indexes" / "index_manifest.json").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("content", "bad\ud800"),
        ("heading_path", ["bad\ud800"]),
    ],
)
def test_index_rejects_non_utf8_chunk_citation_inputs_before_json_serialization(
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    project = tmp_path / "project"
    _prepare_embedded_project(project)
    chunks_path = project / "chunks" / "chunks.jsonl"
    rows = _jsonl(chunks_path)
    rows[0][field] = value
    _write_jsonl(chunks_path, rows)

    response = api.index(embeddings=project / "embeddings" / "embeddings.jsonl")

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "chunks_invalid_jsonl"
    response.model_dump_json()


def test_query_rejects_stale_chunk_citations_after_indexing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    _prepare_embedded_project(project)
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK
    chunks_path = project / "chunks" / "chunks.jsonl"
    rows = _jsonl(chunks_path)
    for row in rows:
        if "Alpha retrieval" in row["content"]:
            row["line_start"] = 999
            row["line_end"] = 999
            row["heading_path"] = ["Forged"]
            break
    _write_jsonl(chunks_path, rows)
    monkeypatch.chdir(project)

    response = api.query("alpha retrieval")

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "index_chunk_mismatch"


@pytest.mark.parametrize("field", ["metadata", "provenance"])
def test_query_rejects_chunk_metadata_or_provenance_drift_after_indexing(
    tmp_path: Path,
    monkeypatch,
    field: str,
) -> None:
    project = tmp_path / "project"
    _prepare_embedded_project(project)
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK
    chunks_path = project / "chunks" / "chunks.jsonl"
    rows = _jsonl(chunks_path)
    rows[0][field] = {"changed": True}
    _write_jsonl(chunks_path, rows)
    monkeypatch.chdir(project)

    response = api.query("alpha retrieval")

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "index_chunk_mismatch"


def test_query_rejects_nonempty_index_without_chunks_path_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from md_to_rag import index as index_module

    project = tmp_path / "project"
    _prepare_embedded_project(project)
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK
    embeddings_path = project / "embeddings" / "embeddings.jsonl"
    embedding_rows = _jsonl(embeddings_path)
    for row in embedding_rows:
        row["provenance"].pop("chunks_path", None)
    _write_jsonl(embeddings_path, embedding_rows)
    index_path = project / "indexes" / "vectors.jsonl"
    index_rows = _jsonl(index_path)
    for row in index_rows:
        row["provenance"].pop("chunks_path", None)
    _write_jsonl(index_path, index_rows)
    index_manifest_path = project / "indexes" / "index_manifest.json"
    index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    index_manifest.pop("chunks_path", None)
    index_manifest["embeddings_hash"] = index_module._hash_bytes(embeddings_path.read_bytes())
    index_manifest["index_hash"] = index_module._hash_bytes(index_path.read_bytes())
    index_manifest_path.write_text(
        json.dumps(index_manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    response = api.query("alpha retrieval")

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "embedding_schema_invalid"


def test_query_rejects_missing_index_citations_for_recorded_chunks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from md_to_rag import index as index_module

    project = tmp_path / "project"
    _prepare_embedded_project(project)
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK
    index_path = project / "indexes" / "vectors.jsonl"
    index_rows = _jsonl(index_path)
    for row in index_rows:
        row.pop("line_start", None)
        row.pop("line_end", None)
        row.pop("heading_path", None)
    _write_jsonl(index_path, index_rows)
    index_manifest_path = project / "indexes" / "index_manifest.json"
    index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    index_manifest["index_hash"] = index_module._hash_bytes(index_path.read_bytes())
    index_manifest_path.write_text(
        json.dumps(index_manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    response = api.query("alpha retrieval")

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "index_schema_invalid"


def test_query_rejects_missing_indexed_chunks_after_truncation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    _prepare_embedded_project(project)
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK
    (project / "chunks" / "chunks.jsonl").write_text("", encoding="utf-8")
    monkeypatch.chdir(project)

    response = api.query("alpha retrieval")

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "index_chunk_mismatch"


def test_query_rejects_unindexed_extra_chunks_after_indexing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from md_to_rag.chunk import _hash_text

    project = tmp_path / "project"
    _prepare_embedded_project(project)
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK
    chunks_path = project / "chunks" / "chunks.jsonl"
    rows = _jsonl(chunks_path)
    extra_content = "Unindexed new chunk content."
    rows.append(
        rows[0]
        | {
            "chunk_id": "chk_unindexed_extra",
            "chunk_index": max(row["chunk_index"] for row in rows) + 1,
            "content": extra_content,
            "content_hash": _hash_text(extra_content),
            "line_start": 999,
            "line_end": 999,
        }
    )
    _write_jsonl(chunks_path, rows)
    monkeypatch.chdir(project)

    response = api.query("unindexed new")

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "index_chunk_mismatch"


@pytest.mark.parametrize(
    ("line_start", "line_end"),
    [
        (0, 1),
        (3, 2),
    ],
)
def test_query_rejects_invalid_chunk_line_ranges_after_indexing(
    tmp_path: Path,
    monkeypatch,
    line_start: int,
    line_end: int,
) -> None:
    project = tmp_path / "project"
    _prepare_embedded_project(project)
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK
    chunks_path = project / "chunks" / "chunks.jsonl"
    rows = _jsonl(chunks_path)
    rows[0]["line_start"] = line_start
    rows[0]["line_end"] = line_end
    _write_jsonl(chunks_path, rows)
    monkeypatch.chdir(project)

    response = api.query("alpha retrieval")

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "chunks_invalid_jsonl"


def test_query_reports_chunk_error_for_nonportable_chunk_source_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    _prepare_embedded_project(project)
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK
    chunks_path = project / "chunks" / "chunks.jsonl"
    rows = _jsonl(chunks_path)
    rows[0]["source_path"] = "../source/alpha.md"
    _write_jsonl(chunks_path, rows)
    monkeypatch.chdir(project)

    response = api.query("alpha retrieval")

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "chunks_invalid_jsonl"


def test_query_reports_missing_artifact_when_indexed_embeddings_are_deleted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    _prepare_embedded_project(project)
    embeddings_path = project / "embeddings" / "embeddings.jsonl"
    assert api.index(embeddings=embeddings_path).status is CommandStatus.OK
    embeddings_path.unlink()
    monkeypatch.chdir(project)

    response = api.query("alpha retrieval")

    assert response.status is CommandStatus.MISSING_ARTIFACT
    assert response.error is not None
    assert response.error.code == "embeddings_not_found"


def test_query_rejects_non_utf8_metadata_before_json_serialization(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from md_to_rag import index as index_module

    project = tmp_path / "project"
    _prepare_embedded_project(project)
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK
    index_path = project / "indexes" / "vectors.jsonl"
    index_rows = _jsonl(index_path)
    index_rows[0]["metadata"]["bad"] = "\ud800"
    _write_jsonl(index_path, index_rows)
    index_manifest_path = project / "indexes" / "index_manifest.json"
    index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    index_manifest["index_hash"] = index_module._hash_text(index_path.read_text(encoding="utf-8"))
    index_manifest_path.write_text(
        json.dumps(index_manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    response = api.query("needle")

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "index_schema_invalid"
    response.model_dump_json()


def test_index_rejects_non_utf8_metadata_before_writing_index(tmp_path: Path) -> None:
    project = tmp_path / "project"
    embeddings_path = _prepare_embedded_project(project)
    rows = _jsonl(embeddings_path)
    rows[0]["metadata"]["bad"] = "\ud800"
    _write_jsonl(embeddings_path, rows)

    response = api.index(embeddings=embeddings_path)

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "embedding_schema_invalid"
    response.model_dump_json()


def test_index_rejects_non_utf8_profile_before_json_serialization(tmp_path: Path) -> None:
    from md_to_rag import index as index_module

    project = tmp_path / "project"
    api.init(project)
    profile = {
        "provider": "bad\ud800",
        "model": "local",
        "dimensions": 1,
        "version": "1.0",
    }
    vector = [0.25]
    embedding_hash = index_module._hash_text(index_module._json_dumps_canonical(vector))
    profile_hash = index_module._hash_text(index_module._json_dumps_canonical(profile))
    chunk_content_hash = "sha256:" + "0" * 64
    row = {
        "schema_name": "md_to_rag.embedding",
        "schema_version": "1.0",
        "embedding_id": index_module._stable_embedding_id(
            "chk_bad_profile",
            chunk_content_hash,
            profile_hash,
            embedding_hash,
        ),
        "chunk_id": "chk_bad_profile",
        "doc_id": "doc_bad_profile",
        "source_id": "src_bad_profile",
        "source_path": "source/doc.md",
        "source_hash": "sha256:" + "1" * 64,
        "document_content_hash": "sha256:" + "2" * 64,
        "chunk_content_hash": chunk_content_hash,
        "chunk_index": 0,
        "embedding": vector,
        "embedding_hash": embedding_hash,
        "profile": profile,
        "metadata": {},
        "provenance": {},
    }
    _write_jsonl(project / "embeddings" / "embeddings.jsonl", [row])

    response = api.index(embeddings=project / "embeddings" / "embeddings.jsonl")

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "embedding_schema_invalid"
    response.model_dump_json()


def test_index_rejects_secret_profile_before_persisting_manifests(tmp_path: Path) -> None:
    from md_to_rag import index as index_module

    project = tmp_path / "project"
    api.init(project)
    profile = {
        "provider": "legacy",
        "model": "local",
        "dimensions": 1,
        "version": "1.0",
        "api_key": "sk-secret",
    }
    vector = [0.25]
    embedding_hash = index_module._hash_text(index_module._json_dumps_canonical(vector))
    profile_hash = index_module._hash_text(index_module._json_dumps_canonical(profile))
    chunk_content_hash = "sha256:" + "0" * 64
    row = {
        "schema_name": "md_to_rag.embedding",
        "schema_version": "1.0",
        "embedding_id": index_module._stable_embedding_id(
            "chk_secret_profile",
            chunk_content_hash,
            profile_hash,
            embedding_hash,
        ),
        "chunk_id": "chk_secret_profile",
        "doc_id": "doc_secret_profile",
        "source_id": "src_secret_profile",
        "source_path": "source/doc.md",
        "source_hash": "sha256:" + "1" * 64,
        "document_content_hash": "sha256:" + "2" * 64,
        "chunk_content_hash": chunk_content_hash,
        "chunk_index": 0,
        "embedding": vector,
        "embedding_hash": embedding_hash,
        "profile": profile,
        "metadata": {},
        "provenance": {},
    }
    _write_jsonl(project / "embeddings" / "embeddings.jsonl", [row])

    response = api.index(embeddings=project / "embeddings" / "embeddings.jsonl")

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "embedding_profile_invalid"
    assert not (project / "indexes" / "index_manifest.json").exists()
    assert "sk-secret" not in (project / "corpus_manifest.json").read_text(encoding="utf-8")
    assert "sk-secret" not in response.model_dump_json()


def test_index_rejects_non_utf8_chunks_path_before_json_serialization(tmp_path: Path) -> None:
    project = tmp_path / "project"
    embeddings_path = _prepare_embedded_project(project)
    rows = _jsonl(embeddings_path)
    for row in rows:
        row["provenance"]["chunks_path"] = "chunks/bad\ud800.jsonl"
    _write_jsonl(embeddings_path, rows)

    response = api.index(embeddings=embeddings_path)

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "embedding_schema_invalid"
    response.model_dump_json()


def test_index_rejects_non_utf8_embeddings_path_before_json_serialization(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    monkeypatch.chdir(project)

    response = api.index(embeddings="bad\ud800.jsonl")

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "embeddings_path_not_portable"
    response.model_dump_json()


def test_index_rejects_hard_linked_embeddings_input(tmp_path: Path) -> None:
    project = tmp_path / "project"
    embeddings_path = _prepare_embedded_project(project)
    outside_embeddings = tmp_path / "outside" / "embeddings.jsonl"
    outside_embeddings.parent.mkdir(parents=True, exist_ok=True)
    outside_embeddings.write_bytes(embeddings_path.read_bytes())
    embeddings_path.unlink()
    try:
        os.link(outside_embeddings, embeddings_path)
    except OSError as error:
        pytest.skip(f"hard link creation unavailable: {error}")

    response = api.index(embeddings=embeddings_path)

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "artifact_path_collision"
    assert not (project / "indexes" / "vectors.jsonl").exists()


def test_query_rejects_incompatible_chunk_schema_version_after_indexing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    _prepare_embedded_project(project)
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK
    chunks_path = project / "chunks" / "chunks.jsonl"
    rows = _jsonl(chunks_path)
    rows[0]["schema_version"] = "2.0"
    _write_jsonl(chunks_path, rows)
    monkeypatch.chdir(project)

    response = api.query("alpha retrieval")

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "chunks_invalid_jsonl"


def test_query_rejects_incompatible_index_version(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    _prepare_embedded_project(project)
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK
    index_manifest_path = project / "indexes" / "index_manifest.json"
    index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    index_manifest["index_version"] = "0.0"
    index_manifest_path.write_text(
        json.dumps(index_manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    response = api.query("alpha retrieval")

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "index_manifest_invalid"


def test_index_rejects_invalid_embedding_rows_and_nonportable_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from md_to_rag import index as index_module

    project = tmp_path / "project"
    embeddings_path = _prepare_embedded_project(project)
    rows = _jsonl(embeddings_path)

    bad_hash_project = tmp_path / "bad-hash"
    api.init(bad_hash_project)
    _write_jsonl(
        bad_hash_project / "embeddings" / "embeddings.jsonl",
        [rows[0] | {"embedding_hash": "sha256:not-the-vector-hash"}],
    )
    bad_hash_response = api.index(
        embeddings=bad_hash_project / "embeddings" / "embeddings.jsonl"
    )
    assert bad_hash_response.status is CommandStatus.ERROR
    assert bad_hash_response.error is not None
    assert bad_hash_response.error.code == "embedding_schema_invalid"
    assert not (bad_hash_project / "indexes" / "vectors.jsonl").exists()

    original_relative_to_project = index_module._relative_to_project

    def fake_relative_to_project(path: Path, project_root: Path):
        if path == embeddings_path.resolve():
            return "embeddings/CON.jsonl"
        return original_relative_to_project(path, project_root)

    monkeypatch.setattr(index_module, "_relative_to_project", fake_relative_to_project)
    nonportable_response = api.index(embeddings=embeddings_path)

    assert nonportable_response.status is CommandStatus.ERROR
    assert nonportable_response.error is not None
    assert nonportable_response.error.code == "embeddings_path_not_portable"
