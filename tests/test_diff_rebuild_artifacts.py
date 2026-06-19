from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from md_to_rag import api
from md_to_rag.embed import DeterministicHashEmbeddingProvider, embed_project
from md_to_rag.cli import app
from md_to_rag.schemas import CommandName, CommandStatus


runner = CliRunner()


class CommonProfileProvider:
    def profile(self) -> dict[str, Any]:
        return {
            "provider": "example.custom",
            "model": "common-profile-v1",
            "dimensions": 3,
            "version": "1.0",
        }

    def embed(self, content: str, *, chunk_id: str, content_hash: str) -> list[float]:
        return [0.125, -0.25, 0.5]


class ExplicitEmptyOptionsLocalHashProvider(DeterministicHashEmbeddingProvider):
    def profile(self) -> dict[str, Any]:
        profile = super().profile()
        profile["options"] = {}
        return profile


def _stage_by_command(payload: dict[str, Any], command: str) -> dict[str, Any]:
    return next(stage for stage in payload["data"]["stages"] if stage["command"] == command)


def _step_by_command(payload: dict[str, Any], command: str) -> dict[str, Any]:
    return next(step for step in payload["data"]["steps"] if step["command"] == command)


def _prepare_indexed_project(project: Path) -> None:
    api.init(project)
    (project / "source" / "alpha.md").write_text(
        "# Alpha\n\nAlpha retrieval target.\n",
        encoding="utf-8",
    )
    assert api.ingest(source=project / "source").status is CommandStatus.OK
    assert api.chunk(manifest=project / "documents" / "documents.jsonl").status is CommandStatus.OK
    assert api.embed(chunks=project / "chunks" / "chunks.jsonl").status is CommandStatus.OK
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK


def test_diff_reports_fresh_chain_without_mutating_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    _prepare_indexed_project(project)
    manifest_path = project / "corpus_manifest.json"
    manifest_before = manifest_path.read_bytes()
    index_before = (project / "indexes" / "vectors.jsonl").read_bytes()
    monkeypatch.chdir(project)

    response = api.diff()

    assert response.status is CommandStatus.OK
    assert response.data.rebuild_needed is False
    assert response.data.stale_stages == []
    assert response.data.missing_stages == []
    assert response.data.error_stages == []
    assert [stage.status for stage in response.data.stages] == ["fresh"] * 4
    assert manifest_path.read_bytes() == manifest_before
    assert (project / "indexes" / "vectors.jsonl").read_bytes() == index_before


def test_diff_is_marked_available_in_inspect_status(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _prepare_indexed_project(project)

    diff_response = api.diff(project)
    inspect_response = api.inspect(project)

    diff_status = next(
        status
        for status in inspect_response.data.manifest.command_status
        if status.command.value == "diff"
    )
    assert diff_response.status is CommandStatus.OK
    assert diff_status.status is CommandStatus.OK
    assert diff_status.message == "Diff available."


def test_diff_reports_source_drift_and_downstream_staleness(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    _prepare_indexed_project(project)
    (project / "source" / "alpha.md").write_text(
        "# Alpha\n\nAlpha retrieval target changed.\n",
        encoding="utf-8",
    )
    documents_before = (project / "documents" / "documents.jsonl").read_bytes()
    monkeypatch.chdir(project)

    response = api.diff()

    assert response.status is CommandStatus.OK
    assert response.data.rebuild_needed is True
    assert [stage.value for stage in response.data.stale_stages] == [
        "ingest",
        "chunk",
        "embed",
        "index",
    ]
    assert response.data.missing_stages == []
    assert response.data.error_stages == []
    assert (project / "documents" / "documents.jsonl").read_bytes() == documents_before


def test_diff_reports_partially_rebuilt_downstream_as_stale(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    _prepare_indexed_project(project)
    (project / "source" / "alpha.md").write_text(
        "# Alpha\n\nChunk rerun drift target.\n",
        encoding="utf-8",
    )
    assert api.ingest(source=project / "source").status is CommandStatus.OK
    assert api.chunk(manifest=project / "documents" / "documents.jsonl").status is CommandStatus.OK

    response = api.diff(project)

    index_stage = next(
        stage for stage in response.data.stages if stage.command.value == "index"
    )
    assert response.status is CommandStatus.OK
    assert response.data.rebuild_needed is True
    assert CommandName.INDEX in response.data.stale_stages
    assert CommandName.INDEX not in response.data.error_stages
    assert index_stage.status == "stale"
    assert index_stage.error is None
    assert "An upstream stage requires rebuild." in index_stage.issues


def test_diff_replays_recorded_stage_input_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _prepare_indexed_project(project)
    custom = project / "custom"
    custom.mkdir()
    custom_documents = custom / "documents.jsonl"
    custom_chunks = custom / "chunks.jsonl"
    custom_embeddings = custom / "embeddings.jsonl"
    custom_documents.write_bytes((project / "documents" / "documents.jsonl").read_bytes())
    assert api.chunk(manifest=custom_documents).status is CommandStatus.OK
    custom_chunks.write_bytes((project / "chunks" / "chunks.jsonl").read_bytes())
    assert embed_project(custom_chunks).status is CommandStatus.OK
    custom_embeddings.write_bytes((project / "embeddings" / "embeddings.jsonl").read_bytes())
    assert api.index(embeddings=custom_embeddings).status is CommandStatus.OK

    response = api.diff(project)

    assert response.status is CommandStatus.OK
    assert response.data.rebuild_needed is False
    assert response.data.stale_stages == []
    assert response.data.error_stages == []
    assert [stage.status for stage in response.data.stages] == ["fresh"] * 4


def test_rebuild_rejects_nonportable_default_source_without_mutating_other_project(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    other = tmp_path / "other"
    api.init(project)
    api.init(other)
    (other / "source" / "other.md").write_text("# Other\n", encoding="utf-8")
    manifest_path = project / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_directories"]["source"] = str(other / "source")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    response = api.rebuild(project)

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "ingest_source_invalid"
    assert not (other / "documents" / "documents.jsonl").exists()


def test_rebuild_rejects_nested_project_source_without_mutating_child(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    child = project / "child"
    api.init(project)
    api.init(child)
    (child / "source" / "child.md").write_text("# Child\n", encoding="utf-8")
    manifest_path = project / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_directories"]["source"] = "child/source"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    response = api.rebuild(project)

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "source_nested_project"
    assert not (child / "documents" / "documents.jsonl").exists()


def test_diff_rejects_nested_project_source(tmp_path: Path) -> None:
    project = tmp_path / "project"
    child = project / "child"
    api.init(project)
    api.init(child)
    (child / "source" / "child.md").write_text("# Child\n", encoding="utf-8")
    manifest_path = project / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_directories"]["source"] = "child/source"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    response = api.diff(project)

    ingest_stage = next(
        stage for stage in response.data.stages if stage.command.value == "ingest"
    )
    assert response.status is CommandStatus.OK
    assert ingest_stage.status == "error"
    assert ingest_stage.error is not None
    assert ingest_stage.error.code == "source_nested_project"


def test_rebuild_legacy_fallback_does_not_hash_nested_project_source(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    child = project / "child"
    api.init(project)
    api.init(child)
    (child / "source" / "child.md").write_text("# Child\n", encoding="utf-8")
    manifest_path = project / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ingest_status = next(
        status for status in manifest["command_status"] if status["command"] == "ingest"
    )
    ingest_status["status"] = "ok"
    ingest_status["artifact_path"] = "documents/documents.jsonl"
    ingest_status["data"] = {
        "source_count": 1,
        "document_count": 1,
        "source_manifest_path": "source/source_manifest.jsonl",
        "documents_path": "documents/documents.jsonl",
        "source_manifest_hash": "sha256:not-a-real-hash",
        "documents_hash": "sha256:not-a-real-hash",
    }
    manifest["artifact_directories"]["source"] = "child/source"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    response = api.rebuild(project)

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "ingest_source_unavailable"
    assert not (child / "documents" / "documents.jsonl").exists()


def test_diff_legacy_fallback_rejects_nested_project_source(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    child = project / "child"
    api.init(project)
    api.init(child)
    (child / "source" / "child.md").write_text("# Child\n", encoding="utf-8")
    manifest_path = project / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ingest_status = next(
        status for status in manifest["command_status"] if status["command"] == "ingest"
    )
    ingest_status["status"] = "ok"
    ingest_status["artifact_path"] = "documents/documents.jsonl"
    ingest_status["data"] = {
        "source_count": 1,
        "document_count": 1,
        "source_manifest_path": "source/source_manifest.jsonl",
        "documents_path": "documents/documents.jsonl",
        "source_manifest_hash": "sha256:not-a-real-hash",
        "documents_hash": "sha256:not-a-real-hash",
    }
    manifest["artifact_directories"]["source"] = "child/source"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    response = api.diff(project)

    ingest_stage = next(
        stage for stage in response.data.stages if stage.command.value == "ingest"
    )
    assert ingest_stage.status == "error"
    assert ingest_stage.error is not None
    assert ingest_stage.error.code == "source_nested_project"


def test_diff_rejects_nonportable_default_source_without_reading_other_project(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    other = tmp_path / "other"
    api.init(project)
    api.init(other)
    (other / "source" / "other.md").write_text("# Other\n", encoding="utf-8")
    manifest_path = project / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_directories"]["source"] = str(other / "source")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    response = api.diff(project)

    ingest_stage = next(
        stage for stage in response.data.stages if stage.command.value == "ingest"
    )
    assert response.status is CommandStatus.OK
    assert response.data.rebuild_needed is True
    assert ingest_stage.status == "error"
    assert ingest_stage.error is not None
    assert ingest_stage.error.code == "ingest_source_invalid"
    assert not (other / "documents" / "documents.jsonl").exists()


def test_diff_and_rebuild_reject_rooted_default_source_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    api.init(project)
    manifest_path = project / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_directories"]["source"] = "/rooted/source"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    diff_response = api.diff(project)
    rebuild_response = api.rebuild(project)

    ingest_stage = next(
        stage for stage in diff_response.data.stages if stage.command.value == "ingest"
    )
    assert diff_response.status is CommandStatus.OK
    assert ingest_stage.status == "error"
    assert ingest_stage.error is not None
    assert ingest_stage.error.code == "ingest_source_invalid"
    assert rebuild_response.status is CommandStatus.ERROR
    assert rebuild_response.error is not None
    assert rebuild_response.error.code == "ingest_source_invalid"


def test_diff_and_rebuild_reject_backslash_default_source_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    api.init(project)
    manifest_path = project / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_directories"]["source"] = r"source\alpha.md"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    diff_response = api.diff(project)
    rebuild_response = api.rebuild(project)

    ingest_stage = next(
        stage for stage in diff_response.data.stages if stage.command.value == "ingest"
    )
    assert ingest_stage.error is not None
    assert ingest_stage.error.code == "ingest_source_invalid"
    assert rebuild_response.error is not None
    assert rebuild_response.error.code == "ingest_source_invalid"


def test_diff_rejects_rooted_recorded_stage_input_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _prepare_indexed_project(project)
    manifest_path = project / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunk_status = next(
        status for status in manifest["command_status"] if status["command"] == "chunk"
    )
    chunk_status["data"]["documents_path"] = "/rooted/documents.jsonl"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    response = api.diff(project)

    chunk_stage = next(
        stage for stage in response.data.stages if stage.command.value == "chunk"
    )
    assert response.status is CommandStatus.OK
    assert chunk_stage.status == "error"
    assert chunk_stage.error is not None
    assert chunk_stage.error.code == "stage_input_invalid"


def test_diff_rejects_backslash_recorded_stage_input_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _prepare_indexed_project(project)
    manifest_path = project / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunk_status = next(
        status for status in manifest["command_status"] if status["command"] == "chunk"
    )
    chunk_status["data"]["documents_path"] = r"custom\documents.jsonl"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    response = api.diff(project)

    chunk_stage = next(
        stage for stage in response.data.stages if stage.command.value == "chunk"
    )
    assert chunk_stage.error is not None
    assert chunk_stage.error.code == "stage_input_invalid"


def test_diff_and_rebuild_reject_non_utf8_manifest_source_paths(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    manifest_path = project / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_directories"]["source"] = "bad\ud800"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    diff_response = api.diff(project)
    rebuild_response = api.rebuild(project)

    ingest_stage = next(
        stage for stage in diff_response.data.stages if stage.command.value == "ingest"
    )
    assert ingest_stage.error is not None
    assert ingest_stage.error.code == "ingest_source_invalid"
    assert rebuild_response.error is not None
    assert rebuild_response.error.code == "ingest_source_invalid"
    diff_response.model_dump_json()
    rebuild_response.model_dump_json()


def test_diff_rejects_non_utf8_recorded_stage_input_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _prepare_indexed_project(project)
    manifest_path = project / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunk_status = next(
        status for status in manifest["command_status"] if status["command"] == "chunk"
    )
    chunk_status["data"]["documents_path"] = "bad\ud800.jsonl"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    response = api.diff(project)

    chunk_stage = next(
        stage for stage in response.data.stages if stage.command.value == "chunk"
    )
    assert chunk_stage.error is not None
    assert chunk_stage.error.code == "stage_input_invalid"
    response.model_dump_json()


def test_diff_rejects_generated_artifact_ingest_source(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _prepare_indexed_project(project)
    manifest_path = project / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ingest_status = next(
        status for status in manifest["command_status"] if status["command"] == "ingest"
    )
    ingest_status["data"]["source_path"] = "documents"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    response = api.diff(project)

    ingest_stage = next(
        stage for stage in response.data.stages if stage.command.value == "ingest"
    )
    assert ingest_stage.status == "error"
    assert ingest_stage.error is not None
    assert ingest_stage.error.code == "source_artifact_collision"


def test_diff_rejects_missing_artifact_under_linked_directory(tmp_path: Path) -> None:
    project = tmp_path / "project"
    api.init(project)
    outside_documents = tmp_path / "outside-documents"
    outside_documents.mkdir()
    documents_dir = project / "documents"
    documents_dir.rmdir()
    try:
        documents_dir.symlink_to(outside_documents, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")

    response = api.diff(project)

    chunk_stage = next(
        stage for stage in response.data.stages if stage.command.value == "chunk"
    )
    assert chunk_stage.status == "error"
    assert chunk_stage.error is not None
    assert chunk_stage.error.code in {
        "artifact_path_collision",
        "artifact_path_outside_project",
    }


def test_diff_rejects_hard_linked_documents_before_parsing(tmp_path: Path) -> None:
    project = tmp_path / "project"
    api.init(project)
    outside_documents = tmp_path / "outside-documents.jsonl"
    outside_documents.write_text("{not-json", encoding="utf-8")
    documents_path = project / "documents" / "documents.jsonl"
    try:
        os.link(outside_documents, documents_path)
    except OSError as error:
        pytest.skip(f"hard link creation unavailable: {error}")

    response = api.diff(project)

    assert response.status is CommandStatus.OK
    chunk_stage = next(
        stage for stage in response.data.stages if stage.command.value == "chunk"
    )
    assert chunk_stage.status == "error"
    assert chunk_stage.error is not None
    assert chunk_stage.error.code == "artifact_path_collision"


def test_diff_allows_fresh_custom_embedding_provider(tmp_path: Path) -> None:
    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "alpha.md").write_text(
        "# Alpha\n\nAlpha custom provider target.\n",
        encoding="utf-8",
    )
    assert api.ingest(source=project / "source").status is CommandStatus.OK
    assert api.chunk(manifest=project / "documents" / "documents.jsonl").status is CommandStatus.OK
    provider = DeterministicHashEmbeddingProvider(
        model="custom-hash-v1",
        dimensions=5,
        options={"salt": "custom"},
    )
    assert embed_project(project / "chunks" / "chunks.jsonl", provider=provider).status is CommandStatus.OK
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK

    response = api.diff(project)

    assert response.status is CommandStatus.OK
    assert response.data.rebuild_needed is False
    assert [stage.status for stage in response.data.stages] == ["fresh"] * 4


def test_diff_uses_recorded_local_profile_when_embeddings_are_missing(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "alpha.md").write_text(
        "# Alpha\n\nMissing custom embeddings target.\n",
        encoding="utf-8",
    )
    assert api.ingest(source=project / "source").status is CommandStatus.OK
    assert api.chunk(manifest=project / "documents" / "documents.jsonl").status is CommandStatus.OK
    provider = DeterministicHashEmbeddingProvider(
        model="custom-hash-v1",
        dimensions=5,
        options={"salt": "missing"},
    )
    assert embed_project(project / "chunks" / "chunks.jsonl", provider=provider).status is CommandStatus.OK
    manifest = json.loads((project / "corpus_manifest.json").read_text(encoding="utf-8"))
    embed_status = next(
        status for status in manifest["command_status"] if status["command"] == "embed"
    )
    recorded_embeddings_hash = embed_status["data"]["embeddings_hash"]
    (project / "embeddings" / "embeddings.jsonl").unlink()

    response = api.diff(project)

    embed_stage = next(
        stage for stage in response.data.stages if stage.command.value == "embed"
    )
    assert response.status is CommandStatus.OK
    assert embed_stage.status == "missing"
    assert embed_stage.expected_hashes["embeddings_hash"] == recorded_embeddings_hash


def test_diff_allows_fresh_non_local_common_profile_provider(tmp_path: Path) -> None:
    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "alpha.md").write_text(
        "# Alpha\n\nAlpha custom provider target.\n",
        encoding="utf-8",
    )
    assert api.ingest(source=project / "source").status is CommandStatus.OK
    assert api.chunk(manifest=project / "documents" / "documents.jsonl").status is CommandStatus.OK
    assert embed_project(
        project / "chunks" / "chunks.jsonl",
        provider=CommonProfileProvider(),
    ).status is CommandStatus.OK
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK

    response = api.diff(project)

    assert response.status is CommandStatus.OK
    assert response.data.rebuild_needed is False
    assert [stage.status for stage in response.data.stages] == ["fresh"] * 4


def test_diff_and_rebuild_replay_recorded_non_default_ingest_source(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    docs = project / "docs"
    docs.mkdir()
    (docs / "alpha.md").write_text(
        "# Alpha\n\nAlpha alternate source target.\n",
        encoding="utf-8",
    )
    assert api.ingest(source=docs).status is CommandStatus.OK
    assert api.chunk(manifest=project / "documents" / "documents.jsonl").status is CommandStatus.OK
    assert api.embed(chunks=project / "chunks" / "chunks.jsonl").status is CommandStatus.OK
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK

    diff_response = api.diff(project)
    rebuild_response = api.rebuild(project)

    assert diff_response.status is CommandStatus.OK
    assert diff_response.data.rebuild_needed is False
    assert rebuild_response.status is CommandStatus.OK
    assert rebuild_response.data.completed is True
    assert "Alpha alternate source target" in (
        project / "documents" / "documents.jsonl"
    ).read_text(encoding="utf-8")
    ingest_step = next(
        step for step in rebuild_response.data.steps if step.command.value == "ingest"
    )
    assert ingest_step.status == "ok"
    assert ingest_step.changed is False


def test_diff_and_rebuild_fallback_for_older_default_ingest_status(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    _prepare_indexed_project(project)
    manifest_path = project / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ingest_status = next(
        status for status in manifest["command_status"] if status["command"] == "ingest"
    )
    ingest_status["data"].pop("source_path", None)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    diff_response = api.diff(project)
    rebuild_response = api.rebuild(project)

    assert diff_response.status is CommandStatus.OK
    assert diff_response.data.rebuild_needed is False
    assert rebuild_response.status is CommandStatus.OK
    assert rebuild_response.data.completed is True
    assert "Alpha retrieval target" in (
        project / "documents" / "documents.jsonl"
    ).read_text(encoding="utf-8")


def test_diff_and_rebuild_replay_legacy_doc_to_md_manifest_source(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    converted = project / "converted"
    converted.mkdir()
    (converted / "report.md").write_text(
        "# Report\n\nConverted doc_to_md target.\n",
        encoding="utf-8",
    )
    doc_to_md_manifest = project / "source" / "doc_to_md.jsonl"
    doc_to_md_manifest.write_text(
        json.dumps(
            {
                "markdown_path": "converted/report.md",
                "source_path": "raw/report.pdf",
                "metadata": {"title": "Converted Report"},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    assert api.ingest(source=doc_to_md_manifest).status is CommandStatus.OK
    assert api.chunk(manifest=project / "documents" / "documents.jsonl").status is CommandStatus.OK
    assert api.embed(chunks=project / "chunks" / "chunks.jsonl").status is CommandStatus.OK
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK
    manifest_path = project / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ingest_status = next(
        status for status in manifest["command_status"] if status["command"] == "ingest"
    )
    ingest_status["data"].pop("source_path", None)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    documents_before = (project / "documents" / "documents.jsonl").read_text(
        encoding="utf-8"
    )
    document_before = json.loads(documents_before.splitlines()[0])

    diff_response = api.diff(project)
    rebuild_response = api.rebuild(project)
    documents_after = (project / "documents" / "documents.jsonl").read_text(
        encoding="utf-8"
    )
    document_after = json.loads(documents_after.splitlines()[0])

    ingest_stage = next(
        stage for stage in diff_response.data.stages if stage.command.value == "ingest"
    )
    assert diff_response.status is CommandStatus.OK
    assert ingest_stage.artifact_paths["source_path"] == "source/doc_to_md.jsonl"
    assert (
        ingest_stage.current_hashes["documents_hash"]
        == ingest_stage.expected_hashes["documents_hash"]
    )
    assert rebuild_response.status is CommandStatus.OK
    assert rebuild_response.data.completed is True
    assert documents_after == documents_before
    assert document_after["doc_id"] == document_before["doc_id"]
    assert document_after["source_id"] == document_before["source_id"]
    assert document_after["provenance"]["kind"] == "doc_to_md_manifest"
    assert document_after["provenance"]["manifest_path"] == "source/doc_to_md.jsonl"
    assert document_after["provenance"]["source_path"] == "raw/report.pdf"


def test_rebuild_rejects_older_unknown_non_default_ingest_source(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    docs = project / "docs"
    docs.mkdir()
    (docs / "alpha.md").write_text(
        "# Alpha\n\nOlder non-default source target.\n",
        encoding="utf-8",
    )
    assert api.ingest(source=docs).status is CommandStatus.OK
    assert api.chunk(manifest=project / "documents" / "documents.jsonl").status is CommandStatus.OK
    assert api.embed(chunks=project / "chunks" / "chunks.jsonl").status is CommandStatus.OK
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK
    manifest_path = project / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ingest_status = next(
        status for status in manifest["command_status"] if status["command"] == "ingest"
    )
    ingest_status["data"].pop("source_path", None)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    documents_before = (project / "documents" / "documents.jsonl").read_bytes()

    response = api.rebuild(project)

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "ingest_source_unavailable"
    assert response.data.stopped_at.value == "ingest"
    assert (project / "documents" / "documents.jsonl").read_bytes() == documents_before


def test_diff_reports_older_unknown_non_default_ingest_source(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    docs = project / "docs"
    docs.mkdir()
    (docs / "alpha.md").write_text(
        "# Alpha\n\nOlder non-default source target.\n",
        encoding="utf-8",
    )
    assert api.ingest(source=docs).status is CommandStatus.OK
    assert api.chunk(manifest=project / "documents" / "documents.jsonl").status is CommandStatus.OK
    assert api.embed(chunks=project / "chunks" / "chunks.jsonl").status is CommandStatus.OK
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK
    manifest_path = project / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ingest_status = next(
        status for status in manifest["command_status"] if status["command"] == "ingest"
    )
    ingest_status["data"].pop("source_path", None)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    response = api.diff(project)

    ingest_stage = next(
        stage for stage in response.data.stages if stage.command.value == "ingest"
    )
    assert response.status is CommandStatus.OK
    assert response.data.rebuild_needed is True
    assert ingest_stage.status == "error"
    assert ingest_stage.error is not None
    assert ingest_stage.error.code == "ingest_source_unavailable"


def test_rebuild_allows_older_default_source_after_source_edit(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    _prepare_indexed_project(project)
    manifest_path = project / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ingest_status = next(
        status for status in manifest["command_status"] if status["command"] == "ingest"
    )
    ingest_status["data"].pop("source_path", None)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (project / "source" / "alpha.md").write_text(
        "# Alpha\n\nEdited legacy default source.\n",
        encoding="utf-8",
    )

    response = api.rebuild(project)

    assert response.status is CommandStatus.OK
    assert response.data.completed is True
    assert "Edited legacy default source" in (
        project / "documents" / "documents.jsonl"
    ).read_text(encoding="utf-8")


def test_rebuild_replays_legacy_single_markdown_file_without_widening(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "alpha.md").write_text(
        "# Alpha\n\nLegacy file target.\n",
        encoding="utf-8",
    )
    assert api.ingest(source=project / "source" / "alpha.md").status is CommandStatus.OK
    assert api.chunk(manifest=project / "documents" / "documents.jsonl").status is CommandStatus.OK
    assert api.embed(chunks=project / "chunks" / "chunks.jsonl").status is CommandStatus.OK
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK
    manifest_path = project / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ingest_status = next(
        status for status in manifest["command_status"] if status["command"] == "ingest"
    )
    ingest_status["data"].pop("source_path", None)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (project / "source" / "sibling.md").write_text(
        "# Sibling\n\nMust stay out of scope.\n",
        encoding="utf-8",
    )

    response = api.rebuild(project)

    documents_text = (project / "documents" / "documents.jsonl").read_text(
        encoding="utf-8"
    )
    assert response.status is CommandStatus.OK
    assert "Legacy file target" in documents_text
    assert "Must stay out of scope" not in documents_text


def test_rebuild_rejects_drifted_legacy_source_manifest_rows(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "alpha.md").write_text("# Alpha\n", encoding="utf-8")
    (project / "source" / "beta.md").write_text("# Beta\n", encoding="utf-8")
    assert api.ingest(source=project / "source" / "alpha.md").status is CommandStatus.OK
    assert api.chunk(manifest=project / "documents" / "documents.jsonl").status is CommandStatus.OK
    assert api.embed(chunks=project / "chunks" / "chunks.jsonl").status is CommandStatus.OK
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK
    manifest_path = project / "corpus_manifest.json"
    legacy_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ingest_status = next(
        status
        for status in legacy_manifest["command_status"]
        if status["command"] == "ingest"
    )
    ingest_status["data"].pop("source_path", None)
    assert api.ingest(source=project / "source" / "beta.md").status is CommandStatus.OK
    manifest_path.write_text(
        json.dumps(legacy_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    response = api.rebuild(project)

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "ingest_source_unavailable"


def test_diff_and_rebuild_reject_linked_legacy_source_manifest_rows(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "alpha.md").write_text("# Alpha\n", encoding="utf-8")
    assert api.ingest(source=project / "source" / "alpha.md").status is CommandStatus.OK
    manifest_path = project / "corpus_manifest.json"
    legacy_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ingest_status = next(
        status
        for status in legacy_manifest["command_status"]
        if status["command"] == "ingest"
    )
    ingest_status["data"].pop("source_path", None)
    outside_manifest = tmp_path / "outside-source-manifest.jsonl"
    outside_manifest.write_text(
        (project / "source" / "source_manifest.jsonl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (project / "source" / "source_manifest.jsonl").unlink()
    try:
        os.link(outside_manifest, project / "source" / "source_manifest.jsonl")
    except OSError as error:
        pytest.skip(f"hard link creation unavailable: {error}")
    manifest_path.write_text(
        json.dumps(legacy_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    diff_response = api.diff(project)
    rebuild_response = api.rebuild(project)

    ingest_stage = next(
        stage for stage in diff_response.data.stages if stage.command.value == "ingest"
    )
    assert ingest_stage.status == "error"
    assert ingest_stage.error is not None
    assert ingest_stage.error.code == "artifact_path_collision"
    assert rebuild_response.status is CommandStatus.ERROR
    assert rebuild_response.error is not None
    assert rebuild_response.error.code == "artifact_path_collision"


def test_ingest_backfills_missing_source_path_for_non_default_source(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    docs = project / "docs"
    docs.mkdir()
    (docs / "alpha.md").write_text("# Alpha\n\nBackfill target.\n", encoding="utf-8")
    first = api.ingest(source=docs)
    manifest_path = project / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ingest_status = next(
        status for status in manifest["command_status"] if status["command"] == "ingest"
    )
    ingest_status["data"].pop("source_path", None)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    repaired = api.ingest(source=docs)

    assert first.status is CommandStatus.OK
    assert repaired.status is CommandStatus.OK
    assert repaired.data.changed is True
    repaired_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repaired_status = next(
        status for status in repaired_manifest["command_status"] if status["command"] == "ingest"
    )
    assert repaired_status["data"]["source_path"] == "docs"


def test_rebuild_preserves_recorded_local_hash_embedding_profile(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "alpha.md").write_text(
        "# Alpha\n\nAlpha custom profile target.\n",
        encoding="utf-8",
    )
    assert api.ingest(source=project / "source").status is CommandStatus.OK
    assert api.chunk(manifest=project / "documents" / "documents.jsonl").status is CommandStatus.OK
    provider = DeterministicHashEmbeddingProvider(
        model="custom-hash-v1",
        dimensions=5,
        options={"salt": "custom"},
    )
    assert embed_project(project / "chunks" / "chunks.jsonl", provider=provider).status is CommandStatus.OK
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK
    before_profile = json.loads(
        (project / "embeddings" / "embeddings.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )["profile"]

    response = api.rebuild(project)

    after_profile = json.loads(
        (project / "embeddings" / "embeddings.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )["profile"]
    embed_step = next(
        step for step in response.data.steps if step.command.value == "embed"
    )
    assert response.status is CommandStatus.OK
    assert embed_step.status == "ok"
    assert embed_step.changed is False
    assert after_profile == before_profile


def test_diff_and_rebuild_preserve_empty_local_hash_profile_options(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "alpha.md").write_text(
        "# Alpha\n\nExplicit empty options target.\n",
        encoding="utf-8",
    )
    assert api.ingest(source=project / "source").status is CommandStatus.OK
    assert api.chunk(manifest=project / "documents" / "documents.jsonl").status is CommandStatus.OK
    provider = ExplicitEmptyOptionsLocalHashProvider()
    assert embed_project(project / "chunks" / "chunks.jsonl", provider=provider).status is CommandStatus.OK
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK
    before_embeddings = (project / "embeddings" / "embeddings.jsonl").read_bytes()
    before_profile = json.loads(before_embeddings.decode("utf-8").splitlines()[0])[
        "profile"
    ]

    diff_response = api.diff(project)
    rebuild_response = api.rebuild(project)
    after_embeddings = (project / "embeddings" / "embeddings.jsonl").read_bytes()
    after_profile = json.loads(after_embeddings.decode("utf-8").splitlines()[0])[
        "profile"
    ]

    embed_stage = next(
        stage for stage in diff_response.data.stages if stage.command.value == "embed"
    )
    embed_step = next(
        step for step in rebuild_response.data.steps if step.command.value == "embed"
    )
    assert before_profile["options"] == {}
    assert diff_response.status is CommandStatus.OK
    assert embed_stage.status == "fresh"
    assert rebuild_response.status is CommandStatus.OK
    assert embed_step.changed is False
    assert after_embeddings == before_embeddings
    assert after_profile == before_profile


def test_rebuild_returns_typed_error_for_invalid_recorded_local_profile(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    _prepare_indexed_project(project)
    manifest_path = project / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    embed_status = next(
        status for status in manifest["command_status"] if status["command"] == "embed"
    )
    embed_status["data"]["profile"]["options"] = {"bad": float("nan")}
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    response = api.rebuild(project)

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "embedding_profile_invalid"
    assert response.data.stopped_at.value == "embed"


def test_rebuild_stops_for_non_local_embedding_profile_without_overwriting(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "alpha.md").write_text(
        "# Alpha\n\nAlpha external provider target.\n",
        encoding="utf-8",
    )
    assert api.ingest(source=project / "source").status is CommandStatus.OK
    assert api.chunk(manifest=project / "documents" / "documents.jsonl").status is CommandStatus.OK
    assert embed_project(
        project / "chunks" / "chunks.jsonl",
        provider=CommonProfileProvider(),
    ).status is CommandStatus.OK
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK
    before_embeddings = (project / "embeddings" / "embeddings.jsonl").read_bytes()

    response = api.rebuild(project)

    embed_step = next(
        step for step in response.data.steps if step.command.value == "embed"
    )
    index_step = next(
        step for step in response.data.steps if step.command.value == "index"
    )
    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "embedding_profile_unsupported"
    assert response.data.stopped_at.value == "embed"
    assert embed_step.error is not None
    assert embed_step.error.code == "embedding_profile_unsupported"
    assert index_step.skipped is True
    assert (project / "embeddings" / "embeddings.jsonl").read_bytes() == before_embeddings


def test_diff_json_reports_missing_artifacts_without_traceback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    monkeypatch.chdir(project)

    result = runner.invoke(app, ["diff", "--json"], prog_name="md-to-rag")

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["command"] == "diff"
    assert payload["status"] == "ok"
    assert payload["data"]["rebuild_needed"] is True
    assert payload["data"]["missing_stages"] == ["ingest", "chunk", "embed", "index"]
    assert _stage_by_command(payload, "ingest")["status"] == "missing"
    assert "traceback" not in result.output.lower()
    assert "raganything" not in result.output.lower()


def test_rebuild_runs_chain_updates_manifest_and_is_idempotent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "alpha.md").write_text(
        "# Alpha\n\nAlpha rebuild target.\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    first = api.rebuild()
    manifest_path = project / "corpus_manifest.json"
    manifest_after_first = manifest_path.read_bytes()
    index_after_first = (project / "indexes" / "vectors.jsonl").read_bytes()
    second = api.rebuild()

    assert first.status is CommandStatus.OK
    assert first.data.changed is True
    assert [step.status for step in first.data.steps] == ["ok"] * 4
    assert [step.skipped for step in first.data.steps] == [False] * 4
    manifest = json.loads(manifest_after_first)
    status_by_command = {
        status["command"]: status for status in manifest["command_status"]
    }
    for command in ("ingest", "chunk", "embed", "index", "rebuild"):
        assert status_by_command[command]["status"] == "ok"
    assert status_by_command["rebuild"]["artifact_path"] == "indexes/index_manifest.json"
    assert status_by_command["rebuild"]["data"]["completed_steps"] == [
        "ingest",
        "chunk",
        "embed",
        "index",
    ]

    inspect_response = api.inspect(project)
    rebuild_status = next(
        status
        for status in inspect_response.data.manifest.command_status
        if status.command.value == "rebuild"
    )
    assert rebuild_status.status is CommandStatus.OK
    assert rebuild_status.artifact_path == "indexes/index_manifest.json"

    assert second.status is CommandStatus.OK
    assert second.data.changed is False
    assert [step.changed for step in second.data.steps] == [False] * 4
    assert manifest_path.read_bytes() == manifest_after_first
    assert (project / "indexes" / "vectors.jsonl").read_bytes() == index_after_first


def test_rebuild_json_stops_on_missing_source_and_skips_later_steps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    (project / "source").rmdir()
    monkeypatch.chdir(project)

    result = runner.invoke(app, ["rebuild", "--json"], prog_name="md-to-rag")

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["command"] == "rebuild"
    assert payload["status"] == "missing_artifact"
    assert payload["error"]["code"] == "source_not_found"
    assert payload["data"]["completed"] is False
    assert payload["data"]["changed"] is False
    assert _step_by_command(payload, "ingest")["status"] == "missing_artifact"
    assert _step_by_command(payload, "chunk")["status"] == "skipped"
    assert _step_by_command(payload, "index")["skipped"] is True
    assert not (project / "documents" / "documents.jsonl").exists()
    assert "traceback" not in result.output.lower()


def test_rebuild_rejects_missing_explicit_project_without_mutating_parent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "alpha.md").write_text("# Alpha\n", encoding="utf-8")
    monkeypatch.chdir(project)

    response = api.rebuild("typo")

    assert response.status is CommandStatus.MISSING_ARTIFACT
    assert response.error is not None
    assert response.error.code == "project_not_found"
    assert response.data.project_path == str(project / "typo")
    assert not (project / "documents" / "documents.jsonl").exists()


def test_rebuild_rejects_existing_non_project_path_without_mutating_parent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "alpha.md").write_text("# Alpha\n", encoding="utf-8")
    monkeypatch.chdir(project)

    directory_response = api.rebuild(project / "source")
    file_response = api.rebuild(project / "source" / "alpha.md")

    assert directory_response.status is CommandStatus.ERROR
    assert directory_response.error is not None
    assert directory_response.error.code == "project_path_not_project"
    assert file_response.status is CommandStatus.ERROR
    assert file_response.error is not None
    assert file_response.error.code == "project_path_not_project"
    assert not (project / "documents" / "documents.jsonl").exists()


def test_rebuild_refreshes_status_after_changed_rerun(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from md_to_rag import rebuild as rebuild_module

    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "alpha.md").write_text(
        "# Alpha\n\nFirst content.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        rebuild_module,
        "_utc_now",
        lambda: "2026-06-18T00:00:00Z",
    )
    first = api.rebuild(project)
    first_manifest = json.loads((project / "corpus_manifest.json").read_text())
    first_rebuild_status = next(
        status for status in first_manifest["command_status"] if status["command"] == "rebuild"
    )

    (project / "source" / "alpha.md").write_text(
        "# Alpha\n\nChanged content.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        rebuild_module,
        "_utc_now",
        lambda: "2026-06-18T00:00:10Z",
    )
    second = api.rebuild(project)
    second_manifest = json.loads((project / "corpus_manifest.json").read_text())
    second_rebuild_status = next(
        status for status in second_manifest["command_status"] if status["command"] == "rebuild"
    )

    assert first.status is CommandStatus.OK
    assert second.status is CommandStatus.OK
    assert second.data.changed is True
    assert [step.changed for step in second.data.steps] == [True] * 4
    assert first_rebuild_status["updated_at"] == "2026-06-18T00:00:00Z"
    assert second_rebuild_status["updated_at"] == "2026-06-18T00:00:10Z"


def test_diff_rebuild_reject_non_utf8_project_path_before_json_serialization() -> None:
    diff_response = api.diff("bad\ud800")
    rebuild_response = api.rebuild("bad\ud800")

    assert diff_response.status is CommandStatus.ERROR
    assert diff_response.error is not None
    assert diff_response.error.code == "project_path_not_portable"
    assert diff_response.data.project_path is None
    diff_response.model_dump_json()

    assert rebuild_response.status is CommandStatus.ERROR
    assert rebuild_response.error is not None
    assert rebuild_response.error.code == "project_path_not_portable"
    assert rebuild_response.data.project_path is None
    rebuild_response.model_dump_json()
