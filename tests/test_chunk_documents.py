from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from md_to_rag import api
from md_to_rag.chunk import MAX_CHUNK_CHARS
from md_to_rag.cli import app
from md_to_rag.schemas import CommandStatus


runner = CliRunner()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _prepare_ingested_project(project: Path) -> Path:
    api.init(project)
    (project / "source" / "alpha.md").write_text(
        "# Alpha\n\nFirst paragraph.\nSecond line.\n\nSecond paragraph.\n",
        encoding="utf-8",
    )
    (project / "source" / "beta.md").write_text(
        "## Beta\n\nOnly block.\n",
        encoding="utf-8",
    )
    ingest_response = api.ingest(source=project / "source")
    assert ingest_response.status is CommandStatus.OK
    return project / "documents" / "documents.jsonl"


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


def test_chunk_defaults_to_current_project_documents_and_writes_portable_chunks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    documents_path = _prepare_ingested_project(project)
    document_rows = _jsonl(documents_path)
    monkeypatch.chdir(project / "source")

    response = api.chunk()

    assert response.__class__.__name__ == "ChunkResponse"
    assert response.status is CommandStatus.OK
    assert response.message == "Chunk artifacts generated."
    assert response.artifact_path == str((project / "chunks" / "chunks.jsonl").resolve())
    assert response.data.project_root == str(project.resolve())
    assert response.data.documents_path == "documents/documents.jsonl"
    assert response.data.chunks_path == "chunks/chunks.jsonl"
    assert response.data.changed is True
    assert response.data.document_count == 2
    assert response.data.chunk_count == 5

    chunk_rows = _jsonl(project / "chunks" / "chunks.jsonl")
    assert [row["content"] for row in chunk_rows] == [
        "# Alpha",
        "First paragraph.\nSecond line.",
        "Second paragraph.",
        "## Beta",
        "Only block.",
    ]
    assert [(row["line_start"], row["line_end"]) for row in chunk_rows] == [
        (1, 1),
        (3, 4),
        (6, 6),
        (1, 1),
        (3, 3),
    ]

    first_document = document_rows[0]
    first_chunk = chunk_rows[0]
    assert first_chunk["schema_name"] == "md_to_rag.chunk"
    assert first_chunk["schema_version"] == "1.0"
    assert first_chunk["chunk_id"].startswith("chk_")
    assert first_chunk["doc_id"] == first_document["doc_id"]
    assert first_chunk["source_id"] == first_document["source_id"]
    assert first_chunk["source_path"] == "source/alpha.md"
    assert first_chunk["source_hash"] == first_document["source_hash"]
    assert first_chunk["document_content_hash"] == first_document["content_hash"]
    assert first_chunk["content_hash"].startswith("sha256:")
    assert first_chunk["chunk_index"] == 0
    assert first_chunk["metadata"] == first_document["metadata"]
    assert first_chunk["provenance"] == first_document["provenance"]

    manifest = json.loads((project / "corpus_manifest.json").read_text(encoding="utf-8"))
    chunk_status = next(
        status for status in manifest["command_status"] if status["command"] == "chunk"
    )
    assert chunk_status["status"] == "ok"
    assert chunk_status["artifact_path"] == "chunks/chunks.jsonl"
    assert chunk_status["data"]["document_count"] == 2
    assert chunk_status["data"]["chunk_count"] == 5
    assert chunk_status["data"]["documents_path"] == "documents/documents.jsonl"
    assert chunk_status["data"]["chunks_path"] == "chunks/chunks.jsonl"
    assert chunk_status["data"]["chunks_hash"] == response.data.chunks_hash


def test_chunk_accepts_explicit_documents_manifest_relative_to_cwd(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    _prepare_ingested_project(project)
    monkeypatch.chdir(tmp_path)

    response = api.chunk(manifest=Path("project") / "documents" / "documents.jsonl")

    assert response.status is CommandStatus.OK
    assert response.data.project_root == str(project.resolve())
    assert response.data.documents_path == "documents/documents.jsonl"
    assert (project / "chunks" / "chunks.jsonl").is_file()


def test_chunk_rerun_is_idempotent_and_keeps_stable_chunk_ids(tmp_path: Path) -> None:
    project = tmp_path / "project"
    documents_path = _prepare_ingested_project(project)

    first = api.chunk(manifest=documents_path)
    chunks_path = project / "chunks" / "chunks.jsonl"
    first_rows = _jsonl(chunks_path)
    first_ids = [row["chunk_id"] for row in first_rows]
    first_bytes = chunks_path.read_bytes()
    first_mtime = chunks_path.stat().st_mtime_ns
    first_manifest_bytes = (project / "corpus_manifest.json").read_bytes()

    second = api.chunk(manifest=documents_path)

    assert first.status is CommandStatus.OK
    assert second.status is CommandStatus.OK
    assert second.message == "Chunk artifacts unchanged."
    assert second.data.changed is False
    assert [row["chunk_id"] for row in _jsonl(chunks_path)] == first_ids
    assert chunks_path.read_bytes() == first_bytes
    assert chunks_path.stat().st_mtime_ns == first_mtime
    assert (project / "corpus_manifest.json").read_bytes() == first_manifest_bytes


def test_chunk_reports_missing_documents_artifact_without_traceback(tmp_path: Path) -> None:
    project = tmp_path / "project"
    api.init(project)

    result = runner.invoke(
        app,
        ["chunk", "--manifest", str(project / "documents" / "documents.jsonl"), "--json"],
        prog_name="md-to-rag",
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["command"] == "chunk"
    assert payload["status"] == "missing_artifact"
    assert payload["error"]["code"] == "documents_not_found"
    assert "traceback" not in result.output.lower()


def test_chunk_rejects_invalid_jsonl_and_document_schema(tmp_path: Path) -> None:
    for case_name, artifact_text, expected_code in (
        ("invalid-jsonl", "{not json}\n", "documents_invalid_jsonl"),
        (
            "invalid-schema",
            json.dumps(
                {
                    "schema_name": "md_to_rag.source",
                    "schema_version": "1.0",
                    "doc_id": "doc_bad",
                    "source_id": "src_bad",
                    "source_path": "source/doc.md",
                    "source_hash": "sha256:bad",
                    "content_hash": "sha256:bad",
                    "content": "Bad",
                    "line_count": 1,
                    "metadata": {},
                    "provenance": {},
                },
                sort_keys=True,
            )
            + "\n",
            "document_schema_invalid",
        ),
    ):
        project = tmp_path / case_name
        api.init(project)
        documents_path = project / "documents" / "documents.jsonl"
        documents_path.write_text(artifact_text, encoding="utf-8")

        response = api.chunk(manifest=documents_path)

        assert response.status is CommandStatus.ERROR
        assert response.error is not None
        assert response.error.code == expected_code


def test_chunk_rejects_corrupt_document_row_integrity(tmp_path: Path) -> None:
    project = tmp_path / "project"
    documents_path = _prepare_ingested_project(project)
    rows = _jsonl(documents_path)
    cases = {
        "content-hash": {"content_hash": "sha256:not-the-content-hash"},
        "empty-doc-id": {"doc_id": ""},
        "negative-line-count": {"line_count": -1},
        "bool-line-count": {"line_count": True},
        "mismatched-line-count": {"line_count": 999},
    }

    for case_name, patch in cases.items():
        case_project = tmp_path / case_name
        api.init(case_project)
        case_documents_path = case_project / "documents" / "documents.jsonl"
        corrupt_row = rows[0] | patch
        case_documents_path.write_text(
            json.dumps(corrupt_row, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        response = api.chunk(manifest=case_documents_path)

        assert response.status is CommandStatus.ERROR
        assert response.error is not None
        assert response.error.code == "document_schema_invalid"
        assert not (case_project / "chunks" / "chunks.jsonl").exists()


def test_chunk_rejects_surrogate_document_content_without_traceback(tmp_path: Path) -> None:
    project = tmp_path / "project"
    api.init(project)
    documents_path = project / "documents" / "documents.jsonl"
    documents_path.write_text(
        (
            '{"schema_name":"md_to_rag.document","schema_version":"1.0",'
            '"doc_id":"doc_bad","source_id":"src_bad","source_path":"source/doc.md",'
            '"source_hash":"sha256:bad","content_hash":"sha256:bad",'
            '"content":"\\ud800","line_count":1,"metadata":{},"provenance":{}}\n'
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["chunk", "--manifest", str(documents_path), "--json"],
        prog_name="md-to-rag",
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "document_schema_invalid"
    assert "traceback" not in result.output.lower()


def test_chunk_accepts_empty_documents_emitted_by_ingest(tmp_path: Path) -> None:
    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "empty.md").write_text("", encoding="utf-8")
    ingest_response = api.ingest(source=project / "source")
    assert ingest_response.status is CommandStatus.OK

    response = api.chunk(manifest=project / "documents" / "documents.jsonl")

    assert response.status is CommandStatus.OK
    assert response.data.document_count == 1
    assert response.data.chunk_count == 0
    assert (project / "chunks" / "chunks.jsonl").read_text(encoding="utf-8") == ""


def test_chunk_rejects_nonportable_document_source_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    documents_path = _prepare_ingested_project(project)
    rows = _jsonl(documents_path)
    cases = ("../outside.md", r"source\a.md", "source/CON.md")

    for source_path in cases:
        case_project = tmp_path / f"case-{abs(hash(source_path))}"
        api.init(case_project)
        case_documents_path = case_project / "documents" / "documents.jsonl"
        corrupt_row = rows[0] | {"source_path": source_path}
        case_documents_path.write_text(
            json.dumps(corrupt_row, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        response = api.chunk(manifest=case_documents_path)

        assert response.status is CommandStatus.ERROR
        assert response.error is not None
        assert response.error.code == "document_schema_invalid"
        assert not (case_project / "chunks" / "chunks.jsonl").exists()


def test_chunk_rejects_nonportable_documents_artifact_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from md_to_rag import chunk as chunk_module

    project = tmp_path / "project"
    documents_path = _prepare_ingested_project(project)
    original_relative_to_project = chunk_module._relative_to_project

    def fake_relative_to_project(path: Path, project_root: Path):
        if path == documents_path.resolve():
            return "documents/CON.jsonl"
        return original_relative_to_project(path, project_root)

    monkeypatch.setattr(chunk_module, "_relative_to_project", fake_relative_to_project)

    response = api.chunk(manifest=documents_path)

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "documents_path_not_portable"
    assert not (project / "chunks" / "chunks.jsonl").exists()


def test_chunk_rejects_duplicate_document_ids(tmp_path: Path) -> None:
    project = tmp_path / "project"
    documents_path = _prepare_ingested_project(project)
    rows = _jsonl(documents_path)
    rows[1]["doc_id"] = rows[0]["doc_id"]
    documents_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    response = api.chunk(manifest=documents_path)

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "duplicate_document_id"
    assert not (project / "chunks" / "chunks.jsonl").exists()


def test_chunk_rejects_explicit_documents_artifact_outside_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    api.init(project)
    outside_documents = tmp_path / "documents.jsonl"
    outside_documents.write_text("", encoding="utf-8")

    response = api.chunk(manifest=outside_documents)

    assert response.status is CommandStatus.MISSING_ARTIFACT
    assert response.error is not None
    assert response.error.code == "manifest_not_found"


def test_chunk_rejects_linked_documents_artifact_outside_lexical_project(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    target = tmp_path / "target"
    api.init(parent)
    api.init(target)
    (target / "source" / "doc.md").write_text("# Target\n", encoding="utf-8")
    ingest_response = api.ingest(source=target / "source")
    assert ingest_response.status is CommandStatus.OK
    (parent / "documents").rmdir()
    _link_directory_or_skip(parent / "documents", target / "documents")

    response = api.chunk(manifest=parent / "documents" / "documents.jsonl")

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "documents_outside_project"
    assert not (parent / "chunks" / "chunks.jsonl").exists()
    assert not (target / "chunks" / "chunks.jsonl").exists()


def test_chunk_rejects_linked_documents_artifact_inside_nested_project(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    nested = parent / "nested"
    api.init(parent)
    api.init(nested)
    (nested / "source" / "doc.md").write_text("# Nested\n", encoding="utf-8")
    ingest_response = api.ingest(source=nested / "source")
    assert ingest_response.status is CommandStatus.OK
    linked_nested = parent / "linked-nested"
    _link_directory_or_skip(linked_nested, nested)

    response = api.chunk(manifest=linked_nested / "documents" / "documents.jsonl")

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "documents_nested_project"
    assert not (parent / "chunks" / "chunks.jsonl").exists()
    assert not (nested / "chunks" / "chunks.jsonl").exists()


def test_chunk_rejects_generated_chunks_artifact_as_input(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _prepare_ingested_project(project)
    chunks_path = project / "chunks" / "chunks.jsonl"
    chunks_path.write_text("do not overwrite\n", encoding="utf-8")

    response = api.chunk(manifest=chunks_path)

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "documents_artifact_collision"
    assert chunks_path.read_text(encoding="utf-8") == "do not overwrite\n"


def test_chunk_rows_include_markdown_heading_path(tmp_path: Path) -> None:
    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "sectioned.md").write_text(
        "# Top\n\nIntro.\n\n## Details\n\nDetail.\n\n### Deep\n\nDeep text.\n",
        encoding="utf-8",
    )
    ingest_response = api.ingest(source=project / "source")
    assert ingest_response.status is CommandStatus.OK

    response = api.chunk(manifest=project / "documents" / "documents.jsonl")

    assert response.status is CommandStatus.OK
    rows = _jsonl(project / "chunks" / "chunks.jsonl")
    assert [(row["content"], row["heading_path"]) for row in rows] == [
        ("# Top", ["Top"]),
        ("Intro.", ["Top"]),
        ("## Details", ["Top", "Details"]),
        ("Detail.", ["Top", "Details"]),
        ("### Deep", ["Top", "Details", "Deep"]),
        ("Deep text.", ["Top", "Details", "Deep"]),
    ]


def test_chunk_rows_include_setext_heading_path(tmp_path: Path) -> None:
    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "setext.md").write_text(
        "Title\n"
        "=====\n"
        "Intro.\n\n"
        "Subsection\n"
        "----------\n"
        "Detail.\n",
        encoding="utf-8",
    )
    ingest_response = api.ingest(source=project / "source")
    assert ingest_response.status is CommandStatus.OK

    response = api.chunk(manifest=project / "documents" / "documents.jsonl")

    assert response.status is CommandStatus.OK
    rows = _jsonl(project / "chunks" / "chunks.jsonl")
    assert [(row["content"], row["heading_path"]) for row in rows] == [
        ("Title\n=====", ["Title"]),
        ("Intro.", ["Title"]),
        ("Subsection\n----------", ["Title", "Subsection"]),
        ("Detail.", ["Title", "Subsection"]),
    ]
    assert [(row["line_start"], row["line_end"]) for row in rows] == [
        (1, 2),
        (3, 3),
        (5, 6),
        (7, 7),
    ]


def test_chunk_ignores_markdown_headings_inside_fenced_code_blocks(tmp_path: Path) -> None:
    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "fenced.md").write_text(
        "# Top\n\n"
        "```python\n"
        "# Not a heading\n"
        "print('hello')\n"
        "```\n\n"
        "After fence.\n",
        encoding="utf-8",
    )
    ingest_response = api.ingest(source=project / "source")
    assert ingest_response.status is CommandStatus.OK

    response = api.chunk(manifest=project / "documents" / "documents.jsonl")

    assert response.status is CommandStatus.OK
    rows = _jsonl(project / "chunks" / "chunks.jsonl")
    assert [(row["content"], row["heading_path"]) for row in rows] == [
        ("# Top", ["Top"]),
        ("```python\n# Not a heading\nprint('hello')\n```", ["Top"]),
        ("After fence.", ["Top"]),
    ]


def test_chunk_rejects_backticks_in_backtick_fence_info_strings(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "invalid-info.md").write_text(
        "```bad`info\n# Real\nBody\n",
        encoding="utf-8",
    )
    ingest_response = api.ingest(source=project / "source")
    assert ingest_response.status is CommandStatus.OK

    response = api.chunk(manifest=project / "documents" / "documents.jsonl")

    assert response.status is CommandStatus.OK
    rows = _jsonl(project / "chunks" / "chunks.jsonl")
    assert [(row["content"], row["heading_path"]) for row in rows] == [
        ("```bad`info", []),
        ("# Real", ["Real"]),
        ("Body", ["Real"]),
    ]


def test_chunk_preserves_blank_lines_inside_fenced_code_blocks(tmp_path: Path) -> None:
    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "code.md").write_text(
        "# Top\n\n"
        "```python\n"
        "print('before')\n"
        "\n"
        "print('after')\n"
        "```\n\n"
        "After.\n",
        encoding="utf-8",
    )
    ingest_response = api.ingest(source=project / "source")
    assert ingest_response.status is CommandStatus.OK

    response = api.chunk(manifest=project / "documents" / "documents.jsonl")

    assert response.status is CommandStatus.OK
    rows = _jsonl(project / "chunks" / "chunks.jsonl")
    assert [row["content"] for row in rows] == [
        "# Top",
        "```python\nprint('before')\n\nprint('after')\n```",
        "After.",
    ]
    assert [row["heading_path"] for row in rows] == [["Top"], ["Top"], ["Top"]]


def test_chunk_tracks_fence_lengths_for_nested_fence_examples(tmp_path: Path) -> None:
    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "long-fence.md").write_text(
        "# Top\n\n"
        "````markdown\n"
        "```python\n"
        "# Not a heading\n"
        "```\n"
        "````\n\n"
        "After.\n",
        encoding="utf-8",
    )
    ingest_response = api.ingest(source=project / "source")
    assert ingest_response.status is CommandStatus.OK

    response = api.chunk(manifest=project / "documents" / "documents.jsonl")

    assert response.status is CommandStatus.OK
    rows = _jsonl(project / "chunks" / "chunks.jsonl")
    assert [(row["content"], row["heading_path"]) for row in rows] == [
        ("# Top", ["Top"]),
        ("````markdown\n```python\n# Not a heading\n```\n````", ["Top"]),
        ("After.", ["Top"]),
    ]


def test_chunk_does_not_close_fences_on_info_string_lines(tmp_path: Path) -> None:
    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "inner-info.md").write_text(
        "```text\n"
        "```python\n"
        "# Not a heading\n"
        "```\n"
        "\n"
        "# Real\n"
        "body\n",
        encoding="utf-8",
    )
    ingest_response = api.ingest(source=project / "source")
    assert ingest_response.status is CommandStatus.OK

    response = api.chunk(manifest=project / "documents" / "documents.jsonl")

    assert response.status is CommandStatus.OK
    rows = _jsonl(project / "chunks" / "chunks.jsonl")
    assert [(row["content"], row["heading_path"]) for row in rows] == [
        ("```text\n```python\n# Not a heading\n```", []),
        ("# Real", ["Real"]),
        ("body", ["Real"]),
    ]


def test_chunk_does_not_open_fences_indented_as_code_blocks(tmp_path: Path) -> None:
    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "indented-fence.md").write_text(
        "    ```\n# Real Heading\nBody\n",
        encoding="utf-8",
    )
    ingest_response = api.ingest(source=project / "source")
    assert ingest_response.status is CommandStatus.OK

    response = api.chunk(manifest=project / "documents" / "documents.jsonl")

    assert response.status is CommandStatus.OK
    rows = _jsonl(project / "chunks" / "chunks.jsonl")
    assert [(row["content"], row["heading_path"]) for row in rows] == [
        ("    ```", []),
        ("# Real Heading", ["Real Heading"]),
        ("Body", ["Real Heading"]),
    ]


def test_chunk_splits_fenced_code_blocks_that_exceed_chunk_limit(tmp_path: Path) -> None:
    project = tmp_path / "project"
    api.init(project)
    long_line = "A" * (MAX_CHUNK_CHARS + 25)
    (project / "source" / "long-code.md").write_text(
        "```text\n" + long_line + "\n```\n",
        encoding="utf-8",
    )
    ingest_response = api.ingest(source=project / "source")
    assert ingest_response.status is CommandStatus.OK

    response = api.chunk(manifest=project / "documents" / "documents.jsonl")

    assert response.status is CommandStatus.OK
    rows = _jsonl(project / "chunks" / "chunks.jsonl")
    assert len(rows) == 2
    assert all(len(row["content"]) <= MAX_CHUNK_CHARS for row in rows)
    assert all(row["content"].startswith("```text\n") for row in rows)
    assert all(row["content"].endswith("\n```") for row in rows)
    assert [(row["line_start"], row["line_end"]) for row in rows] == [(1, 3), (1, 3)]
    inner_text = "".join(
        row["content"].removeprefix("```text\n").removesuffix("\n```")
        for row in rows
    )
    assert inner_text == long_line


def test_chunk_splits_unclosed_fenced_code_blocks_without_synthetic_closers(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    long_line = "A" * (MAX_CHUNK_CHARS + 25)
    (project / "source" / "unclosed-code.md").write_text(
        "```text\n" + long_line,
        encoding="utf-8",
    )
    ingest_response = api.ingest(source=project / "source")
    assert ingest_response.status is CommandStatus.OK

    response = api.chunk(manifest=project / "documents" / "documents.jsonl")

    assert response.status is CommandStatus.OK
    rows = _jsonl(project / "chunks" / "chunks.jsonl")
    assert [row["content"] for row in rows] == [
        "```text",
        long_line[:MAX_CHUNK_CHARS],
        long_line[MAX_CHUNK_CHARS:],
    ]
    assert all(not row["content"].endswith("\n```") for row in rows)
    assert [(row["line_start"], row["line_end"]) for row in rows] == [
        (1, 1),
        (2, 2),
        (2, 2),
    ]


def test_chunk_only_treats_valid_atx_headings_as_headings(tmp_path: Path) -> None:
    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "not-headings.md").write_text(
        "#tag\nnext\n    # not a heading\nstill text\n",
        encoding="utf-8",
    )
    ingest_response = api.ingest(source=project / "source")
    assert ingest_response.status is CommandStatus.OK

    response = api.chunk(manifest=project / "documents" / "documents.jsonl")

    assert response.status is CommandStatus.OK
    rows = _jsonl(project / "chunks" / "chunks.jsonl")
    assert [(row["content"], row["heading_path"]) for row in rows] == [
        ("#tag\nnext\n    # not a heading\nstill text", []),
    ]


def test_chunk_splits_adjacent_markdown_headings_into_block_boundaries(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "adjacent.md").write_text(
        "# Top\nIntro.\n## Details\nDetail.\n",
        encoding="utf-8",
    )
    ingest_response = api.ingest(source=project / "source")
    assert ingest_response.status is CommandStatus.OK

    response = api.chunk(manifest=project / "documents" / "documents.jsonl")

    assert response.status is CommandStatus.OK
    rows = _jsonl(project / "chunks" / "chunks.jsonl")
    assert [(row["content"], row["heading_path"]) for row in rows] == [
        ("# Top", ["Top"]),
        ("Intro.", ["Top"]),
        ("## Details", ["Top", "Details"]),
        ("Detail.", ["Top", "Details"]),
    ]


def test_chunk_splits_single_lines_that_exceed_chunk_limit(tmp_path: Path) -> None:
    project = tmp_path / "project"
    api.init(project)
    long_line = "A" * (MAX_CHUNK_CHARS + 25)
    (project / "source" / "long.md").write_text(long_line, encoding="utf-8")
    ingest_response = api.ingest(source=project / "source")
    assert ingest_response.status is CommandStatus.OK

    response = api.chunk(manifest=project / "documents" / "documents.jsonl")

    assert response.status is CommandStatus.OK
    rows = _jsonl(project / "chunks" / "chunks.jsonl")
    assert len(rows) == 2
    assert all(len(row["content"]) <= MAX_CHUNK_CHARS for row in rows)
    assert "".join(row["content"] for row in rows) == long_line
    assert [(row["line_start"], row["line_end"]) for row in rows] == [(1, 1), (1, 1)]


def test_index_query_report_missing_artifacts_before_index_exists() -> None:
    responses = [
        api.index(embeddings="embeddings/embeddings.jsonl"),
        api.query("What is indexed?"),
    ]

    for response in responses:
        assert response.status is CommandStatus.MISSING_ARTIFACT
        assert response.error is not None
        assert response.error.code == "manifest_not_found"
        assert response.__class__.__name__ in {"IndexResponse", "QueryResponse"}
        assert "raganything" not in response.model_dump_json().lower()
