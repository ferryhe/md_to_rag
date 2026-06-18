import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from md_to_rag import api
from md_to_rag.cli import app
from md_to_rag.schemas import CommandStatus


runner = CliRunner()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _sha256_text(text: str) -> str:
    return f"sha256:{sha256(text.encode('utf-8')).hexdigest()}"


def _assert_relative_path(value: str) -> None:
    assert value == Path(value).as_posix()
    assert not Path(value).is_absolute()
    assert ".." not in Path(value).parts


def test_ingest_markdown_source_directory_is_idempotent_and_portable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    source_dir = project / "source"
    intro_text = "# Intro\n\nHello from the corpus.\n"
    nested_text = "# Nested\n\nMore text.\n"
    (source_dir / "intro.md").write_text(intro_text, encoding="utf-8")
    (source_dir / "nested").mkdir()
    (source_dir / "nested" / "guide.markdown").write_text(nested_text, encoding="utf-8")
    monkeypatch.chdir(project)

    first = api.ingest()

    assert first.status is CommandStatus.OK
    assert first.message == "Ingest artifacts generated."
    assert first.data.changed is True
    assert first.data.document_count == 2
    assert first.data.source_manifest_path == "source/source_manifest.jsonl"
    assert first.data.documents_path == "documents/documents.jsonl"

    source_manifest_path = project / "source" / "source_manifest.jsonl"
    documents_path = project / "documents" / "documents.jsonl"
    first_source_bytes = source_manifest_path.read_bytes()
    first_document_bytes = documents_path.read_bytes()
    first_project_manifest_bytes = (project / "corpus_manifest.json").read_bytes()

    source_rows = _jsonl(source_manifest_path)
    document_rows = _jsonl(documents_path)
    assert [row["source_path"] for row in source_rows] == [
        "source/intro.md",
        "source/nested/guide.markdown",
    ]
    assert [row["source_path"] for row in document_rows] == [
        "source/intro.md",
        "source/nested/guide.markdown",
    ]
    assert len({row["doc_id"] for row in document_rows}) == 2
    assert all(row["doc_id"].startswith("doc_") for row in document_rows)

    intro_doc = document_rows[0]
    assert intro_doc["content"] == intro_text
    assert intro_doc["content_hash"] == _sha256_text(intro_text)
    assert intro_doc["source_hash"] == source_rows[0]["source_hash"]
    assert intro_doc["metadata"]["title"] == "Intro"
    assert intro_doc["provenance"] == {
        "kind": "markdown",
        "source_path": "source/intro.md",
    }
    for row in source_rows + document_rows:
        for key, value in row.items():
            if key.endswith("_path") and value is not None:
                _assert_relative_path(value)

    manifest = json.loads((project / "corpus_manifest.json").read_text(encoding="utf-8"))
    ingest_status = next(
        status for status in manifest["command_status"] if status["command"] == "ingest"
    )
    assert ingest_status["status"] == "ok"
    assert ingest_status["artifact_path"] == "documents/documents.jsonl"
    assert ingest_status["data"]["document_count"] == 2
    assert ingest_status["data"]["source_manifest_path"] == "source/source_manifest.jsonl"
    assert ingest_status["data"]["documents_path"] == "documents/documents.jsonl"

    second = api.ingest()

    assert second.status is CommandStatus.OK
    assert second.message == "Ingest artifacts unchanged."
    assert second.data.changed is False
    assert source_manifest_path.read_bytes() == first_source_bytes
    assert documents_path.read_bytes() == first_document_bytes
    assert (project / "corpus_manifest.json").read_bytes() == first_project_manifest_bytes
    assert _jsonl(documents_path)[0]["doc_id"] == intro_doc["doc_id"]


def test_ingest_repairs_stale_manifest_status_without_rewriting_artifacts(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "doc.md").write_text("# Doc\n\nStable content.\n", encoding="utf-8")
    first = api.ingest(source=project / "source")
    source_manifest_path = project / "source" / "source_manifest.jsonl"
    documents_path = project / "documents" / "documents.jsonl"
    source_manifest_bytes = source_manifest_path.read_bytes()
    documents_bytes = documents_path.read_bytes()

    manifest_path = project / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ingest_status = next(
        status for status in manifest["command_status"] if status["command"] == "ingest"
    )
    ingest_status["status"] = "not_implemented"
    ingest_status["data"]["documents_hash"] = "sha256:stale"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    repaired = api.ingest(source=project / "source")

    assert first.status is CommandStatus.OK
    assert repaired.status is CommandStatus.OK
    assert repaired.data.changed is True
    assert source_manifest_path.read_bytes() == source_manifest_bytes
    assert documents_path.read_bytes() == documents_bytes
    repaired_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repaired_ingest_status = next(
        status for status in repaired_manifest["command_status"] if status["command"] == "ingest"
    )
    assert repaired_ingest_status["status"] == "ok"
    assert repaired_ingest_status["data"]["documents_hash"] == first.data.documents_hash

    unchanged_manifest_bytes = manifest_path.read_bytes()
    unchanged = api.ingest(source=project / "source")

    assert unchanged.data.changed is False
    assert manifest_path.read_bytes() == unchanged_manifest_bytes


def test_ingest_doc_to_md_json_and_jsonl_manifests_inside_project(tmp_path: Path) -> None:
    for extension, payload_writer in {
        "jsonl": lambda path, rows: path.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
            encoding="utf-8",
        ),
        "json": lambda path, rows: path.write_text(
            json.dumps({"documents": rows}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        ),
    }.items():
        project = tmp_path / f"project-{extension}"
        api.init(project)
        markdown_path = project / "source" / "converted" / "report.md"
        markdown_path.parent.mkdir()
        markdown_text = "# Report\n\nConverted content.\n"
        markdown_path.write_text(markdown_text, encoding="utf-8")
        manifest_path = project / "source" / f"doc_to_md.{extension}"
        rows = [
            {
                "markdown_path": "source/converted/report.md",
                "source_path": "raw/report.pdf",
                "title": "Converted Report",
                "metadata": {"department": "risk"},
            }
        ]
        payload_writer(manifest_path, rows)

        response = api.ingest(source=manifest_path)

        assert response.status is CommandStatus.OK
        assert response.data.changed is True
        document_row = _jsonl(project / "documents" / "documents.jsonl")[0]
        source_row = _jsonl(project / "source" / "source_manifest.jsonl")[0]
        assert document_row["source_path"] == "source/converted/report.md"
        assert document_row["content"] == markdown_text
        assert document_row["content_hash"] == _sha256_text(markdown_text)
        assert document_row["metadata"]["title"] == "Converted Report"
        assert document_row["metadata"]["department"] == "risk"
        assert document_row["provenance"] == {
            "kind": "doc_to_md_manifest",
            "manifest_path": f"source/doc_to_md.{extension}",
            "manifest_row_index": 0,
            "source_path": "raw/report.pdf",
        }
        assert source_row["source_type"] == "doc_to_md_manifest"
        assert source_row["manifest_path"] == f"source/doc_to_md.{extension}"
        assert source_row["manifest_row_index"] == 0
        assert source_row["upstream_source_path"] == "raw/report.pdf"


def test_ingest_preserves_doc_to_md_metadata_title(tmp_path: Path) -> None:
    project = tmp_path / "project"
    api.init(project)
    markdown_path = project / "source" / "converted.md"
    markdown_path.write_text("# Markdown Heading\n", encoding="utf-8")
    manifest_path = project / "source" / "doc_to_md.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "markdown_path": "source/converted.md",
                "metadata": {"title": "Metadata Title", "department": "risk"},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    response = api.ingest(source=manifest_path)

    assert response.status is CommandStatus.OK
    document_row = _jsonl(project / "documents" / "documents.jsonl")[0]
    assert document_row["metadata"]["title"] == "Metadata Title"
    assert document_row["metadata"]["department"] == "risk"


def test_ingest_doc_to_md_rows_have_unique_document_ids(tmp_path: Path) -> None:
    project = tmp_path / "project"
    api.init(project)
    markdown_path = project / "source" / "converted.md"
    markdown_text = "# Converted\n\nSame markdown, different upstream rows.\n"
    markdown_path.write_text(markdown_text, encoding="utf-8")
    manifest_path = project / "source" / "doc_to_md.jsonl"
    manifest_path.write_text(
        "\n".join(
            json.dumps(row, sort_keys=True)
            for row in (
                {"markdown_path": "source/converted.md", "source_path": "raw/a.pdf"},
                {"markdown_path": "source/converted.md", "source_path": "raw/b.pdf"},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    response = api.ingest(source=manifest_path)

    assert response.status is CommandStatus.OK
    document_rows = _jsonl(project / "documents" / "documents.jsonl")
    source_rows = _jsonl(project / "source" / "source_manifest.jsonl")
    assert [row["source_path"] for row in document_rows] == [
        "source/converted.md",
        "source/converted.md",
    ]
    assert len({row["doc_id"] for row in document_rows}) == 2
    assert len({row["source_id"] for row in source_rows}) == 2
    assert {row["content_hash"] for row in document_rows} == {_sha256_text(markdown_text)}


def test_ingest_rejects_duplicate_doc_to_md_identities(tmp_path: Path) -> None:
    project = tmp_path / "project"
    api.init(project)
    markdown_path = project / "source" / "converted.md"
    markdown_path.write_text("# Converted\n", encoding="utf-8")
    manifest_path = project / "source" / "doc_to_md.jsonl"
    manifest_path.write_text(
        "\n".join(
            json.dumps(row, sort_keys=True)
            for row in (
                {
                    "markdown_path": "source/converted.md",
                    "source_path": "raw/report.pdf",
                    "title": "First title",
                },
                {
                    "markdown_path": "source/converted.md",
                    "source_path": "raw/report.pdf",
                    "title": "Second title",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    response = api.ingest(source=manifest_path)

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "duplicate_document_identity"


def test_ingest_preserves_doc_to_md_upstream_document_ids(tmp_path: Path) -> None:
    project = tmp_path / "project"
    api.init(project)
    markdown_path = project / "source" / "converted.md"
    markdown_path.write_text("# Converted\n", encoding="utf-8")
    manifest_path = project / "source" / "doc_to_md.jsonl"
    manifest_path.write_text(
        "\n".join(
            json.dumps(row, sort_keys=True)
            for row in (
                {
                    "markdown_path": "source/converted.md",
                    "source_path": "raw/report.pdf",
                    "document_id": "page-1",
                },
                {
                    "markdown_path": "source/converted.md",
                    "source_path": "raw/report.pdf",
                    "document_id": "page-2",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    response = api.ingest(source=manifest_path)

    assert response.status is CommandStatus.OK
    document_rows = _jsonl(project / "documents" / "documents.jsonl")
    source_rows = _jsonl(project / "source" / "source_manifest.jsonl")
    assert len({row["doc_id"] for row in document_rows}) == 2
    assert len({row["source_id"] for row in source_rows}) == 2
    assert len({row["source_hash"] for row in source_rows}) == 2
    assert {row["provenance"]["upstream_document_id"] for row in document_rows} == {
        "page-1",
        "page-2",
    }
    assert {row["upstream_document_id"] for row in source_rows} == {
        "page-1",
        "page-2",
    }


def test_ingest_preserves_doc_to_md_upstream_uri_provenance(tmp_path: Path) -> None:
    for upstream_uri in ("https://example.com/a.pdf", "file:///tmp/a.pdf"):
        project = tmp_path / f"project-{sha256(upstream_uri.encode()).hexdigest()[:8]}"
        api.init(project)
        markdown_path = project / "source" / "converted.md"
        markdown_path.write_text("# Converted\n", encoding="utf-8")
        manifest_path = project / "source" / "doc_to_md.jsonl"
        manifest_path.write_text(
            json.dumps(
                {
                    "markdown_path": "source/converted.md",
                    "source_path": upstream_uri,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        response = api.ingest(source=manifest_path)

        assert response.status is CommandStatus.OK
        document_row = _jsonl(project / "documents" / "documents.jsonl")[0]
        source_row = _jsonl(project / "source" / "source_manifest.jsonl")[0]
        assert document_row["provenance"]["source_path"] == upstream_uri
        assert source_row["upstream_source_path"] == upstream_uri


def test_ingest_doc_to_md_ids_survive_manifest_row_reordering(tmp_path: Path) -> None:
    project = tmp_path / "project"
    api.init(project)
    for name in ("a", "b"):
        (project / "source" / f"{name}.md").write_text(
            f"# {name.upper()}\n\nStable {name}.\n",
            encoding="utf-8",
        )
    manifest_path = project / "source" / "doc_to_md.jsonl"
    rows = [
        {"markdown_path": "source/a.md", "source_path": "raw/a.pdf"},
        {"markdown_path": "source/b.md", "source_path": "raw/b.pdf"},
    ]
    manifest_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    api.ingest(source=manifest_path)
    first_source_rows = {
        row["source_path"]: row for row in _jsonl(project / "source" / "source_manifest.jsonl")
    }
    first_document_rows = {
        row["source_path"]: row for row in _jsonl(project / "documents" / "documents.jsonl")
    }

    manifest_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in reversed(rows)) + "\n",
        encoding="utf-8",
    )
    response = api.ingest(source=manifest_path)

    assert response.status is CommandStatus.OK
    second_source_rows = {
        row["source_path"]: row for row in _jsonl(project / "source" / "source_manifest.jsonl")
    }
    second_document_rows = {
        row["source_path"]: row for row in _jsonl(project / "documents" / "documents.jsonl")
    }
    assert {
        path: row["source_id"] for path, row in second_source_rows.items()
    } == {
        path: row["source_id"] for path, row in first_source_rows.items()
    }
    assert {
        path: row["source_hash"] for path, row in second_source_rows.items()
    } == {
        path: row["source_hash"] for path, row in first_source_rows.items()
    }
    assert {
        path: row["doc_id"] for path, row in second_document_rows.items()
    } == {
        path: row["doc_id"] for path, row in first_document_rows.items()
    }


def test_ingest_missing_project_and_source_are_typed_json_errors(tmp_path: Path) -> None:
    plain_source = tmp_path / "plain"
    plain_source.mkdir()
    (plain_source / "doc.md").write_text("# Not initialized\n", encoding="utf-8")

    missing_manifest = runner.invoke(
        app,
        ["ingest", "--source", str(plain_source), "--json"],
        prog_name="md-to-rag",
    )
    assert missing_manifest.exit_code == 0
    missing_manifest_payload = json.loads(missing_manifest.output)
    assert missing_manifest_payload["status"] == "missing_artifact"
    assert missing_manifest_payload["error"]["code"] == "manifest_not_found"
    assert "traceback" not in missing_manifest.output.lower()

    project = tmp_path / "project"
    api.init(project)
    missing_source = runner.invoke(
        app,
        ["ingest", "--source", str(project / "source" / "missing.md"), "--json"],
        prog_name="md-to-rag",
    )
    assert missing_source.exit_code == 0
    missing_source_payload = json.loads(missing_source.output)
    assert missing_source_payload["status"] == "missing_artifact"
    assert missing_source_payload["error"]["code"] == "source_not_found"
    assert missing_source_payload["data"]["project_root"] == str(project.resolve())
    assert "traceback" not in missing_source.output.lower()


def test_ingest_rejects_nonportable_manifest_upstream_paths(tmp_path: Path) -> None:
    for upstream_path in (
        "/raw/report.pdf",
        "C:raw/report.pdf",
        "C:/raw/report.pdf",
        "C://raw/report.pdf",
        "../raw/report.pdf",
    ):
        project = tmp_path / f"project-{sha256(upstream_path.encode()).hexdigest()[:8]}"
        api.init(project)
        markdown_path = project / "source" / "converted.md"
        markdown_path.write_text("# Converted\n", encoding="utf-8")
        manifest_path = project / "source" / "doc_to_md.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "documents": [
                        {
                            "markdown_path": "source/converted.md",
                            "source_path": upstream_path,
                        }
                    ]
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        result = runner.invoke(
            app,
            ["ingest", "--source", str(manifest_path), "--json"],
            prog_name="md-to-rag",
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "error"
        assert payload["error"]["code"] == "manifest_path_not_portable"
        assert "traceback" not in result.output.lower()


def test_ingest_rejects_unsafe_manifest_markdown_targets(tmp_path: Path) -> None:
    unsafe_paths = {
        "/source/doc.md": "manifest_path_not_portable",
        "C:source/doc.md": "manifest_path_not_portable",
        "C:/source/doc.md": "manifest_path_not_portable",
        "source/../.env": "manifest_path_not_portable",
        "source/source_manifest.jsonl": "source_artifact_collision",
        "documents/documents.jsonl": "source_artifact_collision",
        "source/not-markdown.txt": "unsupported_source",
    }
    for markdown_path, expected_code in unsafe_paths.items():
        project = tmp_path / f"project-{sha256(markdown_path.encode()).hexdigest()[:8]}"
        api.init(project)
        (project / "source" / "not-markdown.txt").write_text("not markdown", encoding="utf-8")
        manifest_path = project / "source" / "doc_to_md.json"
        manifest_path.write_text(
            json.dumps(
                {"documents": [{"markdown_path": markdown_path}]},
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        result = runner.invoke(
            app,
            ["ingest", "--source", str(manifest_path), "--json"],
            prog_name="md-to-rag",
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "error"
        assert payload["error"]["code"] == expected_code
        assert "traceback" not in result.output.lower()


def test_ingest_directory_sorting_returns_typed_error_for_outside_resolved_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from md_to_rag import ingest as ingest_module

    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "good.md").write_text("# Good\n", encoding="utf-8")
    (project / "source" / "bad.md").write_text("# Bad\n", encoding="utf-8")
    original_relative_to_project = ingest_module._relative_to_project

    def fake_relative_to_project(path: Path, project_root: Path):
        if path.name == "bad.md":
            return ingest_module.IngestInputError(
                "source_outside_project",
                f"Ingest source must be inside the initialized project: {path}",
            )
        return original_relative_to_project(path, project_root)

    monkeypatch.setattr(ingest_module, "_relative_to_project", fake_relative_to_project)
    result = runner.invoke(
        app,
        ["ingest", "--source", str(project / "source"), "--json"],
        prog_name="md-to-rag",
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "source_outside_project"
    assert "traceback" not in result.output.lower()


def test_ingest_rejects_sources_that_collide_with_generated_artifacts(tmp_path: Path) -> None:
    for relative_source in (
        Path("source") / "source_manifest.jsonl",
        Path("documents") / "documents.jsonl",
    ):
        project = tmp_path / f"project-{relative_source.parent.name}"
        api.init(project)
        markdown_path = project / "source" / "doc.md"
        markdown_path.write_text("# Doc\n", encoding="utf-8")
        colliding_source = project / relative_source
        original_text = json.dumps(
            {"markdown_path": "source/doc.md"},
            sort_keys=True,
        ) + "\n"
        colliding_source.write_text(original_text, encoding="utf-8")

        result = runner.invoke(
            app,
            ["ingest", "--source", str(colliding_source), "--json"],
            prog_name="md-to-rag",
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "error"
        assert payload["error"]["code"] == "source_artifact_collision"
        assert colliding_source.read_text(encoding="utf-8") == original_text
        assert "traceback" not in result.output.lower()


def test_ingest_rejects_generated_artifact_directories_without_overwriting(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "doc.md").write_text("# Doc\n", encoding="utf-8")
    first = api.ingest(source=project / "source")
    documents_path = project / "documents" / "documents.jsonl"
    documents_bytes = documents_path.read_bytes()

    for relative_source in (
        Path("documents"),
        Path("chunks"),
        Path("embeddings"),
        Path("indexes"),
        Path("reports"),
    ):
        result = runner.invoke(
            app,
            ["ingest", "--source", str(project / relative_source), "--json"],
            prog_name="md-to-rag",
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "error"
        assert payload["error"]["code"] == "source_artifact_collision"
        assert documents_path.read_bytes() == documents_bytes
        assert first.data.document_count == 1
        assert "traceback" not in result.output.lower()


def test_ingest_rejects_generated_artifact_markdown_reached_indirectly(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "doc.md").write_text("# Doc\n", encoding="utf-8")
    first = api.ingest(source=project / "source")
    documents_path = project / "documents" / "documents.jsonl"
    documents_bytes = documents_path.read_bytes()
    generated_markdown = project / "documents" / "generated.md"
    generated_markdown.write_text("# Generated\n", encoding="utf-8")

    for source in (project, project / "source" / "doc_to_md.jsonl"):
        if source.suffix == ".jsonl":
            source.write_text(
                json.dumps(
                    {"markdown_path": "documents/generated.md"},
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        result = runner.invoke(
            app,
            ["ingest", "--source", str(source), "--json"],
            prog_name="md-to-rag",
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "error"
        assert payload["error"]["code"] == "source_artifact_collision"
        assert documents_path.read_bytes() == documents_bytes
        assert first.data.document_count == 1
        assert "traceback" not in result.output.lower()


def test_ingest_rewrites_corrupt_existing_artifacts_without_traceback(tmp_path: Path) -> None:
    for relative_artifact in (
        Path("source") / "source_manifest.jsonl",
        Path("documents") / "documents.jsonl",
    ):
        project = tmp_path / f"project-corrupt-{relative_artifact.parent.name}"
        api.init(project)
        (project / "source" / "doc.md").write_text("# Doc\n", encoding="utf-8")
        (project / relative_artifact).write_bytes(b"\xff\xfe\x00")

        result = runner.invoke(
            app,
            ["ingest", "--source", str(project / "source"), "--json"],
            prog_name="md-to-rag",
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "ok"
        assert payload["data"]["changed"] is True
        assert "traceback" not in result.output.lower()
        assert _jsonl(project / "source" / "source_manifest.jsonl")
        assert _jsonl(project / "documents" / "documents.jsonl")
