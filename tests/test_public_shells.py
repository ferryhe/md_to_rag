import json
import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from md_to_rag import __version__, api, mcp
from md_to_rag.cli import app
from pydantic import ValidationError

from md_to_rag.schemas import (
    CommandName,
    CommandResponse,
    CommandStatus,
    InitResponse,
    InspectResponse,
)


runner = CliRunner()
COMMANDS = [command.value for command in CommandName]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_package_imports_with_version() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    assert __version__ == pyproject["project"]["version"]


def test_dependency_bounds_match_public_shell_requirements() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    dependencies = set(pyproject["project"]["dependencies"])

    assert "pydantic>=2.5,<3" in dependencies
    assert "typer>=0.16,<1" in dependencies


def test_api_facade_functions_return_owned_responses(tmp_path: Path) -> None:
    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "doc.md").write_text("# Doc\n", encoding="utf-8")
    api.ingest(source=project / "source")
    calls = {
        "init": lambda: api.init(project),
        "ingest": lambda: api.ingest(source=project / "source"),
        "chunk": lambda: api.chunk(manifest=project / "documents" / "documents.jsonl"),
        "embed": lambda: api.embed(chunks="chunks.jsonl"),
        "index": lambda: api.index(embeddings="embeddings.jsonl"),
        "query": lambda: api.query("What is indexed?"),
        "inspect": lambda: api.inspect(artifact=project),
    }

    assert set(calls) == set(COMMANDS)
    for command, call in calls.items():
        response = call()
        assert isinstance(response, CommandResponse)
        assert response.__class__.__module__.startswith("md_to_rag.")
        if command in {"init", "ingest", "chunk", "inspect"}:
            assert response.status is CommandStatus.OK
        else:
            assert response.status is CommandStatus.NOT_IMPLEMENTED
        assert "raganything" not in response.model_dump_json().lower()


def test_init_creates_idempotent_project_layout(tmp_path: Path) -> None:
    project = tmp_path / "portable-rag"

    first = api.init(project)
    manifest_path = project / "corpus_manifest.json"
    first_manifest_text = manifest_path.read_text(encoding="utf-8")
    second = api.init(project)

    assert isinstance(first, InitResponse)
    assert first.status is CommandStatus.OK
    assert first.data.created is True
    assert first.data.changed is True
    assert second.data.created is False
    assert second.data.changed is False
    assert manifest_path.read_text(encoding="utf-8") == first_manifest_text

    manifest = json.loads(first_manifest_text)
    assert manifest["schema_name"] == "md_to_rag.corpus_manifest"
    assert manifest["schema_version"] == "1.0"
    assert manifest["artifact_directories"] == {
        "chunks": "chunks",
        "documents": "documents",
        "embeddings": "embeddings",
        "indexes": "indexes",
        "reports": "reports",
        "source": "source",
    }
    assert all(not Path(path).is_absolute() for path in manifest["artifact_directories"].values())
    assert next(
        status for status in manifest["command_status"] if status["command"] == "init"
    )["status"] == "ok"
    assert next(
        status for status in manifest["command_status"] if status["command"] == "inspect"
    )["status"] == "ok"
    for relative_path in manifest["artifact_directories"].values():
        assert (project / relative_path).is_dir()


def test_init_reports_repaired_artifact_directories_as_changed(tmp_path: Path) -> None:
    project = tmp_path / "repairable"
    api.init(project)
    manifest_path = project / "corpus_manifest.json"
    first_manifest_text = manifest_path.read_text(encoding="utf-8")
    (project / "chunks").rmdir()

    repaired = api.init(project)

    assert repaired.status is CommandStatus.OK
    assert repaired.data.created is False
    assert repaired.data.changed is True
    assert repaired.message == "Project updated."
    assert (project / "chunks").is_dir()
    assert manifest_path.read_text(encoding="utf-8") == first_manifest_text


def test_init_backfills_manifest_statuses_with_current_timestamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from md_to_rag import manifest as manifest_module

    project = tmp_path / "old-manifest"
    project.mkdir()
    for relative_path in ("source", "documents", "chunks", "embeddings", "indexes", "reports"):
        (project / relative_path).mkdir()
    manifest_path = project / "corpus_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_name": "md_to_rag.corpus_manifest",
                "schema_version": "1.0",
                "md_to_rag_version": __version__,
                "created_at": "2026-06-01T00:00:00Z",
                "updated_at": "2026-06-01T00:00:00Z",
                "artifact_directories": {
                    "source": "source",
                    "documents": "documents",
                    "chunks": "chunks",
                    "embeddings": "embeddings",
                    "indexes": "indexes",
                    "reports": "reports",
                },
                "command_status": [
                    {
                        "command": "init",
                        "status": "ok",
                        "message": "Project initialized.",
                        "artifact_path": "corpus_manifest.json",
                        "updated_at": "2026-06-01T00:00:00Z",
                        "data": {},
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(manifest_module, "_utc_now", lambda: "2026-06-17T21:30:00Z")

    response = api.init(project)

    assert response.status is CommandStatus.OK
    assert response.data.created is False
    assert response.data.changed is True
    assert response.message == "Project updated."
    stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert stored_manifest["created_at"] == "2026-06-01T00:00:00Z"
    assert stored_manifest["updated_at"] == "2026-06-17T21:30:00Z"
    status_by_command = {
        status["command"]: status for status in stored_manifest["command_status"]
    }
    assert status_by_command["init"]["updated_at"] == "2026-06-01T00:00:00Z"
    assert status_by_command["inspect"]["updated_at"] == "2026-06-17T21:30:00Z"


def test_cli_inspect_json_reads_manifest_after_init(tmp_path: Path) -> None:
    project = tmp_path / "inspectable"

    init_result = runner.invoke(
        app,
        ["init", str(project), "--json"],
        prog_name="md-to-rag",
    )
    inspect_result = runner.invoke(
        app,
        ["inspect", str(project), "--json"],
        prog_name="md-to-rag",
    )

    assert init_result.exit_code == 0
    assert inspect_result.exit_code == 0
    payload = json.loads(inspect_result.output)
    assert payload["command"] == "inspect"
    assert payload["status"] == "ok"
    assert payload["artifact_path"] == str((project / "corpus_manifest.json").resolve())
    assert payload["data"]["artifact_type"] == "project"
    assert payload["data"]["manifest"]["schema_name"] == "md_to_rag.corpus_manifest"
    assert "raganything" not in inspect_result.output.lower()


def test_inspect_defaults_to_current_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "default-project"
    api.init(project)
    monkeypatch.chdir(project)

    response = api.inspect()

    assert isinstance(response, InspectResponse)
    assert response.status is CommandStatus.OK
    assert response.data.artifact_type == "project"
    assert response.data.manifest_path == str((project / "corpus_manifest.json").resolve())


def test_inspect_missing_manifest_and_artifact_are_typed(tmp_path: Path) -> None:
    missing_artifact = tmp_path / "missing-project"
    missing_result = runner.invoke(
        app,
        ["inspect", str(missing_artifact), "--json"],
        prog_name="md-to-rag",
    )

    assert missing_result.exit_code == 0
    missing_payload = json.loads(missing_result.output)
    assert missing_payload["status"] == "missing_artifact"
    assert missing_payload["error"]["code"] == "artifact_not_found"
    assert missing_payload["data"]["artifact_exists"] is False
    assert "traceback" not in missing_result.output.lower()

    no_manifest_dir = tmp_path / "plain-dir"
    no_manifest_dir.mkdir()
    no_manifest_result = runner.invoke(
        app,
        ["inspect", str(no_manifest_dir), "--json"],
        prog_name="md-to-rag",
    )

    assert no_manifest_result.exit_code == 0
    no_manifest_payload = json.loads(no_manifest_result.output)
    assert no_manifest_payload["status"] == "missing_artifact"
    assert no_manifest_payload["error"]["code"] == "manifest_not_found"
    assert no_manifest_payload["data"]["manifest_exists"] is False
    assert "traceback" not in no_manifest_result.output.lower()


def test_init_reports_artifact_directory_collisions_as_json(tmp_path: Path) -> None:
    project = tmp_path / "collision"
    project.mkdir()
    (project / "source").write_text("not a directory", encoding="utf-8")

    result = runner.invoke(
        app,
        ["init", str(project), "--json"],
        prog_name="md-to-rag",
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["command"] == "init"
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "artifact_path_is_file"
    assert "traceback" not in result.output.lower()

    parent_file = tmp_path / "parent-file"
    parent_file.write_text("not a directory", encoding="utf-8")
    child_result = runner.invoke(
        app,
        ["init", str(parent_file / "child"), "--json"],
        prog_name="md-to-rag",
    )
    child_payload = json.loads(child_result.output)
    assert child_result.exit_code == 0
    assert child_payload["status"] == "error"
    assert child_payload["error"]["code"] == "project_create_failed"
    assert "traceback" not in child_result.output.lower()


def test_init_reports_manifest_write_failures_as_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from md_to_rag import manifest as manifest_module

    project = tmp_path / "write-failure"
    api.init(project)
    manifest_path = project / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["md_to_rag_version"] = "old"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def fail_write_text(self: Path, text: str, encoding: str | None = None) -> int:
        raise PermissionError("simulated write failure")

    monkeypatch.setattr(manifest_module.Path, "write_text", fail_write_text)
    result = runner.invoke(
        app,
        ["init", str(project), "--json"],
        prog_name="md-to-rag",
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "manifest_write_failed"
    assert "traceback" not in result.output.lower()


def test_inspect_missing_artifact_anchors_to_target_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "anchored-project"
    api.init(project)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    nested_missing = project / "future" / "missing.md"
    nested_response = api.inspect(nested_missing)

    assert nested_response.status is CommandStatus.MISSING_ARTIFACT
    assert nested_response.data.manifest_exists is True
    assert nested_response.data.manifest_path == str(
        (project / "corpus_manifest.json").resolve()
    )

    monkeypatch.chdir(project)
    unrelated_missing = tmp_path / "outside" / "missing.md"
    unrelated_response = api.inspect(unrelated_missing)

    assert unrelated_response.status is CommandStatus.MISSING_ARTIFACT
    assert unrelated_response.data.manifest_exists is False
    assert unrelated_response.data.manifest_path is None


def test_inspect_rejects_incompatible_manifest_schema(tmp_path: Path) -> None:
    project = tmp_path / "bad-schema"
    project.mkdir()
    manifest_path = project / "corpus_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_name": "other.corpus_manifest",
                "schema_version": "1.0",
                "md_to_rag_version": "0.1.0",
                "created_at": "2026-06-17T00:00:00Z",
                "updated_at": "2026-06-17T00:00:00Z",
                "artifact_directories": {
                    "source": "source",
                    "documents": "documents",
                    "chunks": "chunks",
                    "embeddings": "embeddings",
                    "indexes": "indexes",
                    "reports": "reports",
                },
                "command_status": [],
            }
        ),
        encoding="utf-8",
    )

    inspect_response = api.inspect(project)
    init_response = api.init(project)

    assert inspect_response.status is CommandStatus.ERROR
    assert inspect_response.error is not None
    assert inspect_response.error.code == "manifest_invalid"
    assert init_response.status is CommandStatus.ERROR
    assert init_response.error is not None
    assert init_response.error.code == "manifest_invalid"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["schema_name"] == (
        "other.corpus_manifest"
    )


def test_inspect_rejects_missing_manifest_schema_markers(tmp_path: Path) -> None:
    project = tmp_path / "missing-markers"
    api.init(project)
    manifest_path = project / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["schema_name"]
    del manifest["schema_version"]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    inspect_response = api.inspect(project)
    init_response = api.init(project)

    assert inspect_response.status is CommandStatus.ERROR
    assert inspect_response.error is not None
    assert inspect_response.error.code == "manifest_invalid"
    assert init_response.status is CommandStatus.ERROR
    assert init_response.error is not None
    assert init_response.error.code == "manifest_invalid"
    stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "schema_name" not in stored_manifest
    assert "schema_version" not in stored_manifest


def test_missing_artifact_reports_invalid_manifest_as_present(tmp_path: Path) -> None:
    project = tmp_path / "invalid-project"
    project.mkdir()
    manifest_path = project / "corpus_manifest.json"
    manifest_path.write_text("{not valid json", encoding="utf-8")

    response = api.inspect(project / "future" / "missing.md")

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "manifest_invalid"
    assert response.data.artifact_exists is False
    assert response.data.manifest_exists is True
    assert response.data.manifest_path == str(manifest_path.resolve())
    assert response.data.manifest is None
    assert "Artifact does not exist." in response.data.issues


def test_mcp_tool_listing_uses_owned_schemas() -> None:
    tools = mcp.list_tools()

    assert {tool.command for tool in tools} == set(CommandName)
    for tool in tools:
        serialized = tool.model_dump_json().lower()
        assert tool.name == f"md_to_rag_{tool.command.value}"
        assert "command" not in tool.input_schema.get("properties", {})
        assert "raganything" not in serialized

    output_titles = {tool.command: tool.output_schema["title"] for tool in tools}
    assert output_titles[CommandName.INIT] == "InitResponse"
    assert output_titles[CommandName.INGEST] == "IngestResponse"
    assert output_titles[CommandName.CHUNK] == "ChunkResponse"
    assert output_titles[CommandName.INSPECT] == "InspectResponse"
    for command in set(CommandName) - {
        CommandName.INIT,
        CommandName.INGEST,
        CommandName.CHUNK,
        CommandName.INSPECT,
    }:
        assert output_titles[command] == "CommandResponse"

    query_tool = next(tool for tool in tools if tool.command is CommandName.QUERY)
    assert query_tool.input_schema["title"] == "QueryRequest"
    assert query_tool.input_schema["required"] == ["question"]

    init_tool = next(tool for tool in tools if tool.command is CommandName.INIT)
    ingest_tool = next(tool for tool in tools if tool.command is CommandName.INGEST)
    inspect_tool = next(tool for tool in tools if tool.command is CommandName.INSPECT)
    assert set(init_tool.output_schema["required"]) >= {"command", "status", "message", "data"}
    assert set(inspect_tool.output_schema["required"]) >= {
        "command",
        "status",
        "message",
        "data",
    }
    init_data_options = init_tool.output_schema["properties"]["data"]["anyOf"]
    assert {option["$ref"].split("/")[-1] for option in init_data_options} == {
        "EmptyResponseData",
        "InitResponseData",
    }
    assert set(ingest_tool.output_schema["required"]) >= {
        "command",
        "status",
        "message",
        "data",
    }
    ingest_data_options = ingest_tool.output_schema["properties"]["data"]["anyOf"]
    assert {option["$ref"].split("/")[-1] for option in ingest_data_options} == {
        "EmptyResponseData",
        "IngestErrorData",
        "IngestResponseData",
    }
    chunk_tool = next(tool for tool in tools if tool.command is CommandName.CHUNK)
    assert set(chunk_tool.output_schema["required"]) >= {
        "command",
        "status",
        "message",
        "data",
    }
    chunk_data_options = chunk_tool.output_schema["properties"]["data"]["anyOf"]
    assert {option["$ref"].split("/")[-1] for option in chunk_data_options} == {
        "ChunkErrorData",
        "ChunkResponseData",
        "EmptyResponseData",
    }


def test_md_to_rag_help_surface() -> None:
    result = runner.invoke(app, ["--help"], prog_name="md-to-rag")

    assert result.exit_code == 0
    assert "md-to-rag" in result.output
    for command in COMMANDS:
        assert command in result.output


def test_every_command_help_surface_includes_json_option() -> None:
    for command in COMMANDS:
        result = runner.invoke(app, [command, "--help"], prog_name="md-to-rag")
        assert result.exit_code == 0
        assert "--json" in result.output


def test_json_skeleton_output_is_stable_and_backend_neutral() -> None:
    args = ["query", "What artifacts exist?", "--json"]

    first = runner.invoke(app, args, prog_name="md-to-rag")
    second = runner.invoke(app, args, prog_name="md-to-rag")

    assert first.exit_code == 0
    assert first.output == second.output

    payload = json.loads(first.output)
    assert payload["command"] == "query"
    assert payload["status"] == "not_implemented"
    assert payload["message"]
    assert "raganything" not in first.output.lower()


def test_embed_index_query_keep_json_skeleton_behavior() -> None:
    command_args = {
        "embed": ["embed", "--json"],
        "index": ["index", "--json"],
        "query": ["query", "What artifacts exist?", "--json"],
    }

    assert set(command_args) == {
        "embed",
        "index",
        "query",
    }
    for command, args in command_args.items():
        result = runner.invoke(app, args, prog_name="md-to-rag")
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["command"] == command
        assert payload["status"] == "not_implemented"
        assert "raganything" not in result.output.lower()


def test_public_response_data_is_json_compatible() -> None:
    with pytest.raises(ValidationError):
        CommandResponse(
            command=CommandName.QUERY,
            status=CommandStatus.NOT_IMPLEMENTED,
            message="bad payload",
            data={"backend_object": object()},
        )
