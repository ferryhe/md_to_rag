import json
import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from md_to_rag import __version__, api, mcp
from md_to_rag.cli import app
from pydantic import ValidationError

from md_to_rag.schemas import CommandName, CommandResponse, CommandStatus


runner = CliRunner()
COMMANDS = [command.value for command in CommandName]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_package_imports_with_version() -> None:
    assert __version__ == "0.1.0"


def test_dependency_bounds_match_public_shell_requirements() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    dependencies = set(pyproject["project"]["dependencies"])

    assert "pydantic>=2.5,<3" in dependencies
    assert "typer>=0.16,<1" in dependencies


def test_api_facade_functions_return_owned_responses() -> None:
    calls = {
        "init": lambda: api.init("."),
        "ingest": lambda: api.ingest(source="."),
        "chunk": lambda: api.chunk(manifest="documents.jsonl"),
        "embed": lambda: api.embed(chunks="chunks.jsonl"),
        "index": lambda: api.index(embeddings="embeddings.jsonl"),
        "query": lambda: api.query("What is indexed?"),
        "inspect": lambda: api.inspect(artifact="manifest.json"),
    }

    assert set(calls) == set(COMMANDS)
    for call in calls.values():
        response = call()
        assert isinstance(response, CommandResponse)
        assert response.__class__.__module__.startswith("md_to_rag.")
        assert response.status is CommandStatus.NOT_IMPLEMENTED
        assert "raganything" not in response.model_dump_json().lower()


def test_mcp_tool_listing_uses_owned_schemas() -> None:
    tools = mcp.list_tools()

    assert {tool.command for tool in tools} == set(CommandName)
    for tool in tools:
        serialized = tool.model_dump_json().lower()
        assert tool.name == f"md_to_rag_{tool.command.value}"
        assert tool.output_schema["title"] == "CommandResponse"
        assert "command" not in tool.input_schema.get("properties", {})
        assert "raganything" not in serialized

    query_tool = next(tool for tool in tools if tool.command is CommandName.QUERY)
    assert query_tool.input_schema["title"] == "QueryRequest"
    assert query_tool.input_schema["required"] == ["question"]


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


def test_every_command_json_output_is_backend_neutral() -> None:
    command_args = {
        "init": ["init", "--json"],
        "ingest": ["ingest", "--json"],
        "chunk": ["chunk", "--json"],
        "embed": ["embed", "--json"],
        "index": ["index", "--json"],
        "query": ["query", "What artifacts exist?", "--json"],
        "inspect": ["inspect", "--json"],
    }

    assert set(command_args) == set(COMMANDS)
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
