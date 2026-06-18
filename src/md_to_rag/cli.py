from __future__ import annotations

import json
from pathlib import Path

import typer

from . import api
from .schemas import CommandResponse


app = typer.Typer(
    name="md-to-rag",
    help="Create and inspect Markdown-to-RAG artifacts.",
    no_args_is_help=True,
)


def _emit(response: CommandResponse, json_output: bool) -> None:
    payload = response.model_dump(mode="json")
    if json_output:
        typer.echo(json.dumps(payload, separators=(",", ":")))
        return

    typer.echo(f"{response.command.value}: {response.status.value} - {response.message}")
    if payload["data"]:
        typer.echo(json.dumps(payload["data"], sort_keys=True))


@app.command("init")
def init_command(
    project: Path = typer.Argument(Path("."), help="Project directory to initialize."),
    json_output: bool = typer.Option(False, "--json", help="Emit a stable JSON response."),
) -> None:
    """Prepare project artifact layout metadata."""

    _emit(api.init(project), json_output)


@app.command("ingest")
def ingest_command(
    source: Path | None = typer.Option(None, "--source", "-s", help="Markdown source path or manifest."),
    json_output: bool = typer.Option(False, "--json", help="Emit a stable JSON response."),
) -> None:
    """Read Markdown sources into document artifact metadata."""

    _emit(api.ingest(source=source), json_output)


@app.command("chunk")
def chunk_command(
    manifest: Path | None = typer.Option(None, "--manifest", "-m", help="Document manifest path."),
    json_output: bool = typer.Option(False, "--json", help="Emit a stable JSON response."),
) -> None:
    """Create chunk artifact metadata from document manifests."""

    _emit(api.chunk(manifest=manifest), json_output)


@app.command("embed")
def embed_command(
    chunks: Path | None = typer.Option(None, "--chunks", help="Chunk artifact path."),
    json_output: bool = typer.Option(False, "--json", help="Emit a stable JSON response."),
) -> None:
    """Create embedding artifact metadata from chunk artifacts."""

    _emit(api.embed(chunks=chunks), json_output)


@app.command("index")
def index_command(
    embeddings: Path | None = typer.Option(None, "--embeddings", help="Embedding artifact path."),
    json_output: bool = typer.Option(False, "--json", help="Emit a stable JSON response."),
) -> None:
    """Create index artifact metadata from embedding artifacts."""

    _emit(api.index(embeddings=embeddings), json_output)


@app.command("query")
def query_command(
    question: str = typer.Argument(..., help="Query text."),
    json_output: bool = typer.Option(False, "--json", help="Emit a stable JSON response."),
) -> None:
    """Return deterministic local retrieval results."""

    _emit(api.query(question), json_output)


@app.command("inspect")
def inspect_command(
    artifact: Path | None = typer.Argument(None, help="Artifact path to inspect."),
    json_output: bool = typer.Option(False, "--json", help="Emit a stable JSON response."),
) -> None:
    """Inspect artifact status metadata."""

    _emit(api.inspect(artifact=artifact), json_output)


def main() -> None:
    app(prog_name="md-to-rag")
