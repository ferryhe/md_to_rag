from __future__ import annotations

from pathlib import Path

from .schemas import CommandName, CommandResponse, skeleton_response


def init(project: str | Path = ".") -> CommandResponse:
    return skeleton_response(CommandName.INIT, data={"project": str(project)})


def ingest(source: str | Path | None = None) -> CommandResponse:
    data = {"source": str(source)} if source is not None else {}
    return skeleton_response(CommandName.INGEST, data=data)


def chunk(manifest: str | Path | None = None) -> CommandResponse:
    data = {"manifest": str(manifest)} if manifest is not None else {}
    return skeleton_response(CommandName.CHUNK, data=data)


def embed(chunks: str | Path | None = None) -> CommandResponse:
    data = {"chunks": str(chunks)} if chunks is not None else {}
    return skeleton_response(CommandName.EMBED, data=data)


def index(embeddings: str | Path | None = None) -> CommandResponse:
    data = {"embeddings": str(embeddings)} if embeddings is not None else {}
    return skeleton_response(CommandName.INDEX, data=data)


def query(question: str) -> CommandResponse:
    return skeleton_response(CommandName.QUERY, data={"question": question})


def inspect(artifact: str | Path | None = None) -> CommandResponse:
    data = {"artifact": str(artifact)} if artifact is not None else {}
    return skeleton_response(CommandName.INSPECT, data=data)
