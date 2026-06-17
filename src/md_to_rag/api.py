from __future__ import annotations

from pathlib import Path

from .manifest import ManifestError, initialize_project, inspect_project
from .schemas import (
    CommandName,
    CommandResponse,
    CommandStatus,
    EmptyResponseData,
    InitResponse,
    InspectResponse,
    skeleton_response,
)


def init(project: str | Path = ".") -> InitResponse:
    try:
        result = initialize_project(project)
    except ManifestError as error:
        return InitResponse(
            command=CommandName.INIT,
            status=CommandStatus.ERROR,
            message=error.message,
            error=error.to_command_error(),
            data=EmptyResponseData(),
        )

    return InitResponse(
        command=CommandName.INIT,
        status=CommandStatus.OK,
        message=result.message,
        artifact_path=result.data.manifest_path,
        data=result.data,
    )


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


def inspect(artifact: str | Path | None = None) -> InspectResponse:
    result = inspect_project(artifact)
    return InspectResponse(
        command=CommandName.INSPECT,
        status=result.status,
        message=result.message,
        artifact_path=result.data.manifest_path,
        error=result.error,
        data=result.data,
    )
