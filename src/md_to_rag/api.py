from __future__ import annotations

from pathlib import Path

from .chunk import chunk_project
from .embed import embed_project
from .index import index_project
from .ingest import ingest_project
from .manifest import ManifestError, initialize_project, inspect_project
from .query import query_project
from .schemas import (
    ChunkResponse,
    CommandName,
    CommandStatus,
    EmptyResponseData,
    EmbedResponse,
    IndexResponse,
    InitResponse,
    IngestResponse,
    InspectResponse,
    QueryResponse,
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


def ingest(source: str | Path | None = None) -> IngestResponse:
    result = ingest_project(source)
    return IngestResponse(
        command=CommandName.INGEST,
        status=result.status,
        message=result.message,
        artifact_path=result.artifact_path,
        error=result.error,
        data=result.data,
    )


def chunk(manifest: str | Path | None = None) -> ChunkResponse:
    result = chunk_project(manifest)
    return ChunkResponse(
        command=CommandName.CHUNK,
        status=result.status,
        message=result.message,
        artifact_path=result.artifact_path,
        error=result.error,
        data=result.data,
    )


def embed(chunks: str | Path | None = None) -> EmbedResponse:
    result = embed_project(chunks)
    return EmbedResponse(
        command=CommandName.EMBED,
        status=result.status,
        message=result.message,
        artifact_path=result.artifact_path,
        error=result.error,
        data=result.data,
    )


def index(embeddings: str | Path | None = None) -> IndexResponse:
    result = index_project(embeddings)
    return IndexResponse(
        command=CommandName.INDEX,
        status=result.status,
        message=result.message,
        artifact_path=result.artifact_path,
        error=result.error,
        data=result.data,
    )


def query(question: str) -> QueryResponse:
    result = query_project(question)
    return QueryResponse(
        command=CommandName.QUERY,
        status=result.status,
        message=result.message,
        artifact_path=result.artifact_path,
        error=result.error,
        data=result.data,
    )


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
