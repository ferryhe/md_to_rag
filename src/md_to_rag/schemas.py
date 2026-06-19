from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class CommandName(str, Enum):
    INIT = "init"
    INGEST = "ingest"
    CHUNK = "chunk"
    EMBED = "embed"
    INDEX = "index"
    QUERY = "query"
    INSPECT = "inspect"
    DIFF = "diff"
    REBUILD = "rebuild"


class CommandStatus(str, Enum):
    OK = "ok"
    NOT_IMPLEMENTED = "not_implemented"
    MISSING_ARTIFACT = "missing_artifact"
    ERROR = "error"


class CommandError(BaseModel):
    code: str
    message: str


class InitRequest(BaseModel):
    project: str = "."


class IngestRequest(BaseModel):
    source: str | None = None


class ChunkRequest(BaseModel):
    manifest: str | None = None


class EmbedRequest(BaseModel):
    chunks: str | None = None


class IndexRequest(BaseModel):
    embeddings: str | None = None


class QueryRequest(BaseModel):
    question: str


class InspectRequest(BaseModel):
    artifact: str | None = None


class DiffRequest(BaseModel):
    project: str | None = None


class RebuildRequest(BaseModel):
    project: str | None = None


class EmptyResponseData(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IngestResponseData(BaseModel):
    project_root: str
    manifest_path: str
    source_path: str
    changed: bool
    source_count: int
    document_count: int
    source_manifest_path: str
    documents_path: str
    source_manifest_hash: str
    documents_hash: str


class IngestErrorData(BaseModel):
    project_root: str | None = None
    manifest_path: str | None = None
    source_path: str | None = None


class ChunkResponseData(BaseModel):
    project_root: str
    manifest_path: str
    documents_path: str
    changed: bool
    document_count: int
    chunk_count: int
    chunks_path: str
    documents_hash: str
    chunks_hash: str


class ChunkErrorData(BaseModel):
    project_root: str | None = None
    manifest_path: str | None = None
    documents_path: str | None = None


class EmbedResponseData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_root: str
    manifest_path: str
    chunks_path: str
    changed: bool
    chunk_count: int
    embedding_count: int
    embeddings_path: str
    chunks_hash: str
    embeddings_hash: str
    profile: dict[str, JsonValue]


class EmbedErrorData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_root: str | None = None
    manifest_path: str | None = None
    chunks_path: str | None = None


class IndexResponseData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_root: str
    manifest_path: str
    embeddings_path: str
    changed: bool
    embedding_count: int
    vector_count: int
    index_manifest_path: str
    index_path: str
    embeddings_hash: str
    index_hash: str
    index_manifest_hash: str
    index_engine: str
    dimensions: int
    profile: dict[str, JsonValue]
    chunks_path: str | None = None


class IndexErrorData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_root: str | None = None
    manifest_path: str | None = None
    embeddings_path: str | None = None


class QueryResultData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int
    score: float
    chunk_id: str
    embedding_id: str
    doc_id: str
    source_id: str
    source_path: str
    chunk_index: int
    content: str
    line_start: int | None = None
    line_end: int | None = None
    heading_path: list[str] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    provenance: dict[str, JsonValue] = Field(default_factory=dict)


class QueryResponseData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_root: str
    manifest_path: str
    question: str
    index_manifest_path: str
    index_path: str
    embeddings_path: str
    result_count: int
    results: list[QueryResultData]


class QueryErrorData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_root: str | None = None
    manifest_path: str | None = None
    index_manifest_path: str | None = None


class DiffStageData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: CommandName
    status: str
    rebuild_needed: bool
    missing: bool = False
    stale: bool = False
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    current_hashes: dict[str, JsonValue] = Field(default_factory=dict)
    expected_hashes: dict[str, JsonValue] = Field(default_factory=dict)
    recorded_hashes: dict[str, JsonValue] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    error: CommandError | None = None


class DiffResponseData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_root: str
    manifest_path: str
    rebuild_needed: bool
    stages: list[DiffStageData]
    missing_stages: list[CommandName] = Field(default_factory=list)
    stale_stages: list[CommandName] = Field(default_factory=list)
    error_stages: list[CommandName] = Field(default_factory=list)


class DiffErrorData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_root: str | None = None
    manifest_path: str | None = None
    project_path: str | None = None


class RebuildStepData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: CommandName
    status: str
    message: str
    changed: bool | None = None
    artifact_path: str | None = None
    skipped: bool = False
    error: CommandError | None = None


class RebuildResponseData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_root: str
    manifest_path: str
    changed: bool
    completed: bool
    steps: list[RebuildStepData]
    stopped_at: CommandName | None = None


class RebuildErrorData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_root: str | None = None
    manifest_path: str | None = None
    project_path: str | None = None
    changed: bool = False
    completed: bool = False
    steps: list[RebuildStepData] = Field(default_factory=list)
    stopped_at: CommandName | None = None


class ManifestCommandStatus(BaseModel):
    command: CommandName
    status: CommandStatus
    message: str
    artifact_path: str | None = None
    updated_at: str | None = None
    data: dict[str, JsonValue] = Field(default_factory=dict)


class ProjectManifest(BaseModel):
    schema_name: Literal["md_to_rag.corpus_manifest"] = "md_to_rag.corpus_manifest"
    schema_version: Literal["1.0"] = "1.0"
    md_to_rag_version: str
    created_at: str
    updated_at: str
    artifact_directories: dict[str, str]
    command_status: list[ManifestCommandStatus]


class CommandRequest(BaseModel):
    command: CommandName
    options: dict[str, JsonValue] = Field(default_factory=dict)


class CommandResponse(BaseModel):
    command: CommandName
    status: CommandStatus
    message: str
    artifact_path: str | None = None
    error: CommandError | None = None
    data: dict[str, JsonValue] = Field(default_factory=dict)


class InitResponseData(BaseModel):
    project_root: str
    manifest_path: str
    created: bool
    changed: bool
    directories: dict[str, str]
    manifest: ProjectManifest


class InspectResponseData(BaseModel):
    artifact: str
    artifact_exists: bool
    artifact_type: str
    project_root: str | None = None
    manifest_path: str | None = None
    manifest_exists: bool = False
    manifest: ProjectManifest | None = None
    issues: list[str] = Field(default_factory=list)


class InitResponse(CommandResponse):
    command: Literal[CommandName.INIT]
    data: InitResponseData | EmptyResponseData


class IngestResponse(CommandResponse):
    command: Literal[CommandName.INGEST]
    data: IngestResponseData | IngestErrorData | EmptyResponseData


class ChunkResponse(CommandResponse):
    command: Literal[CommandName.CHUNK]
    data: ChunkResponseData | ChunkErrorData | EmptyResponseData


class EmbedResponse(CommandResponse):
    command: Literal[CommandName.EMBED]
    data: EmbedResponseData | EmbedErrorData | EmptyResponseData


class IndexResponse(CommandResponse):
    command: Literal[CommandName.INDEX]
    data: IndexResponseData | IndexErrorData | EmptyResponseData


class QueryResponse(CommandResponse):
    command: Literal[CommandName.QUERY]
    data: QueryResponseData | QueryErrorData | EmptyResponseData


class InspectResponse(CommandResponse):
    command: Literal[CommandName.INSPECT]
    data: InspectResponseData


class DiffResponse(CommandResponse):
    command: Literal[CommandName.DIFF]
    data: DiffResponseData | DiffErrorData | EmptyResponseData


class RebuildResponse(CommandResponse):
    command: Literal[CommandName.REBUILD]
    data: RebuildResponseData | RebuildErrorData | EmptyResponseData


class ToolMetadata(BaseModel):
    name: str
    command: CommandName
    description: str
    input_schema: dict[str, JsonValue]
    output_schema: dict[str, JsonValue]


COMMAND_INPUT_MODELS: dict[CommandName, type[BaseModel]] = {
    CommandName.INIT: InitRequest,
    CommandName.INGEST: IngestRequest,
    CommandName.CHUNK: ChunkRequest,
    CommandName.EMBED: EmbedRequest,
    CommandName.INDEX: IndexRequest,
    CommandName.QUERY: QueryRequest,
    CommandName.INSPECT: InspectRequest,
    CommandName.DIFF: DiffRequest,
    CommandName.REBUILD: RebuildRequest,
}


COMMAND_OUTPUT_MODELS: dict[CommandName, type[BaseModel]] = {
    CommandName.INIT: InitResponse,
    CommandName.INGEST: IngestResponse,
    CommandName.CHUNK: ChunkResponse,
    CommandName.EMBED: EmbedResponse,
    CommandName.INDEX: IndexResponse,
    CommandName.QUERY: QueryResponse,
    CommandName.INSPECT: InspectResponse,
    CommandName.DIFF: DiffResponse,
    CommandName.REBUILD: RebuildResponse,
}


def skeleton_response(
    command: CommandName,
    *,
    data: dict[str, JsonValue] | None = None,
) -> CommandResponse:
    return CommandResponse(
        command=command,
        status=CommandStatus.NOT_IMPLEMENTED,
        message=f"{command.value} is defined but not implemented yet.",
        data=data or {},
    )
