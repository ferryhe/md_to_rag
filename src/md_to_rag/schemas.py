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


class InspectResponse(CommandResponse):
    command: Literal[CommandName.INSPECT]
    data: InspectResponseData


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
}


COMMAND_OUTPUT_MODELS: dict[CommandName, type[BaseModel]] = {
    CommandName.INIT: InitResponse,
    CommandName.INGEST: IngestResponse,
    CommandName.CHUNK: CommandResponse,
    CommandName.EMBED: CommandResponse,
    CommandName.INDEX: CommandResponse,
    CommandName.QUERY: CommandResponse,
    CommandName.INSPECT: InspectResponse,
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
