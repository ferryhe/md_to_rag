from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field, JsonValue


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
