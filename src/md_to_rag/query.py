from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .chunk import _hash_bytes, _hash_text
from .index import (
    INDEX_ENGINE,
    INDEX_MANIFEST_PATH,
    INDEX_PATH,
    INDEX_VERSION,
    IndexInputError,
    _has_windows_reserved_path_component,
    _json_loads_strict,
    _jsonl_artifact_path_error,
    _read_embedding_rows,
    _stable_index_id,
)
from .ingest import _find_manifest_lexical, _nearest_nested_manifest
from .manifest import (
    MANIFEST_FILENAME,
    ManifestReadError,
    ManifestWriteError,
    _read_manifest,
    _utc_now,
    _write_manifest,
)
from .schemas import (
    CommandError,
    CommandName,
    CommandStatus,
    ManifestCommandStatus,
    ProjectManifest,
    QueryErrorData,
    QueryResponseData,
    QueryResultData,
)


TOP_K = 5
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class QueryProjectResult:
    status: CommandStatus
    message: str
    data: QueryResponseData | QueryErrorData
    artifact_path: str | None = None
    error: CommandError | None = None


@dataclass(frozen=True)
class _ProjectContext:
    project_root: Path
    manifest_path: Path
    manifest: ProjectManifest
    index_manifest_path: Path
    index_manifest_path_relative: str


class QueryInputError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: CommandStatus = CommandStatus.ERROR,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status

    def to_command_error(self) -> CommandError:
        return CommandError(code=self.code, message=self.message)


def query_project(question: str) -> QueryProjectResult:
    try:
        question.encode("utf-8")
    except UnicodeEncodeError as error:
        return QueryProjectResult(
            status=CommandStatus.ERROR,
            message="Query question must be valid UTF-8.",
            data=QueryErrorData(),
            error=CommandError(
                code="query_invalid_question",
                message="Query question must be valid UTF-8.",
            ),
        )

    context_result = _resolve_context()
    if isinstance(context_result, QueryProjectResult):
        return context_result
    context = context_result

    try:
        index_manifest = _read_index_manifest(context.index_manifest_path)
        index_path_relative = index_manifest["index_path"]
        index_path = _resolve_owned_artifact_path(
            context,
            index_path_relative,
            artifact_label="Index artifact",
        )
        index_rows, index_hash = _read_index_rows(index_path)
        if index_hash != index_manifest["index_hash"]:
            raise QueryInputError(
                "index_hash_mismatch",
                f"Index artifact hash does not match index manifest: {index_path_relative}",
            )

        embeddings_relative = index_manifest["embeddings_path"]
        embeddings_path = _resolve_owned_artifact_path(
            context,
            embeddings_relative,
            artifact_label="Embeddings artifact",
        )
        embedding_artifact = _read_embeddings_for_query(embeddings_path, embeddings_relative)
        if embedding_artifact.artifact_hash != index_manifest["embeddings_hash"]:
            raise QueryInputError(
                "embeddings_hash_mismatch",
                "Embeddings artifact hash does not match index manifest.",
            )
        _validate_index_manifest_against_artifacts(
            index_manifest,
            index_rows,
            embedding_artifact,
        )
        _validate_index_against_embeddings(index_rows, embedding_artifact.rows, index_manifest)

        chunk_rows = _chunk_rows_by_id(
            context,
            index_manifest.get("chunks_path"),
        )
        _validate_chunks_against_index(
            index_rows,
            chunk_rows,
            chunks_recorded=index_manifest.get("chunks_path") is not None,
        )
        result_rows = _rank_results(question, index_rows, chunk_rows)
    except QueryInputError as error:
        return _input_error_result(error, context)
    except OSError as error:
        query_error = QueryInputError(
            "query_io_failed",
            f"Could not read query artifacts: {error}",
        )
        return _input_error_result(query_error, context)

    data = QueryResponseData(
        project_root=str(context.project_root),
        manifest_path=str(context.manifest_path),
        question=question,
        index_manifest_path=context.index_manifest_path_relative,
        index_path=index_manifest["index_path"],
        embeddings_path=index_manifest["embeddings_path"],
        result_count=len(result_rows),
        results=result_rows,
    )
    try:
        _update_manifest_status(context.manifest_path, context.manifest, data)
    except ManifestWriteError as error:
        return QueryProjectResult(
            status=CommandStatus.ERROR,
            message=error.message,
            data=data,
            artifact_path=str(context.index_manifest_path.resolve()),
            error=error.to_command_error(),
        )
    return QueryProjectResult(
        status=CommandStatus.OK,
        message="Query results generated.",
        data=data,
        artifact_path=str(context.index_manifest_path.resolve()),
    )


def _resolve_context() -> _ProjectContext | QueryProjectResult:
    manifest_path = _find_manifest_lexical(Path.cwd())
    if manifest_path is None:
        return QueryProjectResult(
            status=CommandStatus.MISSING_ARTIFACT,
            message=f"No {MANIFEST_FILENAME} found for query.",
            data=QueryErrorData(),
            error=CommandError(
                code="manifest_not_found",
                message=f"No {MANIFEST_FILENAME} found for query.",
            ),
        )

    try:
        manifest = _read_manifest(manifest_path)
    except ManifestReadError as error:
        project_root = manifest_path.parent.resolve()
        return QueryProjectResult(
            status=CommandStatus.ERROR,
            message=error.message,
            data=QueryErrorData(
                project_root=str(project_root),
                manifest_path=str(manifest_path.resolve()),
            ),
            error=error.to_command_error(),
        )

    project_root = manifest_path.parent.resolve()
    index_manifest_relative = _index_manifest_path_from_status(manifest)
    status_path_is_utf8 = _is_utf8_string(index_manifest_relative)
    path_error = not status_path_is_utf8 or _json_artifact_path_error(index_manifest_relative)
    if path_error:
        portable_message = "Index manifest path must be project-relative and portable."
        if status_path_is_utf8:
            portable_message = f"{portable_message}: {index_manifest_relative}"
        return QueryProjectResult(
            status=CommandStatus.ERROR,
            message=portable_message,
            data=QueryErrorData(
                project_root=str(project_root),
                manifest_path=str(manifest_path.resolve()),
                index_manifest_path=index_manifest_relative if status_path_is_utf8 else None,
            ),
            error=CommandError(
                code="index_path_not_portable",
                message=portable_message,
            ),
        )
    context = _ProjectContext(
        project_root=project_root,
        manifest_path=manifest_path.resolve(),
        manifest=manifest,
        index_manifest_path=project_root / index_manifest_relative,
        index_manifest_path_relative=index_manifest_relative,
    )
    try:
        index_manifest_path = _resolve_owned_artifact_path(
            context,
            index_manifest_relative,
            artifact_label="Index manifest",
        )
    except QueryInputError as error:
        return QueryProjectResult(
            status=error.status,
            message=error.message,
            data=QueryErrorData(
                project_root=str(project_root),
                manifest_path=str(manifest_path.resolve()),
                index_manifest_path=index_manifest_relative,
            ),
            error=error.to_command_error(),
        )
    if not index_manifest_path.exists():
        return QueryProjectResult(
            status=CommandStatus.MISSING_ARTIFACT,
            message=f"Index artifact does not exist: {index_manifest_relative}",
            data=QueryErrorData(
                project_root=str(project_root),
                manifest_path=str(manifest_path.resolve()),
                index_manifest_path=index_manifest_relative,
            ),
            error=CommandError(
                code="index_not_found",
                message=f"Index artifact does not exist: {index_manifest_relative}",
            ),
        )

    return _ProjectContext(
        project_root=project_root,
        manifest_path=manifest_path.resolve(),
        manifest=manifest,
        index_manifest_path=index_manifest_path,
        index_manifest_path_relative=index_manifest_relative,
    )


def _index_manifest_path_from_status(manifest: ProjectManifest) -> str:
    for existing_status in manifest.command_status:
        if (
            existing_status.command is CommandName.INDEX
            and existing_status.status is CommandStatus.OK
            and isinstance(existing_status.artifact_path, str)
            and existing_status.artifact_path
        ):
            return existing_status.artifact_path
    return INDEX_MANIFEST_PATH


def _read_index_manifest(index_manifest_path: Path) -> dict[str, Any]:
    try:
        raw = _json_loads_strict(index_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise QueryInputError(
            "index_manifest_invalid",
            f"Could not read a valid md_to_rag index manifest at {index_manifest_path}: {error}",
        ) from error
    if not isinstance(raw, dict):
        raise QueryInputError(
            "index_manifest_invalid",
            f"Index manifest is not a JSON object: {index_manifest_path}",
        )
    if raw.get("schema_name") != "md_to_rag.index" or raw.get("schema_version") != "1.0":
        raise QueryInputError(
            "index_manifest_invalid",
            "Index manifest is not an md_to_rag.index v1.0 artifact.",
        )
    if raw.get("index_engine") != INDEX_ENGINE or raw.get("index_version") != INDEX_VERSION:
        raise QueryInputError(
            "index_manifest_invalid",
            "Index manifest uses an unsupported index engine or version.",
        )

    required_strings = (
        "embeddings_path",
        "embeddings_hash",
        "index_path",
        "index_hash",
    )
    invalid_strings = [
        field for field in required_strings if not isinstance(raw.get(field), str) or not raw[field]
    ]
    if invalid_strings:
        fields = ", ".join(invalid_strings)
        raise QueryInputError(
            "index_manifest_invalid",
            f"Index manifest has invalid string field(s): {fields}.",
        )
    for field in required_strings:
        _validate_utf8_string(
            raw[field],
            field,
            1,
            schema_name="Index manifest",
            code="index_manifest_invalid",
        )
    if _jsonl_artifact_path_error(raw["embeddings_path"]):
        raise QueryInputError(
            "index_manifest_invalid",
            f"Index manifest embeddings_path is not portable: {raw['embeddings_path']}",
        )
    if _jsonl_artifact_path_error(raw["index_path"]):
        raise QueryInputError(
            "index_manifest_invalid",
            f"Index manifest index_path is not portable: {raw['index_path']}",
        )
    if raw["index_path"] != INDEX_PATH:
        raise QueryInputError(
            "index_manifest_invalid",
            "Index manifest points to an unsupported index artifact path.",
        )
    chunks_path = raw.get("chunks_path")
    if chunks_path is not None:
        if not isinstance(chunks_path, str):
            raise QueryInputError(
                "index_manifest_invalid",
                "Index manifest chunks_path is not portable.",
            )
        _validate_utf8_string(
            chunks_path,
            "chunks_path",
            1,
            schema_name="Index manifest",
            code="index_manifest_invalid",
        )
        if _jsonl_artifact_path_error(chunks_path):
            raise QueryInputError(
                "index_manifest_invalid",
                "Index manifest chunks_path is not portable.",
            )
    for field in ("embedding_count", "vector_count", "dimensions"):
        if not isinstance(raw.get(field), int) or isinstance(raw[field], bool) or raw[field] < 0:
            raise QueryInputError(
                "index_manifest_invalid",
                f"Index manifest has invalid {field}.",
            )
    if not isinstance(raw.get("profile"), dict):
        raise QueryInputError(
            "index_manifest_invalid",
            "Index manifest has invalid profile.",
        )
    _validate_json_value_strings(
        raw["profile"],
        "profile",
        1,
        schema_name="Index manifest",
        code="index_manifest_invalid",
    )
    return raw


def _read_index_rows(index_path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        data = index_path.read_bytes()
        text = data.decode("utf-8")
    except FileNotFoundError as error:
        raise QueryInputError(
            "index_not_found",
            f"Index artifact does not exist: {index_path}",
            status=CommandStatus.MISSING_ARTIFACT,
        ) from error
    except UnicodeDecodeError as error:
        raise QueryInputError(
            "index_invalid_jsonl",
            f"Index artifact is not valid UTF-8: {index_path}",
        ) from error
    except OSError as error:
        raise QueryInputError(
            "index_read_failed",
            f"Could not read index artifact {index_path}: {error}",
        ) from error

    rows: list[dict[str, Any]] = []
    seen_index_ids: set[str] = set()
    seen_embedding_ids: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = _json_loads_strict(line)
        except (json.JSONDecodeError, ValueError) as error:
            raise QueryInputError(
                "index_invalid_jsonl",
                f"Index artifact contains invalid JSONL at line {line_number}: {error}",
            ) from error
        if not isinstance(row, dict):
            raise QueryInputError(
                "index_invalid_jsonl",
                f"Index artifact row {line_number} must be a JSON object.",
            )
        _validate_index_row(row, line_number)
        index_id = row["index_id"]
        if index_id in seen_index_ids:
            raise QueryInputError(
                "duplicate_index_id",
                f"Index artifact contains duplicate index_id at line {line_number}: {index_id}",
            )
        embedding_id = row["embedding_id"]
        if embedding_id in seen_embedding_ids:
            raise QueryInputError(
                "duplicate_embedding_id",
                f"Index artifact contains duplicate embedding_id at line {line_number}: {embedding_id}",
            )
        seen_index_ids.add(index_id)
        seen_embedding_ids.add(embedding_id)
        rows.append(row)
    return rows, _hash_bytes(data)


def _validate_index_row(row: dict[str, Any], line_number: int) -> None:
    if row.get("schema_name") != "md_to_rag.index_vector" or row.get("schema_version") != "1.0":
        raise QueryInputError(
            "index_schema_invalid",
            f"Index artifact row {line_number} is not an md_to_rag.index_vector v1.0 row.",
        )
    required_strings = (
        "index_id",
        "embedding_id",
        "chunk_id",
        "doc_id",
        "source_id",
        "source_path",
        "source_hash",
        "document_content_hash",
        "chunk_content_hash",
        "embedding_hash",
    )
    invalid_strings = [
        field for field in required_strings if not isinstance(row.get(field), str) or not row[field]
    ]
    if invalid_strings:
        fields = ", ".join(invalid_strings)
        raise QueryInputError(
            "index_schema_invalid",
            f"Index artifact row {line_number} has invalid string field(s): {fields}.",
        )
    for field in required_strings:
        _validate_utf8_string(
            row[field],
            field,
            line_number,
            schema_name="Index artifact",
            code="index_schema_invalid",
        )
    _validate_source_path(row["source_path"], line_number, schema_name="Index artifact")
    chunk_index = row.get("chunk_index")
    if not isinstance(chunk_index, int) or isinstance(chunk_index, bool) or chunk_index < 0:
        raise QueryInputError(
            "index_schema_invalid",
            f"Index artifact row {line_number} has invalid chunk_index.",
        )
    has_any_citation = any(field in row for field in ("line_start", "line_end", "heading_path"))
    if has_any_citation:
        line_start = row.get("line_start")
        line_end = row.get("line_end")
        heading_path = row.get("heading_path")
        if (
            not isinstance(line_start, int)
            or isinstance(line_start, bool)
            or line_start < 1
            or not isinstance(line_end, int)
            or isinstance(line_end, bool)
            or line_end < line_start
            or not isinstance(heading_path, list)
            or not all(isinstance(item, str) for item in heading_path)
        ):
            raise QueryInputError(
                "index_schema_invalid",
                f"Index artifact row {line_number} has invalid citation fields.",
            )
        for index, heading in enumerate(heading_path):
            _validate_utf8_string(
                heading,
                f"heading_path[{index}]",
                line_number,
                schema_name="Index artifact",
                code="index_schema_invalid",
            )
    vector = row.get("vector")
    if not isinstance(vector, list) or not all(_is_finite_number(value) for value in vector):
        raise QueryInputError(
            "index_schema_invalid",
            f"Index artifact row {line_number} has invalid vector.",
        )
    vector_norm = row.get("vector_norm")
    if not _is_finite_number(vector_norm) or float(vector_norm) < 0:
        raise QueryInputError(
            "index_schema_invalid",
            f"Index artifact row {line_number} has invalid vector_norm.",
        )
    expected_norm = round(sqrt(sum(float(value) * float(value) for value in vector)), 12)
    if float(vector_norm) != expected_norm:
        raise QueryInputError(
            "index_schema_invalid",
            f"Index artifact row {line_number} vector_norm does not match vector.",
        )
    if not isinstance(row.get("metadata"), dict) or not isinstance(row.get("provenance"), dict):
        raise QueryInputError(
            "index_schema_invalid",
            f"Index artifact row {line_number} has invalid metadata or provenance.",
        )
    _validate_json_value_strings(
        row["metadata"],
        "metadata",
        line_number,
        schema_name="Index artifact",
        code="index_schema_invalid",
    )
    _validate_json_value_strings(
        row["provenance"],
        "provenance",
        line_number,
        schema_name="Index artifact",
        code="index_schema_invalid",
    )
    expected_index_id = _stable_index_id(row["embedding_id"], row["embedding_hash"])
    if row["index_id"] != expected_index_id:
        raise QueryInputError(
            "index_schema_invalid",
            f"Index artifact row {line_number} index_id does not match row identity.",
        )


def _read_embeddings_for_query(embeddings_path: Path, embeddings_relative: str):
    try:
        return _read_embedding_rows(embeddings_path)
    except IndexInputError as error:
        if error.code == "embeddings_read_failed" and isinstance(error.__cause__, FileNotFoundError):
            raise QueryInputError(
                "embeddings_not_found",
                f"Embeddings artifact does not exist: {embeddings_relative}",
                status=CommandStatus.MISSING_ARTIFACT,
            ) from error
        raise QueryInputError(
            error.code,
            error.message,
            status=error.status,
        ) from error


def _validate_index_against_embeddings(
    index_rows: list[dict[str, Any]],
    embedding_rows: list[dict[str, Any]],
    index_manifest: dict[str, Any],
) -> None:
    embeddings_by_id = {row["embedding_id"]: row for row in embedding_rows}
    if len(index_rows) != len(embedding_rows):
        raise QueryInputError(
            "index_embedding_mismatch",
            "Index artifact vector count does not match embeddings artifact.",
        )
    for index_row in index_rows:
        embedding_row = embeddings_by_id.get(index_row["embedding_id"])
        if embedding_row is None:
            raise QueryInputError(
                "index_embedding_mismatch",
                f"Index row references missing embedding_id: {index_row['embedding_id']}",
            )
        copied_fields = (
            "chunk_id",
            "doc_id",
            "source_id",
            "source_path",
            "source_hash",
            "document_content_hash",
            "chunk_content_hash",
            "chunk_index",
            "embedding_hash",
        )
        if any(index_row[field] != embedding_row[field] for field in copied_fields):
            raise QueryInputError(
                "index_embedding_mismatch",
                f"Index row does not match embedding_id: {index_row['embedding_id']}",
            )
        if index_row["vector"] != embedding_row["embedding"]:
            raise QueryInputError(
                "index_embedding_mismatch",
                f"Index vector does not match embedding_id: {index_row['embedding_id']}",
            )
        if index_row["metadata"] != embedding_row["metadata"]:
            raise QueryInputError(
                "index_embedding_mismatch",
                f"Index metadata does not match embedding_id: {index_row['embedding_id']}",
            )
        expected_provenance = {
            **embedding_row["provenance"],
            "embeddings_path": index_manifest["embeddings_path"],
            "embedding_id": embedding_row["embedding_id"],
        }
        if index_row["provenance"] != expected_provenance:
            raise QueryInputError(
                "index_embedding_mismatch",
                f"Index provenance does not match embedding_id: {index_row['embedding_id']}",
            )


def _validate_index_manifest_against_artifacts(
    index_manifest: dict[str, Any],
    index_rows: list[dict[str, Any]],
    embedding_artifact: Any,
) -> None:
    if index_manifest["embedding_count"] != len(embedding_artifact.rows):
        raise QueryInputError(
            "index_embedding_mismatch",
            "Index manifest embedding_count does not match embeddings artifact.",
        )
    if index_manifest["vector_count"] != len(index_rows):
        raise QueryInputError(
            "index_embedding_mismatch",
            "Index manifest vector_count does not match index artifact.",
        )
    if embedding_artifact.rows:
        if index_manifest["dimensions"] != embedding_artifact.dimensions:
            raise QueryInputError(
                "index_embedding_mismatch",
                "Index manifest dimensions do not match embeddings artifact.",
            )
        if index_manifest["profile"] != embedding_artifact.profile:
            raise QueryInputError(
                "index_embedding_mismatch",
                "Index manifest profile does not match embeddings artifact.",
            )
    if embedding_artifact.rows and embedding_artifact.chunks_path is None:
        raise QueryInputError(
            "embedding_schema_invalid",
            "Embeddings artifact rows must include provenance.chunks_path.",
        )
    embedding_chunks_path = embedding_artifact.chunks_path
    if embedding_chunks_path is None and not embedding_artifact.rows:
        embedding_chunks_path = index_manifest.get("chunks_path")
    if index_manifest.get("chunks_path") != embedding_chunks_path:
        raise QueryInputError(
            "index_embedding_mismatch",
            "Index manifest chunks_path does not match embeddings artifact.",
        )


def _chunk_rows_by_id(context: _ProjectContext, chunks_path: Any) -> dict[str, dict[str, Any]]:
    if chunks_path is None:
        return {}
    if not isinstance(chunks_path, str):
        raise QueryInputError(
            "index_manifest_invalid",
            "Index manifest chunks_path is not portable.",
        )
    _validate_utf8_string(
        chunks_path,
        "chunks_path",
        1,
        schema_name="Index manifest",
        code="index_manifest_invalid",
    )
    if _jsonl_artifact_path_error(chunks_path):
        raise QueryInputError(
            "index_manifest_invalid",
            "Index manifest chunks_path is not portable.",
        )
    path = _resolve_owned_artifact_path(
        context,
        chunks_path,
        artifact_label="Chunks artifact",
    )
    try:
        data = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise QueryInputError(
            "chunks_not_found",
            f"Chunks artifact does not exist: {chunks_path}",
            status=CommandStatus.MISSING_ARTIFACT,
        ) from error
    except UnicodeDecodeError as error:
        raise QueryInputError(
            "chunks_invalid_jsonl",
            f"Chunks artifact is not valid UTF-8: {chunks_path}",
        ) from error
    except OSError as error:
        raise QueryInputError(
            "chunks_read_failed",
            f"Could not read chunks artifact {chunks_path}: {error}",
        ) from error

    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(data.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = _json_loads_strict(line)
        except (json.JSONDecodeError, ValueError) as error:
            raise QueryInputError(
                "chunks_invalid_jsonl",
                f"Chunks artifact contains invalid JSONL at line {line_number}: {error}",
            ) from error
        if (
            not isinstance(row, dict)
            or row.get("schema_name") != "md_to_rag.chunk"
            or row.get("schema_version") != "1.0"
        ):
            raise QueryInputError(
                "chunks_invalid_jsonl",
                f"Chunks artifact row {line_number} must be an md_to_rag.chunk v1.0 object.",
            )
        chunk_id = row.get("chunk_id")
        content = row.get("content")
        if not isinstance(chunk_id, str) or not chunk_id or not isinstance(content, str):
            raise QueryInputError(
                "chunks_invalid_jsonl",
                f"Chunks artifact row {line_number} has invalid chunk_id or content.",
            )
        _validate_utf8_string(
            chunk_id,
            "chunk_id",
            line_number,
            schema_name="Chunks artifact",
            code="chunks_invalid_jsonl",
        )
        _validate_utf8_string(
            content,
            "content",
            line_number,
            schema_name="Chunks artifact",
            code="chunks_invalid_jsonl",
        )
        if chunk_id in rows:
            raise QueryInputError(
                "chunks_invalid_jsonl",
                f"Chunks artifact row {line_number} duplicates chunk_id: {chunk_id}",
            )
        if row.get("content_hash") != _hash_text(content):
            raise QueryInputError(
                "chunks_invalid_jsonl",
                f"Chunks artifact row {line_number} content_hash does not match content.",
            )
        for field in (
            "doc_id",
            "source_id",
            "source_path",
            "source_hash",
            "document_content_hash",
        ):
            if not isinstance(row.get(field), str) or not row[field]:
                raise QueryInputError(
                    "chunks_invalid_jsonl",
                    f"Chunks artifact row {line_number} has invalid {field}.",
                )
            _validate_utf8_string(
                row[field],
                field,
                line_number,
                schema_name="Chunks artifact",
                code="chunks_invalid_jsonl",
            )
        chunk_index = row.get("chunk_index")
        if not isinstance(chunk_index, int) or isinstance(chunk_index, bool) or chunk_index < 0:
            raise QueryInputError(
                "chunks_invalid_jsonl",
                f"Chunks artifact row {line_number} has invalid chunk_index.",
            )
        line_start = row.get("line_start")
        line_end = row.get("line_end")
        if (
            not isinstance(line_start, int)
            or isinstance(line_start, bool)
            or line_start < 1
            or not isinstance(line_end, int)
            or isinstance(line_end, bool)
            or line_end < line_start
        ):
            raise QueryInputError(
                "chunks_invalid_jsonl",
                f"Chunks artifact row {line_number} has invalid line range.",
            )
        heading_path = row.get("heading_path")
        if not isinstance(heading_path, list) or not all(
            isinstance(item, str) for item in heading_path
        ):
            raise QueryInputError(
                "chunks_invalid_jsonl",
                f"Chunks artifact row {line_number} has invalid heading_path.",
            )
        for index, heading in enumerate(heading_path):
            _validate_utf8_string(
                heading,
                f"heading_path[{index}]",
                line_number,
                schema_name="Chunks artifact",
                code="chunks_invalid_jsonl",
            )
        for field in ("metadata", "provenance"):
            if not isinstance(row.get(field), dict):
                raise QueryInputError(
                    "chunks_invalid_jsonl",
                    f"Chunks artifact row {line_number} has invalid {field}.",
                )
            _validate_json_value_strings(
                row[field],
                field,
                line_number,
                schema_name="Chunks artifact",
                code="chunks_invalid_jsonl",
            )
        _validate_source_path(
            row["source_path"],
            line_number,
            schema_name="Chunks artifact",
            code="chunks_invalid_jsonl",
        )
        rows[chunk_id] = row
    return rows


def _validate_chunks_against_index(
    index_rows: list[dict[str, Any]],
    chunk_rows: dict[str, dict[str, Any]],
    *,
    chunks_recorded: bool,
) -> None:
    if not chunks_recorded:
        return
    indexed_chunk_ids = {index_row["chunk_id"] for index_row in index_rows}
    extra_chunk_ids = sorted(set(chunk_rows) - indexed_chunk_ids)
    if extra_chunk_ids:
        raise QueryInputError(
            "index_chunk_mismatch",
            f"Chunks artifact contains unindexed chunk_id: {extra_chunk_ids[0]}",
        )
    for index_row in index_rows:
        chunk_row = chunk_rows.get(index_row["chunk_id"])
        if chunk_row is None:
            raise QueryInputError(
                "index_chunk_mismatch",
                f"Chunks artifact is missing indexed chunk_id: {index_row['chunk_id']}",
            )
        copied_fields = (
            "doc_id",
            "source_id",
            "source_path",
            "source_hash",
            "document_content_hash",
            "chunk_index",
        )
        if any(chunk_row[field] != index_row[field] for field in copied_fields):
            raise QueryInputError(
                "index_chunk_mismatch",
                f"Chunk row does not match index row for chunk_id: {index_row['chunk_id']}",
            )
        if chunk_row["metadata"] != index_row["metadata"]:
            raise QueryInputError(
                "index_chunk_mismatch",
                f"Chunk metadata does not match index row for chunk_id: {index_row['chunk_id']}",
            )
        indexed_chunk_provenance = {
            key: value
            for key, value in index_row["provenance"].items()
            if key
            not in {
                "chunks_path",
                "chunk_id",
                "chunk_content_hash",
                "profile_hash",
                "embeddings_path",
                "embedding_id",
            }
        }
        if chunk_row["provenance"] != indexed_chunk_provenance:
            raise QueryInputError(
                "index_chunk_mismatch",
                f"Chunk provenance does not match index row for chunk_id: {index_row['chunk_id']}",
            )
        if chunk_row["content_hash"] != index_row["chunk_content_hash"]:
            raise QueryInputError(
                "index_chunk_mismatch",
                f"Chunk content hash does not match index row for chunk_id: {index_row['chunk_id']}",
            )
        citation_fields = ("line_start", "line_end", "heading_path")
        if not all(field in index_row for field in citation_fields):
            raise QueryInputError(
                "index_schema_invalid",
                f"Index artifact row is missing chunk citation fields for chunk_id: {index_row['chunk_id']}",
            )
        if (
            chunk_row["line_start"] != index_row["line_start"]
            or chunk_row["line_end"] != index_row["line_end"]
            or chunk_row["heading_path"] != index_row["heading_path"]
        ):
            raise QueryInputError(
                "index_chunk_mismatch",
                f"Chunk citation fields do not match index row for chunk_id: {index_row['chunk_id']}",
            )


def _rank_results(
    question: str,
    index_rows: list[dict[str, Any]],
    chunk_rows: dict[str, dict[str, Any]],
) -> list[QueryResultData]:
    query_tokens = _tokens(question)
    scored: list[tuple[float, str, int, str, dict[str, Any], dict[str, Any] | None]] = []
    for index_row in index_rows:
        chunk_row = chunk_rows.get(index_row["chunk_id"])
        score = _score_row(query_tokens, index_row, chunk_row)
        scored.append(
            (
                score,
                index_row["source_path"],
                index_row["chunk_index"],
                index_row["chunk_id"],
                index_row,
                chunk_row,
            )
        )
    scored.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))

    results: list[QueryResultData] = []
    positive_scored = [item for item in scored if item[0] > 0.0]
    for rank, (score, _source_path, _chunk_index, _chunk_id, index_row, chunk_row) in enumerate(
        positive_scored[:TOP_K],
        start=1,
    ):
        content = chunk_row.get("content", "") if chunk_row else ""
        heading_path = (
            index_row.get("heading_path", [])
            if "heading_path" in index_row
            else chunk_row.get("heading_path", [])
            if chunk_row
            else []
        )
        if not isinstance(heading_path, list) or not all(
            isinstance(item, str) for item in heading_path
        ):
            heading_path = []
        line_start = index_row.get("line_start") if "line_start" in index_row else None
        line_end = index_row.get("line_end") if "line_end" in index_row else None
        results.append(
            QueryResultData(
                rank=rank,
                score=round(score, 6),
                chunk_id=index_row["chunk_id"],
                embedding_id=index_row["embedding_id"],
                doc_id=index_row["doc_id"],
                source_id=index_row["source_id"],
                source_path=index_row["source_path"],
                chunk_index=index_row["chunk_index"],
                content=content,
                line_start=line_start if isinstance(line_start, int) else None,
                line_end=line_end if isinstance(line_end, int) else None,
                heading_path=heading_path,
                metadata=index_row["metadata"],
                provenance=index_row["provenance"],
            )
        )
    return results


def _score_row(
    query_tokens: list[str],
    index_row: dict[str, Any],
    chunk_row: dict[str, Any] | None,
) -> float:
    if not query_tokens:
        return 0.0
    text_parts = [
        index_row["source_path"],
        json.dumps(index_row.get("metadata", {}), sort_keys=True, ensure_ascii=False),
    ]
    if chunk_row is not None:
        text_parts.append(str(chunk_row.get("content", "")))
        text_parts.append(
            " ".join(
                item
                for item in chunk_row.get("heading_path", [])
                if isinstance(item, str)
            )
        )
    counts = Counter(_tokens(" ".join(text_parts)))
    unique_query_tokens = list(dict.fromkeys(query_tokens))
    raw_score = sum(counts[token] for token in unique_query_tokens)
    coverage = sum(1 for token in unique_query_tokens if counts[token])
    return raw_score + coverage / max(len(unique_query_tokens), 1)


def _tokens(value: str) -> list[str]:
    tokens: list[str] = []
    for match in TOKEN_RE.finditer(value):
        token = match.group(0).casefold()
        tokens.append(token)
        cjk_chars = [char for char in token if _is_cjk_char(char)]
        tokens.extend(cjk_chars)
        tokens.extend(
            left + right
            for left, right in zip(cjk_chars, cjk_chars[1:], strict=False)
        )
    return tokens


def _is_cjk_char(value: str) -> bool:
    codepoint = ord(value)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x3040 <= codepoint <= 0x30FF
        or 0xAC00 <= codepoint <= 0xD7AF
    )


def _update_manifest_status(
    manifest_path: Path,
    manifest: ProjectManifest,
    data: QueryResponseData,
) -> None:
    status = ManifestCommandStatus(
        command=CommandName.QUERY,
        status=CommandStatus.OK,
        message="Query results generated.",
        artifact_path=data.index_manifest_path,
        updated_at=_utc_now(),
        data={
            "index_manifest_path": data.index_manifest_path,
            "index_path": data.index_path,
            "embeddings_path": data.embeddings_path,
            "result_count": data.result_count,
        },
    )
    command_status = []
    replaced = False
    for existing_status in manifest.command_status:
        if existing_status.command is CommandName.QUERY:
            command_status.append(status)
            replaced = True
        else:
            command_status.append(existing_status)
    if not replaced:
        command_status.append(status)

    updated_manifest = manifest.model_copy(
        update={
            "updated_at": status.updated_at,
            "command_status": command_status,
        }
    )
    _write_manifest(manifest_path, updated_manifest)


def _input_error_result(error: QueryInputError, context: _ProjectContext) -> QueryProjectResult:
    return QueryProjectResult(
        status=error.status,
        message=error.message,
        data=QueryErrorData(
            project_root=str(context.project_root),
            manifest_path=str(context.manifest_path),
            index_manifest_path=context.index_manifest_path_relative,
        ),
        error=error.to_command_error(),
    )


def _is_finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        numeric = float(value)
    except (OverflowError, ValueError):
        return False
    return isfinite(numeric)


def _resolve_owned_artifact_path(
    context: _ProjectContext,
    artifact_path: str,
    *,
    artifact_label: str,
) -> Path:
    path = context.project_root / artifact_path
    try:
        resolved_path = path.resolve()
    except (OSError, RuntimeError, ValueError) as error:
        raise QueryInputError(
            "artifact_path_collision",
            f"{artifact_label} path cannot be resolved safely: {artifact_path}",
        ) from error
    try:
        resolved_path.relative_to(context.project_root.resolve())
    except ValueError as error:
        raise QueryInputError(
            "artifact_path_outside_project",
            f"{artifact_label} path must stay inside the initialized project: {artifact_path}",
        ) from error
    nested_manifest_path = _nearest_nested_manifest(
        resolved_path,
        context.project_root,
        context.manifest_path,
    )
    if nested_manifest_path is not None:
        raise QueryInputError(
            "artifact_path_nested_project",
            f"{artifact_label} path resolves inside a nested initialized project.",
        )
    if resolved_path != path.absolute():
        raise QueryInputError(
            "artifact_path_collision",
            f"{artifact_label} path cannot be a symlink or linked path: {artifact_path}",
        )
    if path.exists() and path.stat().st_nlink > 1:
        raise QueryInputError(
            "artifact_path_collision",
            f"{artifact_label} path cannot be a hard-linked path: {artifact_path}",
        )
    return resolved_path


def _validate_source_path(
    value: str,
    line_number: int,
    *,
    schema_name: str,
    code: str = "index_schema_invalid",
) -> None:
    normalized_text = value.replace("\\", "/")
    posix_path = PurePosixPath(normalized_text)
    windows_path = PureWindowsPath(normalized_text)
    if (
        "\\" in value
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in posix_path.parts
        or any(_has_windows_reserved_path_component(part) for part in posix_path.parts)
    ):
        raise QueryInputError(
            code,
            f"{schema_name} row {line_number} has non-portable source_path.",
        )


def _validate_json_value_strings(
    value: Any,
    field: str,
    line_number: int,
    *,
    schema_name: str,
    code: str,
) -> None:
    if isinstance(value, str):
        _validate_utf8_string(
            value,
            field,
            line_number,
            schema_name=schema_name,
            code=code,
        )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value_strings(
                item,
                f"{field}[{index}]",
                line_number,
                schema_name=schema_name,
                code=code,
            )
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise QueryInputError(
                    code,
                    f"{schema_name} row {line_number} has non-string key in {field}.",
                )
            _validate_utf8_string(
                key,
                f"{field} key",
                line_number,
                schema_name=schema_name,
                code=code,
            )
            _validate_json_value_strings(
                item,
                f"{field}.{key}",
                line_number,
                schema_name=schema_name,
                code=code,
            )
        return
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return
    if isinstance(value, float) and isfinite(value):
        return
    raise QueryInputError(
        code,
        f"{schema_name} row {line_number} has non-portable JSON value in {field}.",
    )


def _validate_utf8_string(
    value: str,
    field: str,
    line_number: int,
    *,
    schema_name: str,
    code: str,
) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise QueryInputError(
            code,
            f"{schema_name} row {line_number} has invalid UTF-8 string field: {field}.",
        ) from error


def _json_artifact_path_error(value: str) -> bool:
    normalized_text = value.replace("\\", "/")
    posix_path = PurePosixPath(normalized_text)
    windows_path = PureWindowsPath(normalized_text)
    return (
        "\\" in value
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in posix_path.parts
        or any(_has_windows_reserved_path_component(part) for part in posix_path.parts)
        or posix_path.suffix.lower() != ".json"
    )


def _is_utf8_string(value: str) -> bool:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True
