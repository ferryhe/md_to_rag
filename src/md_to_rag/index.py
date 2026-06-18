from __future__ import annotations

import json
from dataclasses import dataclass, replace
from hashlib import sha256
from math import isfinite, sqrt
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable

from .chunk import _hash_bytes, _hash_text, _write_if_changed
from .embed import EmbedInputError, _sanitize_profile_value
from .ingest import (
    WINDOWS_RESERVED_BASENAMES,
    _find_manifest_lexical,
    _nearest_nested_manifest,
)
from .manifest import (
    MANIFEST_FILENAME,
    ManifestReadError,
    ManifestWriteError,
    _nearest_existing_ancestor,
    _read_manifest,
    _utc_now,
    _write_manifest,
)
from .schemas import (
    CommandError,
    CommandName,
    CommandStatus,
    IndexErrorData,
    IndexResponseData,
    ManifestCommandStatus,
    ProjectManifest,
)


EMBEDDINGS_PATH = "embeddings/embeddings.jsonl"
INDEX_MANIFEST_PATH = "indexes/index_manifest.json"
INDEX_PATH = "indexes/vectors.jsonl"
INDEX_ENGINE = "md_to_rag.local_vector"
INDEX_VERSION = "1.0"
INDEX_PROVENANCE_KEYS = {"embedding_id", "embeddings_path"}


@dataclass(frozen=True)
class IndexProjectResult:
    status: CommandStatus
    message: str
    data: IndexResponseData | IndexErrorData
    artifact_path: str | None = None
    error: CommandError | None = None


@dataclass(frozen=True)
class _ProjectContext:
    project_root: Path
    manifest_path: Path
    manifest: ProjectManifest
    embeddings_path: Path
    embeddings_path_relative: str


@dataclass(frozen=True)
class _EmbeddingArtifact:
    rows: list[dict[str, Any]]
    artifact_hash: str
    profile: dict[str, Any]
    dimensions: int
    chunks_path: str | None


@dataclass(frozen=True)
class _EmbedStatusMetadata:
    chunks_path: str
    profile: dict[str, Any]
    dimensions: int


class IndexInputError(Exception):
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


def index_project(embeddings: str | Path | None = None) -> IndexProjectResult:
    context_result = _resolve_context(embeddings)
    if isinstance(context_result, IndexProjectResult):
        return context_result
    context = context_result

    try:
        index_manifest_path = context.project_root / INDEX_MANIFEST_PATH
        index_path = context.project_root / INDEX_PATH
        _reject_output_path_outside_project(
            context,
            index_manifest_path,
            INDEX_MANIFEST_PATH,
        )
        _reject_output_path_outside_project(context, index_path, INDEX_PATH)
        embedding_artifact = _read_embedding_rows(context.embeddings_path)
        embedding_artifact = _ensure_chunk_provenance(context, embedding_artifact)
        chunk_rows = _chunk_rows_by_id(context, embedding_artifact.chunks_path)
        index_rows = _index_rows(
            embedding_artifact.rows,
            context,
            chunk_rows,
            chunks_recorded=embedding_artifact.chunks_path is not None,
        )
        index_text = _jsonl_text(index_rows)
        index_hash = _hash_text(index_text)
        index_manifest = _index_manifest_payload(
            context,
            embedding_artifact,
            index_hash,
        )
        index_manifest_text = _json_text(index_manifest)
        index_manifest_hash = _hash_text(index_manifest_text)
        index_changed = _write_if_changed(index_path, index_text)
        index_manifest_changed = _write_if_changed(
            index_manifest_path,
            index_manifest_text,
        )
    except IndexInputError as error:
        return _input_error_result(error, context)
    except OSError as error:
        index_error = IndexInputError(
            "index_io_failed",
            f"Could not generate index artifacts: {error}",
        )
        return _input_error_result(index_error, context)

    data = IndexResponseData(
        project_root=str(context.project_root),
        manifest_path=str(context.manifest_path),
        embeddings_path=context.embeddings_path_relative,
        changed=False,
        embedding_count=len(embedding_artifact.rows),
        vector_count=len(index_rows),
        index_manifest_path=INDEX_MANIFEST_PATH,
        index_path=INDEX_PATH,
        embeddings_hash=embedding_artifact.artifact_hash,
        index_hash=index_hash,
        index_manifest_hash=index_manifest_hash,
        index_engine=INDEX_ENGINE,
        dimensions=embedding_artifact.dimensions,
        profile=embedding_artifact.profile,
        chunks_path=embedding_artifact.chunks_path,
    )
    manifest_status_changed = not _manifest_status_matches(context.manifest, data)
    changed = index_changed or index_manifest_changed or manifest_status_changed
    data = data.model_copy(update={"changed": changed})

    if changed:
        try:
            _update_manifest_status(context.manifest_path, context.manifest, data)
        except ManifestWriteError as error:
            return IndexProjectResult(
                status=CommandStatus.ERROR,
                message=error.message,
                data=data,
                artifact_path=str(index_manifest_path.resolve()),
                error=error.to_command_error(),
            )
        message = "Index artifacts generated."
    else:
        message = "Index artifacts unchanged."

    return IndexProjectResult(
        status=CommandStatus.OK,
        message=message,
        data=data,
        artifact_path=str(index_manifest_path.resolve()),
    )


def _resolve_context(embeddings: str | Path | None) -> _ProjectContext | IndexProjectResult:
    if embeddings is None:
        manifest_path = _find_manifest_lexical(Path.cwd())
        requested_embeddings_path: Path | None = None
    else:
        path_error = _raw_embeddings_path_error(embeddings)
        if path_error is not None:
            embeddings_path_text = _safe_response_text(str(embeddings))
            return IndexProjectResult(
                status=path_error.status,
                message=path_error.message,
                data=IndexErrorData(embeddings_path=embeddings_path_text),
                error=path_error.to_command_error(),
            )
        requested_embeddings_path = _resolve_user_path(embeddings)
        anchor = (
            requested_embeddings_path
            if requested_embeddings_path.exists()
            else _nearest_existing_ancestor(requested_embeddings_path)
        )
        manifest_path = _find_manifest_lexical(anchor)

    if manifest_path is None:
        return IndexProjectResult(
            status=CommandStatus.MISSING_ARTIFACT,
            message=f"No {MANIFEST_FILENAME} found for embeddings artifact.",
            data=IndexErrorData(
                embeddings_path=(
                    str(_resolve_user_path(embeddings))
                    if embeddings is not None
                    else None
                ),
            ),
            error=CommandError(
                code="manifest_not_found",
                message=f"No {MANIFEST_FILENAME} found for embeddings artifact.",
            ),
        )

    try:
        project_manifest = _read_manifest(manifest_path)
    except ManifestReadError as error:
        project_root = manifest_path.parent.resolve()
        return IndexProjectResult(
            status=CommandStatus.ERROR,
            message=error.message,
            data=IndexErrorData(
                project_root=str(project_root),
                manifest_path=str(manifest_path.resolve()),
                embeddings_path=str(embeddings) if embeddings is not None else None,
            ),
            error=error.to_command_error(),
        )

    project_root = manifest_path.parent.resolve()
    if requested_embeddings_path is None:
        requested_embeddings_path = project_root / EMBEDDINGS_PATH

    try:
        embeddings_path = requested_embeddings_path.resolve()
    except (OSError, RuntimeError, ValueError) as error:
        message = f"Could not resolve embeddings artifact: {requested_embeddings_path}"
        return IndexProjectResult(
            status=CommandStatus.ERROR,
            message=message,
            data=IndexErrorData(
                project_root=str(project_root),
                manifest_path=str(manifest_path.resolve()),
                embeddings_path=str(requested_embeddings_path),
            ),
            error=CommandError(
                code="embeddings_path_unresolvable",
                message=f"{message}: {error}",
            ),
        )

    embeddings_relative = _relative_to_project(embeddings_path, project_root)
    context = _ProjectContext(
        project_root=project_root,
        manifest_path=manifest_path.resolve(),
        manifest=project_manifest,
        embeddings_path=embeddings_path,
        embeddings_path_relative=(
            embeddings_relative
            if isinstance(embeddings_relative, str)
            else str(embeddings_path)
        ),
    )
    if isinstance(embeddings_relative, IndexInputError):
        return _input_error_result(embeddings_relative, context)

    if embeddings_path != requested_embeddings_path.absolute() and not _uses_linked_project_root(
        requested_embeddings_path,
        embeddings_path,
        manifest_path.parent,
        project_root,
    ):
        return _input_error_result(
            IndexInputError(
                "artifact_path_collision",
                f"Embeddings artifact path cannot be a symlink or linked path: {requested_embeddings_path}",
            ),
            context,
        )

    portability_error = _embeddings_artifact_path_error(embeddings_relative)
    if portability_error is not None:
        return _input_error_result(portability_error, context)

    nested_manifest_path = _nearest_nested_manifest(
        embeddings_path,
        project_root,
        manifest_path,
    )
    if nested_manifest_path is not None:
        return _input_error_result(
            IndexInputError(
                "embeddings_nested_project",
                "Embeddings artifact resolves inside a nested initialized project; "
                "use that project path directly.",
            ),
            context,
        )

    if not embeddings_path.exists():
        return IndexProjectResult(
            status=CommandStatus.MISSING_ARTIFACT,
            message=f"Embeddings artifact does not exist: {embeddings_relative}",
            data=IndexErrorData(
                project_root=str(project_root),
                manifest_path=str(manifest_path.resolve()),
                embeddings_path=embeddings_relative,
            ),
            error=CommandError(
                code="embeddings_not_found",
                message=f"Embeddings artifact does not exist: {embeddings_relative}",
            ),
        )
    try:
        embeddings_link_count = embeddings_path.stat().st_nlink
    except OSError as error:
        return _input_error_result(
            IndexInputError(
                "embeddings_read_failed",
                f"Could not inspect embeddings artifact {embeddings_relative}: {error}",
            ),
            context,
        )
    if embeddings_link_count > 1:
        return _input_error_result(
            IndexInputError(
                "artifact_path_collision",
                f"Embeddings artifact path cannot be a hard-linked path: {embeddings_relative}",
            ),
            context,
        )

    return context


def _uses_linked_project_root(
    requested_path: Path,
    resolved_path: Path,
    lexical_project_root: Path,
    project_root: Path,
) -> bool:
    try:
        lexical_relative = requested_path.absolute().relative_to(lexical_project_root.absolute())
    except ValueError:
        return False
    return (project_root / lexical_relative).absolute() == resolved_path


def _read_embedding_rows(embeddings_path: Path) -> _EmbeddingArtifact:
    try:
        data = embeddings_path.read_bytes()
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise IndexInputError(
            "embeddings_invalid_jsonl",
            f"Embeddings artifact is not valid UTF-8: {embeddings_path}",
        ) from error
    except OSError as error:
        raise IndexInputError(
            "embeddings_read_failed",
            f"Could not read embeddings artifact {embeddings_path}: {error}",
        ) from error

    rows: list[dict[str, Any]] = []
    seen_embedding_ids: set[str] = set()
    seen_chunk_ids: set[str] = set()
    profile_text: str | None = None
    profile: dict[str, Any] = {}
    dimensions = 0
    chunks_path: str | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = _json_loads_strict(line)
        except (json.JSONDecodeError, ValueError) as error:
            raise IndexInputError(
                "embeddings_invalid_jsonl",
                f"Embeddings artifact contains invalid JSONL at line {line_number}: {error}",
            ) from error
        if not isinstance(row, dict):
            raise IndexInputError(
                "embeddings_invalid_jsonl",
                f"Embeddings artifact row {line_number} must be a JSON object.",
            )

        row_profile, row_dimensions, row_chunks_path = _validate_embedding_row(
            row,
            line_number,
        )
        current_profile_text = _json_dumps_canonical(row_profile)
        if profile_text is None:
            profile_text = current_profile_text
            profile = row_profile
            dimensions = row_dimensions
            chunks_path = row_chunks_path
        elif current_profile_text != profile_text:
            raise IndexInputError(
                "embedding_profile_mismatch",
                "Embeddings artifact rows must use one embedding profile.",
            )
        elif row_chunks_path != chunks_path:
            raise IndexInputError(
                "embedding_schema_invalid",
                "Embeddings artifact rows must reference one chunks artifact.",
            )

        embedding_id = row["embedding_id"]
        if embedding_id in seen_embedding_ids:
            raise IndexInputError(
                "duplicate_embedding_id",
                f"Embeddings artifact contains duplicate embedding_id at line {line_number}: {embedding_id}",
            )
        chunk_id = row["chunk_id"]
        if chunk_id in seen_chunk_ids:
            raise IndexInputError(
                "duplicate_chunk_id",
                f"Embeddings artifact contains duplicate chunk_id at line {line_number}: {chunk_id}",
            )
        seen_embedding_ids.add(embedding_id)
        seen_chunk_ids.add(chunk_id)
        rows.append(row)

    return _EmbeddingArtifact(
        rows=rows,
        artifact_hash=_hash_bytes(data),
        profile=profile,
        dimensions=dimensions,
        chunks_path=chunks_path,
    )


def _ensure_chunk_provenance(
    context: _ProjectContext,
    embedding_artifact: _EmbeddingArtifact,
) -> _EmbeddingArtifact:
    if embedding_artifact.chunks_path is not None:
        return embedding_artifact
    if embedding_artifact.rows:
        raise IndexInputError(
            "embedding_schema_invalid",
            "Embeddings artifact rows must include provenance.chunks_path.",
        )
    embed_status_metadata = _embedding_metadata_from_embed_status(
        context,
        embedding_artifact,
    )
    if embed_status_metadata is None:
        raise IndexInputError(
            "embedding_schema_invalid",
            "Embeddings artifact must record chunks_path provenance before indexing.",
        )
    return replace(
        embedding_artifact,
        chunks_path=embed_status_metadata.chunks_path,
        profile=embed_status_metadata.profile,
        dimensions=embed_status_metadata.dimensions,
    )


def _embedding_metadata_from_embed_status(
    context: _ProjectContext,
    embedding_artifact: _EmbeddingArtifact,
) -> _EmbedStatusMetadata | None:
    for existing_status in context.manifest.command_status:
        if existing_status.command is not CommandName.EMBED:
            continue
        if existing_status.status is not CommandStatus.OK:
            continue
        if (
            existing_status.artifact_path != EMBEDDINGS_PATH
            or existing_status.data.get("embeddings_path") != context.embeddings_path_relative
            or existing_status.data.get("embeddings_hash") != embedding_artifact.artifact_hash
            or existing_status.data.get("embedding_count") != 0
        ):
            raise IndexInputError(
                "embedding_chunk_mismatch",
                "Embed manifest status does not match the empty embeddings artifact.",
            )
        chunks_path = existing_status.data.get("chunks_path")
        if not isinstance(chunks_path, str) or not chunks_path:
            raise IndexInputError(
                "embedding_schema_invalid",
                "Embed manifest status has invalid chunks_path provenance.",
            )
        try:
            chunks_path.encode("utf-8")
        except UnicodeEncodeError as error:
            raise IndexInputError(
                "embedding_schema_invalid",
                "Embed manifest status has invalid chunks_path provenance.",
            ) from error
        if _jsonl_artifact_path_error(chunks_path):
            raise IndexInputError(
                "embedding_schema_invalid",
                "Embed manifest status has invalid chunks_path provenance.",
            )
        chunks_count, chunks_hash = _chunk_artifact_signature(context, chunks_path)
        if (
            existing_status.data.get("chunk_count") != chunks_count
            or existing_status.data.get("chunks_hash") != chunks_hash
        ):
            raise IndexInputError(
                "embedding_chunk_mismatch",
                "Embed manifest status does not match the current chunks artifact.",
            )
        profile = existing_status.data.get("profile")
        if not isinstance(profile, dict):
            raise IndexInputError(
                "embedding_schema_invalid",
                "Embed manifest status has invalid profile provenance.",
            )
        try:
            sanitized_profile = _sanitize_profile_value(profile)
        except EmbedInputError as error:
            raise IndexInputError(
                "embedding_profile_invalid",
                "Embed manifest status profile must be portable JSON.",
            ) from error
        if (
            not isinstance(sanitized_profile, dict)
            or _json_dumps_canonical(sanitized_profile) != _json_dumps_canonical(profile)
        ):
            raise IndexInputError(
                "embedding_profile_invalid",
                "Embed manifest status profile must not contain secrets.",
            )
        dimensions = profile.get("dimensions")
        if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions < 1:
            raise IndexInputError(
                "embedding_schema_invalid",
                "Embed manifest status has invalid profile dimensions.",
            )
        return _EmbedStatusMetadata(
            chunks_path=chunks_path,
            profile=json.loads(_json_dumps_canonical(profile)),
            dimensions=dimensions,
        )
    return None


def _chunk_artifact_signature(context: _ProjectContext, chunks_path: str) -> tuple[int, str]:
    path = _resolve_owned_input_artifact_path(
        context,
        chunks_path,
        artifact_label="Chunks artifact",
    )
    try:
        data = path.read_bytes()
        text = data.decode("utf-8")
    except FileNotFoundError as error:
        raise IndexInputError(
            "chunks_not_found",
            f"Chunks artifact does not exist: {chunks_path}",
            status=CommandStatus.MISSING_ARTIFACT,
        ) from error
    except UnicodeDecodeError as error:
        raise IndexInputError(
            "chunks_invalid_jsonl",
            f"Chunks artifact is not valid UTF-8: {chunks_path}",
        ) from error
    except OSError as error:
        raise IndexInputError(
            "chunks_read_failed",
            f"Could not read chunks artifact {chunks_path}: {error}",
        ) from error
    row_count = sum(1 for line in text.splitlines() if line.strip())
    return row_count, _hash_bytes(data)


def _validate_embedding_row(
    row: dict[str, Any],
    line_number: int,
) -> tuple[dict[str, Any], int, str | None]:
    if row.get("schema_name") != "md_to_rag.embedding" or row.get("schema_version") != "1.0":
        raise IndexInputError(
            "embedding_schema_invalid",
            f"Embeddings artifact row {line_number} is not an md_to_rag.embedding v1.0 row.",
        )

    required_strings = (
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
    invalid_string_fields = [
        field
        for field in required_strings
        if not isinstance(row.get(field), str) or not row[field]
    ]
    if invalid_string_fields:
        fields = ", ".join(invalid_string_fields)
        raise IndexInputError(
            "embedding_schema_invalid",
            f"Embeddings artifact row {line_number} has invalid string field(s): {fields}.",
        )
    for field in required_strings:
        _validate_utf8_string(row[field], field, line_number)
    _validate_source_path(row["source_path"], line_number)

    chunk_index = row.get("chunk_index")
    if not isinstance(chunk_index, int) or isinstance(chunk_index, bool) or chunk_index < 0:
        raise IndexInputError(
            "embedding_schema_invalid",
            f"Embeddings artifact row {line_number} has invalid chunk_index.",
        )
    if not isinstance(row.get("metadata"), dict):
        raise IndexInputError(
            "embedding_schema_invalid",
            f"Embeddings artifact row {line_number} has invalid metadata.",
        )
    _validate_json_value_strings(row["metadata"], "metadata", line_number)
    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        raise IndexInputError(
            "embedding_schema_invalid",
            f"Embeddings artifact row {line_number} has invalid provenance.",
        )
    _validate_json_value_strings(provenance, "provenance", line_number)
    profile = row.get("profile")
    if not isinstance(profile, dict):
        raise IndexInputError(
            "embedding_schema_invalid",
            f"Embeddings artifact row {line_number} has invalid profile.",
        )
    _validate_json_value_strings(profile, "profile", line_number)
    try:
        sanitized_profile = _sanitize_profile_value(profile)
    except EmbedInputError as error:
        raise IndexInputError(
            "embedding_profile_invalid",
            "Embeddings artifact profile must be portable JSON.",
        ) from error
    if _json_dumps_canonical(sanitized_profile) != _json_dumps_canonical(profile):
        raise IndexInputError(
            "embedding_profile_invalid",
            "Embeddings artifact profile must not contain secrets.",
        )
    dimensions = profile.get("dimensions")
    if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions < 1:
        raise IndexInputError(
            "embedding_schema_invalid",
            f"Embeddings artifact row {line_number} has invalid profile dimensions.",
        )
    vector = _validate_vector(row.get("embedding"), dimensions, line_number)
    vector_hash = _hash_text(_json_dumps_canonical(vector))
    if row["embedding_hash"] != vector_hash:
        raise IndexInputError(
            "embedding_schema_invalid",
            f"Embeddings artifact row {line_number} embedding_hash does not match embedding.",
        )
    profile_hash = _hash_text(_json_dumps_canonical(profile))
    expected_embedding_id = _stable_embedding_id(
        row["chunk_id"],
        row["chunk_content_hash"],
        profile_hash,
        row["embedding_hash"],
    )
    if row["embedding_id"] != expected_embedding_id:
        raise IndexInputError(
            "embedding_schema_invalid",
            f"Embeddings artifact row {line_number} embedding_id does not match row identity.",
        )

    expected_provenance = {
        "chunk_id": row["chunk_id"],
        "chunk_content_hash": row["chunk_content_hash"],
        "profile_hash": profile_hash,
    }
    for field, expected_value in expected_provenance.items():
        if provenance.get(field) != expected_value:
            raise IndexInputError(
                "embedding_schema_invalid",
                f"Embeddings artifact row {line_number} has invalid provenance.{field}.",
            )

    chunks_path = provenance.get("chunks_path")
    if chunks_path is not None:
        if not isinstance(chunks_path, str):
            raise IndexInputError(
                "embedding_schema_invalid",
                f"Embeddings artifact row {line_number} has invalid chunks_path provenance.",
            )
        _validate_utf8_string(chunks_path, "provenance.chunks_path", line_number)
        if _jsonl_artifact_path_error(chunks_path):
            raise IndexInputError(
                "embedding_schema_invalid",
                f"Embeddings artifact row {line_number} has invalid chunks_path provenance.",
            )
    return profile, dimensions, chunks_path


def _validate_vector(value: Any, dimensions: int, line_number: int) -> list[float]:
    if not isinstance(value, list):
        raise IndexInputError(
            "embedding_schema_invalid",
            f"Embeddings artifact row {line_number} has invalid embedding vector.",
        )
    vector: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise IndexInputError(
                "embedding_schema_invalid",
                f"Embeddings artifact row {line_number} has non-numeric embedding vector values.",
            )
        try:
            numeric_item = float(item)
        except (OverflowError, ValueError) as error:
            raise IndexInputError(
                "embedding_schema_invalid",
                f"Embeddings artifact row {line_number} has non-finite embedding vector values.",
            ) from error
        if not isfinite(numeric_item):
            raise IndexInputError(
                "embedding_schema_invalid",
                f"Embeddings artifact row {line_number} has non-finite embedding vector values.",
            )
        vector.append(numeric_item)
    if len(vector) != dimensions:
        raise IndexInputError(
            "embedding_schema_invalid",
            f"Embeddings artifact row {line_number} has {len(vector)} dimensions; expected {dimensions}.",
        )
    return vector


def _chunk_rows_by_id(
    context: _ProjectContext,
    chunks_path: str | None,
) -> dict[str, dict[str, Any]]:
    if chunks_path is None:
        return {}
    if _jsonl_artifact_path_error(chunks_path):
        raise IndexInputError(
            "embedding_schema_invalid",
            "Embeddings artifact has invalid chunks_path provenance.",
        )
    path = _resolve_owned_input_artifact_path(
        context,
        chunks_path,
        artifact_label="Chunks artifact",
    )
    try:
        data = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise IndexInputError(
            "chunks_not_found",
            f"Chunks artifact does not exist: {chunks_path}",
            status=CommandStatus.MISSING_ARTIFACT,
        ) from error
    except UnicodeDecodeError as error:
        raise IndexInputError(
            "chunks_invalid_jsonl",
            f"Chunks artifact is not valid UTF-8: {chunks_path}",
        ) from error
    except OSError as error:
        raise IndexInputError(
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
            raise IndexInputError(
                "chunks_invalid_jsonl",
                f"Chunks artifact contains invalid JSONL at line {line_number}: {error}",
            ) from error
        if (
            not isinstance(row, dict)
            or row.get("schema_name") != "md_to_rag.chunk"
            or row.get("schema_version") != "1.0"
        ):
            raise IndexInputError(
                "chunks_invalid_jsonl",
                f"Chunks artifact row {line_number} must be an md_to_rag.chunk v1.0 object.",
            )
        chunk_id = row.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise IndexInputError(
                "chunks_invalid_jsonl",
                f"Chunks artifact row {line_number} has invalid chunk_id.",
            )
        _validate_chunk_utf8_string(chunk_id, "chunk_id", line_number)
        if chunk_id in rows:
            raise IndexInputError(
                "chunks_invalid_jsonl",
                f"Chunks artifact row {line_number} duplicates chunk_id: {chunk_id}",
            )
        content = row.get("content")
        content_hash = row.get("content_hash")
        if not isinstance(content, str) or not isinstance(content_hash, str):
            raise IndexInputError(
                "chunks_invalid_jsonl",
                f"Chunks artifact row {line_number} has invalid content or content_hash.",
            )
        _validate_chunk_utf8_string(content, "content", line_number)
        if content_hash != _hash_text(content):
            raise IndexInputError(
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
                raise IndexInputError(
                    "chunks_invalid_jsonl",
                    f"Chunks artifact row {line_number} has invalid {field}.",
                )
        _validate_source_path(
            row["source_path"],
            line_number,
            schema_name="Chunks artifact",
            code="chunks_invalid_jsonl",
        )
        chunk_index = row.get("chunk_index")
        if not isinstance(chunk_index, int) or isinstance(chunk_index, bool) or chunk_index < 0:
            raise IndexInputError(
                "chunks_invalid_jsonl",
                f"Chunks artifact row {line_number} has invalid chunk_index.",
            )
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
            raise IndexInputError(
                "chunks_invalid_jsonl",
                f"Chunks artifact row {line_number} has invalid citation fields.",
            )
        for index, heading in enumerate(heading_path):
            _validate_chunk_utf8_string(heading, f"heading_path[{index}]", line_number)
        for field in ("metadata", "provenance"):
            if not isinstance(row.get(field), dict):
                raise IndexInputError(
                    "chunks_invalid_jsonl",
                    f"Chunks artifact row {line_number} has invalid {field}.",
                )
            _validate_chunk_json_value_strings(row[field], field, line_number)
        rows[chunk_id] = row
    return rows


def _validate_chunk_json_value_strings(value: Any, field: str, line_number: int) -> None:
    if isinstance(value, str):
        _validate_chunk_utf8_string(value, field, line_number)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_chunk_json_value_strings(item, f"{field}[{index}]", line_number)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise IndexInputError(
                    "chunks_invalid_jsonl",
                    f"Chunks artifact row {line_number} has non-string key in {field}.",
                )
            _validate_chunk_utf8_string(key, f"{field} key", line_number)
            _validate_chunk_json_value_strings(item, f"{field}.{key}", line_number)
        return
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return
    if isinstance(value, float) and isfinite(value):
        return
    raise IndexInputError(
        "chunks_invalid_jsonl",
        f"Chunks artifact row {line_number} has non-portable JSON value in {field}.",
    )


def _validate_chunk_utf8_string(value: str, field: str, line_number: int) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise IndexInputError(
            "chunks_invalid_jsonl",
            f"Chunks artifact row {line_number} has invalid UTF-8 string field: {field}.",
        ) from error


def _resolve_owned_input_artifact_path(
    context: _ProjectContext,
    artifact_path: str,
    *,
    artifact_label: str,
) -> Path:
    path = context.project_root / artifact_path
    try:
        resolved_path = path.resolve()
    except (OSError, RuntimeError, ValueError) as error:
        raise IndexInputError(
            "artifact_path_collision",
            f"{artifact_label} path cannot be resolved safely: {artifact_path}",
        ) from error
    try:
        resolved_path.relative_to(context.project_root.resolve())
    except ValueError as error:
        raise IndexInputError(
            "artifact_path_outside_project",
            f"{artifact_label} path must stay inside the initialized project: {artifact_path}",
        ) from error
    nested_manifest_path = _nearest_nested_manifest(
        resolved_path,
        context.project_root,
        context.manifest_path,
    )
    if nested_manifest_path is not None:
        raise IndexInputError(
            "artifact_path_nested_project",
            f"{artifact_label} path resolves inside a nested initialized project.",
        )
    if resolved_path != path.absolute():
        raise IndexInputError(
            "artifact_path_collision",
            f"{artifact_label} path cannot be a symlink or linked path: {artifact_path}",
        )
    if path.exists() and path.stat().st_nlink > 1:
        raise IndexInputError(
            "artifact_path_collision",
            f"{artifact_label} path cannot be a hard-linked path: {artifact_path}",
        )
    return resolved_path


def _index_rows(
    embedding_rows: list[dict[str, Any]],
    context: _ProjectContext,
    chunk_rows: dict[str, dict[str, Any]],
    *,
    chunks_recorded: bool,
) -> list[dict[str, Any]]:
    if chunks_recorded:
        embedding_chunk_ids = {row["chunk_id"] for row in embedding_rows}
        extra_chunk_ids = sorted(set(chunk_rows) - embedding_chunk_ids)
        if extra_chunk_ids:
            raise IndexInputError(
                "embedding_chunk_mismatch",
                f"Chunks artifact contains unembedded chunk_id: {extra_chunk_ids[0]}",
            )
    rows: list[dict[str, Any]] = []
    for embedding_row in embedding_rows:
        chunk_row = chunk_rows.get(embedding_row["chunk_id"]) if chunks_recorded else None
        if chunks_recorded and chunk_row is None:
            raise IndexInputError(
                "embedding_chunk_mismatch",
                f"Chunks artifact is missing embedded chunk_id: {embedding_row['chunk_id']}",
            )
        if chunk_row is not None:
            reserved_provenance_keys = sorted(
                set(chunk_row.get("provenance", {})) & INDEX_PROVENANCE_KEYS
            )
            if reserved_provenance_keys:
                fields = ", ".join(reserved_provenance_keys)
                raise IndexInputError(
                    "embedding_chunk_mismatch",
                    f"Chunk provenance contains reserved index key(s): {fields}",
                )
            copied_fields = (
                "doc_id",
                "source_id",
                "source_path",
                "source_hash",
                "document_content_hash",
                "chunk_index",
            )
            if (
                any(chunk_row.get(field) != embedding_row[field] for field in copied_fields)
                or chunk_row.get("content_hash") != embedding_row["chunk_content_hash"]
                or chunk_row.get("metadata") != embedding_row["metadata"]
                or chunk_row.get("provenance") != _embedding_chunk_provenance(embedding_row)
            ):
                raise IndexInputError(
                    "embedding_chunk_mismatch",
                    f"Chunk row does not match embedding row for chunk_id: {embedding_row['chunk_id']}",
                )
        vector = [float(value) for value in embedding_row["embedding"]]
        vector_norm = sqrt(sum(value * value for value in vector))
        row = {
            "schema_name": "md_to_rag.index_vector",
            "schema_version": "1.0",
            "index_id": _stable_index_id(
                embedding_row["embedding_id"],
                embedding_row["embedding_hash"],
            ),
            "embedding_id": embedding_row["embedding_id"],
            "chunk_id": embedding_row["chunk_id"],
            "doc_id": embedding_row["doc_id"],
            "source_id": embedding_row["source_id"],
            "source_path": embedding_row["source_path"],
            "source_hash": embedding_row["source_hash"],
            "document_content_hash": embedding_row["document_content_hash"],
            "chunk_content_hash": embedding_row["chunk_content_hash"],
            "chunk_index": embedding_row["chunk_index"],
            "embedding_hash": embedding_row["embedding_hash"],
            "vector": vector,
            "vector_norm": round(vector_norm, 12),
            "metadata": embedding_row["metadata"],
            "provenance": {
                **embedding_row["provenance"],
                "embeddings_path": context.embeddings_path_relative,
                "embedding_id": embedding_row["embedding_id"],
            },
        }
        if chunk_row is not None:
            row["line_start"] = chunk_row["line_start"]
            row["line_end"] = chunk_row["line_end"]
            row["heading_path"] = list(chunk_row["heading_path"])
        rows.append(row)
    return rows


def _embedding_chunk_provenance(embedding_row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in embedding_row["provenance"].items()
        if key not in {"chunks_path", "chunk_id", "chunk_content_hash", "profile_hash"}
    }


def _index_manifest_payload(
    context: _ProjectContext,
    embedding_artifact: _EmbeddingArtifact,
    index_hash: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_name": "md_to_rag.index",
        "schema_version": "1.0",
        "index_engine": INDEX_ENGINE,
        "index_version": INDEX_VERSION,
        "embeddings_path": context.embeddings_path_relative,
        "embeddings_hash": embedding_artifact.artifact_hash,
        "embedding_count": len(embedding_artifact.rows),
        "vector_count": len(embedding_artifact.rows),
        "dimensions": embedding_artifact.dimensions,
        "profile": embedding_artifact.profile,
        "index_path": INDEX_PATH,
        "index_hash": index_hash,
    }
    if embedding_artifact.chunks_path is not None:
        payload["chunks_path"] = embedding_artifact.chunks_path
    return payload


def _update_manifest_status(
    manifest_path: Path,
    manifest: ProjectManifest,
    data: IndexResponseData,
) -> None:
    status = ManifestCommandStatus(
        command=CommandName.INDEX,
        status=CommandStatus.OK,
        message="Index artifacts generated.",
        artifact_path=INDEX_MANIFEST_PATH,
        updated_at=_utc_now(),
        data={
            "embedding_count": data.embedding_count,
            "vector_count": data.vector_count,
            "embeddings_path": data.embeddings_path,
            "index_manifest_path": data.index_manifest_path,
            "index_path": data.index_path,
            "embeddings_hash": data.embeddings_hash,
            "index_hash": data.index_hash,
            "index_manifest_hash": data.index_manifest_hash,
            "index_engine": data.index_engine,
            "dimensions": data.dimensions,
            "profile": data.profile,
            "chunks_path": data.chunks_path,
        },
    )
    command_status = []
    replaced = False
    for existing_status in manifest.command_status:
        if existing_status.command is CommandName.INDEX:
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


def _manifest_status_matches(manifest: ProjectManifest, data: IndexResponseData) -> bool:
    for existing_status in manifest.command_status:
        if existing_status.command is not CommandName.INDEX:
            continue
        return (
            existing_status.status is CommandStatus.OK
            and existing_status.artifact_path == INDEX_MANIFEST_PATH
            and existing_status.data.get("embedding_count") == data.embedding_count
            and existing_status.data.get("vector_count") == data.vector_count
            and existing_status.data.get("embeddings_path") == data.embeddings_path
            and existing_status.data.get("index_manifest_path") == data.index_manifest_path
            and existing_status.data.get("index_path") == data.index_path
            and existing_status.data.get("embeddings_hash") == data.embeddings_hash
            and existing_status.data.get("index_hash") == data.index_hash
            and existing_status.data.get("index_manifest_hash") == data.index_manifest_hash
            and existing_status.data.get("index_engine") == data.index_engine
            and existing_status.data.get("dimensions") == data.dimensions
            and existing_status.data.get("profile") == data.profile
            and existing_status.data.get("chunks_path") == data.chunks_path
        )
    return False


def _input_error_result(error: IndexInputError, context: _ProjectContext) -> IndexProjectResult:
    return IndexProjectResult(
        status=error.status,
        message=error.message,
        data=IndexErrorData(
            project_root=str(context.project_root),
            manifest_path=str(context.manifest_path),
            embeddings_path=context.embeddings_path_relative,
        ),
        error=error.to_command_error(),
    )


def _resolve_user_path(path: str | Path) -> Path:
    user_path = Path(path).expanduser()
    if user_path.is_absolute():
        return user_path
    return Path.cwd() / user_path


def _safe_response_text(value: str) -> str | None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return None
    return value


def _raw_embeddings_path_error(path: str | Path) -> IndexInputError | None:
    path_text = str(path)
    try:
        path_text.encode("utf-8")
    except UnicodeEncodeError:
        return IndexInputError(
            "embeddings_path_not_portable",
            "Embeddings artifact path must be valid UTF-8.",
        )
    windows_path = PureWindowsPath(path_text)
    if windows_path.drive and not windows_path.is_absolute():
        return IndexInputError(
            "embeddings_path_not_portable",
            f"Embeddings artifact path must be project-relative and portable: {path_text}",
        )
    return None


def _relative_to_project(path: Path, project_root: Path) -> str | IndexInputError:
    try:
        relative = path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return IndexInputError(
            "embeddings_outside_project",
            f"Embeddings artifact must be inside the initialized project: {path}",
        )
    return relative.as_posix()


def _embeddings_artifact_path_error(value: str) -> IndexInputError | None:
    path_error = _jsonl_artifact_path_error(value)
    if path_error:
        return IndexInputError(
            "embeddings_path_not_portable",
            f"Embeddings artifact path must be project-relative and portable: {value}",
        )
    if value.casefold() == INDEX_PATH.casefold():
        return IndexInputError(
            "embeddings_artifact_collision",
            f"Embeddings artifact cannot be a generated md_to_rag index artifact: {value}",
        )
    return None


def _jsonl_artifact_path_error(value: str) -> bool:
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
        or posix_path.suffix.lower() != ".jsonl"
    )


def _reject_output_path_outside_project(
    context: _ProjectContext,
    output_path: Path,
    artifact_path: str,
) -> None:
    try:
        resolved_output = output_path.resolve()
    except (OSError, RuntimeError, ValueError) as error:
        raise IndexInputError(
            "artifact_path_collision",
            f"Generated artifact path cannot be resolved safely: {artifact_path}",
        ) from error
    relative = _relative_to_project(resolved_output, context.project_root)
    if isinstance(relative, IndexInputError):
        raise IndexInputError(
            "artifact_path_outside_project",
            f"Generated artifact path must stay inside the initialized project: {artifact_path}",
        )
    nested_manifest_path = _nearest_nested_manifest(
        resolved_output,
        context.project_root,
        context.manifest_path,
    )
    if nested_manifest_path is not None:
        raise IndexInputError(
            "artifact_path_nested_project",
            "Generated index artifact path resolves inside a nested initialized project; "
            "use that project path directly.",
        )
    if resolved_output != output_path.absolute():
        raise IndexInputError(
            "artifact_path_collision",
            f"Generated artifact path cannot be a symlink or linked path: {artifact_path}",
        )
    if output_path.exists() and output_path.stat().st_nlink > 1:
        raise IndexInputError(
            "artifact_path_collision",
            f"Generated artifact path cannot be a hard-linked path: {artifact_path}",
        )


def _validate_utf8_string(value: str, field: str, line_number: int) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise IndexInputError(
            "embedding_schema_invalid",
            f"Embeddings artifact row {line_number} has invalid UTF-8 string field: {field}.",
        ) from error


def _validate_source_path(
    value: str,
    line_number: int,
    *,
    schema_name: str = "Embeddings artifact",
    code: str = "embedding_schema_invalid",
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
        raise IndexInputError(
            code,
            f"{schema_name} row {line_number} has non-portable source_path.",
        )


def _validate_json_value_strings(value: Any, field: str, line_number: int) -> None:
    if isinstance(value, str):
        _validate_utf8_string(value, field, line_number)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value_strings(item, f"{field}[{index}]", line_number)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise IndexInputError(
                    "embedding_schema_invalid",
                    f"Embeddings artifact row {line_number} has non-string key in {field}.",
                )
            _validate_utf8_string(key, f"{field} key", line_number)
            _validate_json_value_strings(item, f"{field}.{key}", line_number)
        return
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return
    if isinstance(value, float) and isfinite(value):
        return
    raise IndexInputError(
        "embedding_schema_invalid",
        f"Embeddings artifact row {line_number} has non-portable JSON value in {field}.",
    )


def _has_windows_reserved_path_component(part: str) -> bool:
    normalized = part.rstrip(" .")
    basename = normalized.split(".", 1)[0].upper()
    return (
        any(character in part for character in '<>:"|?*')
        or any(ord(character) < 32 for character in part)
        or normalized != part
        or basename in WINDOWS_RESERVED_BASENAMES
    )


def _stable_embedding_id(
    chunk_id: str,
    chunk_content_hash: str,
    profile_hash: str,
    embedding_hash: str,
) -> str:
    value = "\n".join([chunk_id, chunk_content_hash, profile_hash, embedding_hash])
    return f"emb_{sha256(value.encode('utf-8')).hexdigest()[:16]}"


def _stable_index_id(embedding_id: str, embedding_hash: str) -> str:
    value = "\n".join([embedding_id, embedding_hash])
    return f"idx_{sha256(value.encode('utf-8')).hexdigest()[:16]}"


def _json_loads_strict(text: str) -> Any:
    return json.loads(
        text,
        parse_constant=_reject_json_constant,
        parse_float=_json_float_strict,
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value is not supported: {value}")


def _json_float_strict(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed):
        raise ValueError(f"non-finite JSON value is not supported: {value}")
    return parsed


def _json_dumps_canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise IndexInputError(
            "embedding_schema_invalid",
            f"Embeddings artifact must contain portable JSON values: {error}",
        ) from error


def _jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    try:
        return "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            for row in rows
        )
    except ValueError as error:
        raise IndexInputError(
            "embedding_schema_invalid",
            f"Index artifacts must be portable JSONL with finite values: {error}",
        ) from error


def _json_text(value: dict[str, Any]) -> str:
    try:
        return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except ValueError as error:
        raise IndexInputError(
            "embedding_schema_invalid",
            f"Index manifest must be portable JSON with finite values: {error}",
        ) from error
