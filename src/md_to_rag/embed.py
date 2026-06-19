from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable

from .chunk import _hash_bytes, _hash_text, _write_if_changed
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
    EmbedErrorData,
    EmbedResponseData,
    ManifestCommandStatus,
    ProjectManifest,
)


CHUNKS_PATH = "chunks/chunks.jsonl"
EMBEDDINGS_PATH = "embeddings/embeddings.jsonl"
_CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_REDACTED_PROFILE_SECRET = "<redacted>"
_HEADER_METADATA_FIELDS = {"active", "disabled", "enabled", "optional", "required"}
_HEADER_NAME_FIELDS = {"field", "field_name", "header", "header_name", "key", "name"}
_NON_SECRET_PROFILE_KEYS = {
    "input_token_limit",
    "max_tokens",
    "output_token_limit",
    "token_limit",
    "tokenizer",
}


@dataclass(frozen=True)
class EmbedProjectResult:
    status: CommandStatus
    message: str
    data: EmbedResponseData | EmbedErrorData
    artifact_path: str | None = None
    error: CommandError | None = None


@dataclass(frozen=True)
class _ProjectContext:
    project_root: Path
    manifest_path: Path
    manifest: ProjectManifest
    chunks_path: Path
    chunks_path_relative: str


@dataclass(frozen=True)
class DeterministicHashEmbeddingProvider:
    model: str = "deterministic-hash-v1"
    dimensions: int = 8
    version: str = "1.0"
    options: dict[str, Any] = field(default_factory=dict)
    provider: str = "md_to_rag.local_hash"

    def profile(self) -> dict[str, Any]:
        profile: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "dimensions": self.dimensions,
            "version": self.version,
        }
        if self.options:
            profile["options"] = self.options
        return profile

    def embed(self, content: str, *, chunk_id: str, content_hash: str) -> list[float]:
        profile = _provider_profile(self)
        dimensions = profile["dimensions"]
        if dimensions < 1:
            raise EmbedInputError(
                "embedding_profile_invalid",
                "Embedding profile dimensions must be a positive integer.",
            )
        profile_text = _json_dumps_canonical(profile)
        seed = "\n".join([profile_text, chunk_id, content_hash, content]).encode("utf-8")
        vector: list[float] = []
        counter = 0
        while len(vector) < dimensions:
            digest = sha256(seed + counter.to_bytes(4, "big")).digest()
            for offset in range(0, len(digest), 4):
                if len(vector) >= dimensions:
                    break
                integer = int.from_bytes(digest[offset:offset + 4], "big")
                vector.append(round((integer / 0xFFFFFFFF) * 2 - 1, 10))
            counter += 1
        return vector


class _RecordedProfileDeterministicHashEmbeddingProvider(
    DeterministicHashEmbeddingProvider
):
    def __init__(self, profile: dict[str, Any]) -> None:
        recorded_profile = json.loads(_json_dumps_canonical(profile))
        super().__init__(
            provider=recorded_profile["provider"],
            model=recorded_profile["model"],
            dimensions=recorded_profile["dimensions"],
            version=recorded_profile["version"],
            options=recorded_profile.get("options", {}),
        )
        object.__setattr__(self, "_recorded_profile", recorded_profile)

    def profile(self) -> dict[str, Any]:
        return json.loads(_json_dumps_canonical(self._recorded_profile))


class EmbedInputError(Exception):
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


def embed_project(
    chunks: str | Path | None = None,
    *,
    provider: DeterministicHashEmbeddingProvider | None = None,
) -> EmbedProjectResult:
    context_result = _resolve_context(chunks)
    if isinstance(context_result, EmbedProjectResult):
        return context_result
    context = context_result
    if provider is None:
        provider = DeterministicHashEmbeddingProvider()

    try:
        embeddings_path = context.project_root / EMBEDDINGS_PATH
        _reject_output_path_outside_project(context, embeddings_path, EMBEDDINGS_PATH)
        chunk_rows, chunks_hash = _read_chunk_rows(context.chunks_path)
        profile = _provider_profile(provider)
        cached_data = _cached_data_if_unchanged(
            context,
            embeddings_path,
            chunks_hash,
            profile,
            chunk_count=len(chunk_rows),
        )
        if cached_data is not None:
            return EmbedProjectResult(
                status=CommandStatus.OK,
                message="Embedding artifacts unchanged.",
                data=cached_data,
                artifact_path=str(embeddings_path.resolve()),
            )
        embedding_rows = _embedding_rows(chunk_rows, provider, profile, context)
        embeddings_text = _jsonl_text(embedding_rows)
        embeddings_changed = _write_if_changed(embeddings_path, embeddings_text)
    except EmbedInputError as error:
        return _input_error_result(error, context)
    except OSError as error:
        embed_error = EmbedInputError(
            "embed_io_failed",
            f"Could not generate embedding artifacts: {error}",
        )
        return _input_error_result(embed_error, context)

    data = EmbedResponseData(
        project_root=str(context.project_root),
        manifest_path=str(context.manifest_path),
        chunks_path=context.chunks_path_relative,
        changed=False,
        chunk_count=len(chunk_rows),
        embedding_count=len(embedding_rows),
        embeddings_path=EMBEDDINGS_PATH,
        chunks_hash=chunks_hash,
        embeddings_hash=_hash_text(embeddings_text),
        profile=profile,
    )
    manifest_status_changed = not _manifest_status_matches(context.manifest, data)
    changed = embeddings_changed or manifest_status_changed
    data = data.model_copy(update={"changed": changed})

    if changed:
        try:
            _update_manifest_status(context.manifest_path, context.manifest, data)
        except ManifestWriteError as error:
            return EmbedProjectResult(
                status=CommandStatus.ERROR,
                message=error.message,
                data=data,
                artifact_path=str(embeddings_path.resolve()),
                error=error.to_command_error(),
            )
        message = "Embedding artifacts generated."
    else:
        message = "Embedding artifacts unchanged."

    return EmbedProjectResult(
        status=CommandStatus.OK,
        message=message,
        data=data,
        artifact_path=str(embeddings_path.resolve()),
    )


def _resolve_context(chunks: str | Path | None) -> _ProjectContext | EmbedProjectResult:
    if chunks is None:
        manifest_path = _find_manifest_lexical(Path.cwd())
        requested_chunks_path: Path | None = None
    else:
        path_error = _raw_chunks_path_error(chunks)
        if path_error is not None:
            return EmbedProjectResult(
                status=path_error.status,
                message=path_error.message,
                data=EmbedErrorData(chunks_path=str(chunks)),
                error=path_error.to_command_error(),
            )
        requested_chunks_path = _resolve_user_path(chunks)
        anchor = (
            requested_chunks_path
            if requested_chunks_path.exists()
            else _nearest_existing_ancestor(requested_chunks_path)
        )
        manifest_path = _find_manifest_lexical(anchor)

    if manifest_path is None:
        return EmbedProjectResult(
            status=CommandStatus.MISSING_ARTIFACT,
            message=f"No {MANIFEST_FILENAME} found for chunks artifact.",
            data=EmbedErrorData(
                chunks_path=str(_resolve_user_path(chunks)) if chunks is not None else None,
            ),
            error=CommandError(
                code="manifest_not_found",
                message=f"No {MANIFEST_FILENAME} found for chunks artifact.",
            ),
        )

    try:
        project_manifest = _read_manifest(manifest_path)
    except ManifestReadError as error:
        project_root = manifest_path.parent.resolve()
        return EmbedProjectResult(
            status=CommandStatus.ERROR,
            message=error.message,
            data=EmbedErrorData(
                project_root=str(project_root),
                manifest_path=str(manifest_path.resolve()),
                chunks_path=str(chunks) if chunks is not None else None,
            ),
            error=error.to_command_error(),
        )

    project_root = manifest_path.parent.resolve()
    if requested_chunks_path is None:
        requested_chunks_path = project_root / CHUNKS_PATH

    try:
        chunks_path = requested_chunks_path.resolve()
    except (OSError, RuntimeError, ValueError) as error:
        message = f"Could not resolve chunks artifact: {requested_chunks_path}"
        return EmbedProjectResult(
            status=CommandStatus.ERROR,
            message=message,
            data=EmbedErrorData(
                project_root=str(project_root),
                manifest_path=str(manifest_path.resolve()),
                chunks_path=str(requested_chunks_path),
            ),
            error=CommandError(
                code="chunks_path_unresolvable",
                message=f"{message}: {error}",
            ),
        )

    chunks_relative = _relative_to_project(chunks_path, project_root)
    if isinstance(chunks_relative, EmbedInputError):
        return _input_error_result(
            chunks_relative,
            _ProjectContext(
                project_root=project_root,
                manifest_path=manifest_path.resolve(),
                manifest=project_manifest,
                chunks_path=chunks_path,
                chunks_path_relative=str(chunks_path),
            ),
        )
    portability_error = _chunks_artifact_path_error(chunks_relative)
    if portability_error is not None:
        return _input_error_result(
            portability_error,
            _ProjectContext(
                project_root=project_root,
                manifest_path=manifest_path.resolve(),
                manifest=project_manifest,
                chunks_path=chunks_path,
                chunks_path_relative=chunks_relative,
            ),
        )
    if chunks_relative.casefold() == EMBEDDINGS_PATH.casefold():
        return _input_error_result(
            EmbedInputError(
                "chunks_artifact_collision",
                f"Chunks artifact cannot be a generated md_to_rag embedding artifact: {chunks_relative}",
            ),
            _ProjectContext(
                project_root=project_root,
                manifest_path=manifest_path.resolve(),
                manifest=project_manifest,
                chunks_path=chunks_path,
                chunks_path_relative=chunks_relative,
            ),
        )
    nested_manifest_path = _nearest_nested_manifest(
        chunks_path,
        project_root,
        manifest_path,
    )
    if nested_manifest_path is not None:
        return _input_error_result(
            EmbedInputError(
                "chunks_nested_project",
                "Chunks artifact resolves inside a nested initialized project; "
                "use that project path directly.",
            ),
            _ProjectContext(
                project_root=project_root,
                manifest_path=manifest_path.resolve(),
                manifest=project_manifest,
                chunks_path=chunks_path,
                chunks_path_relative=chunks_relative,
            ),
        )

    if not chunks_path.exists():
        return EmbedProjectResult(
            status=CommandStatus.MISSING_ARTIFACT,
            message=f"Chunks artifact does not exist: {chunks_relative}",
            data=EmbedErrorData(
                project_root=str(project_root),
                manifest_path=str(manifest_path.resolve()),
                chunks_path=chunks_relative,
            ),
            error=CommandError(
                code="chunks_not_found",
                message=f"Chunks artifact does not exist: {chunks_relative}",
            ),
        )

    return _ProjectContext(
        project_root=project_root,
        manifest_path=manifest_path.resolve(),
        manifest=project_manifest,
        chunks_path=chunks_path,
        chunks_path_relative=chunks_relative,
    )


def _read_chunk_rows(chunks_path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        data = chunks_path.read_bytes()
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EmbedInputError(
            "chunks_invalid_jsonl",
            f"Chunks artifact is not valid UTF-8: {chunks_path}",
        ) from error
    except OSError as error:
        raise EmbedInputError(
            "chunks_read_failed",
            f"Could not read chunks artifact {chunks_path}: {error}",
        ) from error

    rows: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = _json_loads_strict(line)
        except (json.JSONDecodeError, ValueError) as error:
            raise EmbedInputError(
                "chunks_invalid_jsonl",
                f"Chunks artifact contains invalid JSONL at line {line_number}: {error}",
            ) from error
        if not isinstance(row, dict):
            raise EmbedInputError(
                "chunks_invalid_jsonl",
                f"Chunks artifact row {line_number} must be a JSON object.",
            )
        _validate_chunk_row(row, line_number)
        chunk_id = row["chunk_id"]
        if chunk_id in seen_chunk_ids:
            raise EmbedInputError(
                "duplicate_chunk_id",
                f"Chunks artifact contains duplicate chunk_id at line {line_number}: {chunk_id}",
            )
        seen_chunk_ids.add(chunk_id)
        rows.append(row)
    return rows, _hash_bytes(data)


def _validate_chunk_row(row: dict[str, Any], line_number: int) -> None:
    if row.get("schema_name") != "md_to_rag.chunk" or row.get("schema_version") != "1.0":
        raise EmbedInputError(
            "chunk_schema_invalid",
            f"Chunks artifact row {line_number} is not an md_to_rag.chunk v1.0 row.",
        )

    required_strings = (
        "chunk_id",
        "doc_id",
        "source_id",
        "source_path",
        "source_hash",
        "content_hash",
        "document_content_hash",
        "content",
    )
    invalid_string_fields = [
        field
        for field in required_strings
        if not isinstance(row.get(field), str)
        or (field != "content" and not row[field])
    ]
    if invalid_string_fields:
        fields = ", ".join(invalid_string_fields)
        raise EmbedInputError(
            "chunk_schema_invalid",
            f"Chunks artifact row {line_number} has invalid string field(s): {fields}.",
        )
    for field in required_strings:
        _validate_utf8_string(row[field], field, line_number)
    _validate_source_path(row["source_path"], line_number)
    if row["content_hash"] != _hash_text(row["content"]):
        raise EmbedInputError(
            "chunk_schema_invalid",
            f"Chunks artifact row {line_number} content_hash does not match content.",
        )
    for field in ("chunk_index", "line_start", "line_end"):
        if not isinstance(row.get(field), int) or isinstance(row[field], bool):
            raise EmbedInputError(
                "chunk_schema_invalid",
                f"Chunks artifact row {line_number} has invalid {field}.",
            )
    if row["chunk_index"] < 0 or row["line_start"] < 1 or row["line_end"] < row["line_start"]:
        raise EmbedInputError(
            "chunk_schema_invalid",
            f"Chunks artifact row {line_number} has invalid line or index values.",
        )
    if not isinstance(row.get("metadata"), dict):
        raise EmbedInputError(
            "chunk_schema_invalid",
            f"Chunks artifact row {line_number} has invalid metadata.",
        )
    if not isinstance(row.get("provenance"), dict):
        raise EmbedInputError(
            "chunk_schema_invalid",
            f"Chunks artifact row {line_number} has invalid provenance.",
        )
    heading_path = row.get("heading_path")
    if not isinstance(heading_path, list) or not all(
        isinstance(item, str) for item in heading_path
    ):
        raise EmbedInputError(
            "chunk_schema_invalid",
            f"Chunks artifact row {line_number} has invalid heading_path.",
        )
    for item in heading_path:
        _validate_utf8_string(item, "heading_path", line_number)


def _embedding_rows(
    chunk_rows: Iterable[dict[str, Any]],
    provider: DeterministicHashEmbeddingProvider,
    profile: dict[str, Any],
    context: _ProjectContext,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    profile_hash = _hash_text(_json_dumps_canonical(profile))
    for chunk_row in chunk_rows:
        try:
            provider_vector = provider.embed(
                chunk_row["content"],
                chunk_id=chunk_row["chunk_id"],
                content_hash=chunk_row["content_hash"],
            )
        except EmbedInputError as error:
            raise EmbedInputError(
                "embedding_provider_failed",
                f"Embedding provider failed for chunk_id {chunk_row['chunk_id']}.",
            ) from error
        except Exception as error:
            raise EmbedInputError(
                "embedding_provider_failed",
                f"Embedding provider failed for chunk_id {chunk_row['chunk_id']}.",
            ) from error
        vector = _validate_embedding_vector(
            provider_vector,
            profile,
            chunk_row["chunk_id"],
        )
        vector_text = _json_dumps_canonical(vector)
        embedding_hash = _hash_text(vector_text)
        embedding_id = _stable_embedding_id(
            chunk_row["chunk_id"],
            chunk_row["content_hash"],
            profile_hash,
            embedding_hash,
        )
        rows.append(
            {
                "schema_name": "md_to_rag.embedding",
                "schema_version": "1.0",
                "embedding_id": embedding_id,
                "chunk_id": chunk_row["chunk_id"],
                "doc_id": chunk_row["doc_id"],
                "source_id": chunk_row["source_id"],
                "source_path": chunk_row["source_path"],
                "source_hash": chunk_row["source_hash"],
                "document_content_hash": chunk_row["document_content_hash"],
                "chunk_content_hash": chunk_row["content_hash"],
                "chunk_index": chunk_row["chunk_index"],
                "embedding": vector,
                "embedding_hash": embedding_hash,
                "profile": profile,
                "metadata": chunk_row["metadata"],
                "provenance": {
                    **chunk_row["provenance"],
                    "chunks_path": context.chunks_path_relative,
                    "chunk_id": chunk_row["chunk_id"],
                    "chunk_content_hash": chunk_row["content_hash"],
                    "profile_hash": profile_hash,
                },
            }
        )
    return rows


def _provider_profile(provider: DeterministicHashEmbeddingProvider) -> dict[str, Any]:
    try:
        profile = provider.profile()
    except EmbedInputError as error:
        raise EmbedInputError(
            "embedding_provider_failed",
            "Embedding provider profile failed.",
        ) from error
    except Exception as error:
        raise EmbedInputError(
            "embedding_provider_failed",
            "Embedding provider profile failed.",
        ) from error
    if not isinstance(profile, dict):
        raise EmbedInputError(
            "embedding_profile_invalid",
            "Embedding provider profile must be a JSON object.",
        )
    profile = _sanitize_profile_value(profile)
    if not isinstance(profile, dict):
        raise EmbedInputError(
            "embedding_profile_invalid",
            "Embedding provider profile must be a JSON object.",
        )
    dimensions = profile.get("dimensions")
    if (
        not isinstance(dimensions, int)
        or isinstance(dimensions, bool)
        or dimensions < 1
    ):
        raise EmbedInputError(
            "embedding_profile_invalid",
            "Embedding profile dimensions must be a positive integer.",
        )
    return json.loads(_json_dumps_canonical(profile))


def _validate_embedding_vector(
    value: Any,
    profile: dict[str, Any],
    chunk_id: str,
) -> list[float]:
    if not isinstance(value, list):
        raise EmbedInputError(
            "embedding_vector_invalid",
            f"Embedding provider returned a non-list vector for chunk_id {chunk_id}.",
        )
    vector: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise EmbedInputError(
                "embedding_vector_invalid",
                f"Embedding provider returned a non-finite numeric vector for chunk_id {chunk_id}.",
            )
        try:
            numeric_item = float(item)
        except (OverflowError, ValueError) as error:
            raise EmbedInputError(
                "embedding_vector_invalid",
                f"Embedding provider returned a non-finite numeric vector for chunk_id {chunk_id}.",
            ) from error
        if not isfinite(numeric_item):
            raise EmbedInputError(
                "embedding_vector_invalid",
                f"Embedding provider returned a non-finite numeric vector for chunk_id {chunk_id}.",
            )
        vector.append(numeric_item)
    dimensions = profile.get("dimensions")
    if isinstance(dimensions, int) and len(vector) != dimensions:
        raise EmbedInputError(
            "embedding_vector_invalid",
            f"Embedding provider returned {len(vector)} dimensions for chunk_id {chunk_id}; expected {dimensions}.",
        )
    return vector


def _update_manifest_status(
    manifest_path: Path,
    manifest: ProjectManifest,
    data: EmbedResponseData,
) -> None:
    status = ManifestCommandStatus(
        command=CommandName.EMBED,
        status=CommandStatus.OK,
        message="Embedding artifacts generated.",
        artifact_path=EMBEDDINGS_PATH,
        updated_at=_utc_now(),
        data={
            "chunk_count": data.chunk_count,
            "embedding_count": data.embedding_count,
            "chunks_path": data.chunks_path,
            "embeddings_path": data.embeddings_path,
            "chunks_hash": data.chunks_hash,
            "embeddings_hash": data.embeddings_hash,
            "profile": data.profile,
        },
    )
    command_status = []
    replaced = False
    for existing_status in manifest.command_status:
        if existing_status.command is CommandName.EMBED:
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


def _manifest_status_matches(manifest: ProjectManifest, data: EmbedResponseData) -> bool:
    for existing_status in manifest.command_status:
        if existing_status.command is not CommandName.EMBED:
            continue
        return (
            existing_status.status is CommandStatus.OK
            and existing_status.artifact_path == EMBEDDINGS_PATH
            and existing_status.data.get("chunk_count") == data.chunk_count
            and existing_status.data.get("embedding_count") == data.embedding_count
            and existing_status.data.get("chunks_path") == data.chunks_path
            and existing_status.data.get("embeddings_path") == data.embeddings_path
            and existing_status.data.get("chunks_hash") == data.chunks_hash
            and existing_status.data.get("embeddings_hash") == data.embeddings_hash
            and _profile_values_match(existing_status.data.get("profile"), data.profile)
        )
    return False


def _cached_data_if_unchanged(
    context: _ProjectContext,
    embeddings_path: Path,
    chunks_hash: str,
    profile: dict[str, Any],
    *,
    chunk_count: int,
) -> EmbedResponseData | None:
    existing_status = next(
        (
            status
            for status in context.manifest.command_status
            if status.command is CommandName.EMBED
        ),
        None,
    )
    if existing_status is None or existing_status.status is not CommandStatus.OK:
        return None
    data = existing_status.data
    if (
        existing_status.artifact_path != EMBEDDINGS_PATH
        or data.get("chunks_path") != context.chunks_path_relative
        or data.get("embeddings_path") != EMBEDDINGS_PATH
        or data.get("chunks_hash") != chunks_hash
        or not _profile_values_match(data.get("profile"), profile)
        or not isinstance(data.get("embeddings_hash"), str)
    ):
        return None
    try:
        existing_embeddings_hash = _hash_bytes(embeddings_path.read_bytes())
    except OSError:
        return None
    if existing_embeddings_hash != data["embeddings_hash"]:
        return None
    if (
        not isinstance(data.get("chunk_count"), int)
        or isinstance(data["chunk_count"], bool)
        or data["chunk_count"] != chunk_count
    ):
        return None
    if (
        not isinstance(data.get("embedding_count"), int)
        or isinstance(data["embedding_count"], bool)
        or data["embedding_count"] != chunk_count
    ):
        return None
    return EmbedResponseData(
        project_root=str(context.project_root),
        manifest_path=str(context.manifest_path),
        chunks_path=context.chunks_path_relative,
        changed=False,
        chunk_count=chunk_count,
        embedding_count=chunk_count,
        embeddings_path=EMBEDDINGS_PATH,
        chunks_hash=chunks_hash,
        embeddings_hash=existing_embeddings_hash,
        profile=profile,
    )


def _profile_values_match(left: Any, right: dict[str, Any]) -> bool:
    if not isinstance(left, dict):
        return False
    try:
        return _json_dumps_canonical(left) == _json_dumps_canonical(right)
    except EmbedInputError:
        return False


def _input_error_result(error: EmbedInputError, context: _ProjectContext) -> EmbedProjectResult:
    return EmbedProjectResult(
        status=error.status,
        message=error.message,
        data=EmbedErrorData(
            project_root=str(context.project_root),
            manifest_path=str(context.manifest_path),
            chunks_path=context.chunks_path_relative,
        ),
        error=error.to_command_error(),
    )


def _resolve_user_path(path: str | Path) -> Path:
    user_path = Path(path).expanduser()
    if user_path.is_absolute():
        return user_path
    return Path.cwd() / user_path


def _raw_chunks_path_error(path: str | Path) -> EmbedInputError | None:
    path_text = str(path)
    windows_path = PureWindowsPath(path_text)
    if windows_path.drive and not windows_path.is_absolute():
        return EmbedInputError(
            "chunks_path_not_portable",
            f"Chunks artifact path must be project-relative and portable: {path_text}",
        )
    return None


def _relative_to_project(path: Path, project_root: Path) -> str | EmbedInputError:
    try:
        relative = path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return EmbedInputError(
            "chunks_outside_project",
            f"Chunks artifact must be inside the initialized project: {path}",
        )
    return relative.as_posix()


def _chunks_artifact_path_error(value: str) -> EmbedInputError | None:
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
        or posix_path.suffix.lower() != ".jsonl"
    ):
        return EmbedInputError(
            "chunks_path_not_portable",
            f"Chunks artifact path must be project-relative and portable: {value}",
        )
    return None


def _reject_output_path_outside_project(
    context: _ProjectContext,
    output_path: Path,
    artifact_path: str,
) -> None:
    try:
        resolved_output = output_path.resolve()
    except (OSError, RuntimeError, ValueError) as error:
        raise EmbedInputError(
            "artifact_path_collision",
            f"Generated artifact path cannot be resolved safely: {artifact_path}",
        ) from error
    relative = _relative_to_project(resolved_output, context.project_root)
    if isinstance(relative, EmbedInputError):
        raise EmbedInputError(
            "artifact_path_outside_project",
            f"Generated artifact path must stay inside the initialized project: {artifact_path}",
        )
    nested_manifest_path = _nearest_nested_manifest(
        resolved_output,
        context.project_root,
        context.manifest_path,
    )
    if nested_manifest_path is not None:
        raise EmbedInputError(
            "artifact_path_nested_project",
            "Generated embedding artifact path resolves inside a nested initialized project; "
            "use that project path directly.",
        )
    if resolved_output != output_path.absolute():
        raise EmbedInputError(
            "artifact_path_collision",
            f"Generated artifact path cannot be a symlink or linked path: {artifact_path}",
        )
    if output_path.exists() and output_path.stat().st_nlink > 1:
        raise EmbedInputError(
            "artifact_path_collision",
            f"Generated artifact path cannot be a hard-linked path: {artifact_path}",
        )


def _validate_utf8_string(value: str, field: str, line_number: int) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise EmbedInputError(
            "chunk_schema_invalid",
            f"Chunks artifact row {line_number} has invalid UTF-8 string field: {field}.",
        ) from error


def _validate_source_path(value: str, line_number: int) -> None:
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
        raise EmbedInputError(
            "chunk_schema_invalid",
            f"Chunks artifact row {line_number} has non-portable source_path.",
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


def _sanitize_profile_value(
    value: Any,
    key_path: tuple[str, ...] = (),
    active_containers: set[int] | None = None,
) -> Any:
    if active_containers is None:
        active_containers = set()
    if isinstance(value, dict):
        if _is_header_collection_path(key_path):
            header_names = _header_object_names(value)
            if header_names:
                return _sanitize_header_object(
                    value,
                    key_path,
                    header_names,
                    active_containers,
                )
            return _sanitize_header_mapping(value, key_path, active_containers)
        identity = id(value)
        if identity in active_containers:
            raise EmbedInputError(
                "embedding_profile_invalid",
                "Embedding profile values must be portable JSON.",
            )
        active_containers.add(identity)
        sanitized: dict[str, Any] = {}
        try:
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0])):
                if not isinstance(key, str):
                    raise EmbedInputError(
                        "embedding_profile_invalid",
                        "Embedding profile keys must be strings.",
                    )
                _validate_profile_string(key, "key")
                child_key_path = (*key_path, key)
                if _is_secret_key(key) or _is_secret_key_path(child_key_path):
                    continue
                sanitized[key] = _sanitize_profile_value(
                    item,
                    child_key_path,
                    active_containers,
                )
            return sanitized
        finally:
            active_containers.remove(identity)
    if isinstance(value, list):
        identity = id(value)
        if identity in active_containers:
            raise EmbedInputError(
                "embedding_profile_invalid",
                "Embedding profile values must be portable JSON.",
            )
        active_containers.add(identity)
        try:
            if _is_header_collection_path(key_path):
                return [
                    _sanitize_header_list_item(item, key_path, active_containers)
                    for item in value
                ]
            return [
                _sanitize_profile_value(item, key_path, active_containers)
                for item in value
            ]
        finally:
            active_containers.remove(identity)
    if isinstance(value, str):
        _validate_profile_string(value, "value")
        if _should_redact_header_scalar(value, key_path):
            return _REDACTED_PROFILE_SECRET
        return value
    if isinstance(value, (int, bool)) or value is None:
        if _is_header_value_path(key_path):
            return _REDACTED_PROFILE_SECRET
        return value
    if isinstance(value, float) and isfinite(value):
        if _is_header_value_path(key_path):
            return _REDACTED_PROFILE_SECRET
        return value
    raise EmbedInputError(
        "embedding_profile_invalid",
        "Embedding profile values must be portable JSON.",
    )


def _sanitize_header_list_item(
    item: Any,
    key_path: tuple[str, ...],
    active_containers: set[int],
) -> Any:
    if isinstance(item, list) and len(item) >= 2 and isinstance(item[0], str):
        header_name = item[0]
        _validate_profile_string(header_name, "key")
        header_key_path = (*key_path, header_name)
        if _is_secret_key(header_name) or _is_secret_key_path(header_key_path):
            return [
                header_name,
                _REDACTED_PROFILE_SECRET,
                *(_sanitize_secret_header_extra(value) for value in item[2:]),
            ]
        return [
            header_name,
            *(
                _sanitize_profile_value(value, header_key_path, active_containers)
                for value in item[1:]
            ),
        ]
    if isinstance(item, dict):
        header_names = _header_object_names(item)
        if header_names:
            return _sanitize_header_object(
                item,
                key_path,
                header_names,
                active_containers,
            )
    return _sanitize_profile_value(item, key_path, active_containers)


def _sanitize_header_object(
    item: dict[Any, Any],
    key_path: tuple[str, ...],
    header_names: list[str],
    active_containers: set[int],
) -> dict[str, Any]:
    identity = id(item)
    if identity in active_containers:
        raise EmbedInputError(
            "embedding_profile_invalid",
            "Embedding profile values must be portable JSON.",
        )
    active_containers.add(identity)
    secret_header = any(
        _is_secret_key(header_name)
        or _is_secret_key_path((*key_path, header_name))
        for header_name in header_names
    )
    sanitized: dict[str, Any] = {}
    try:
        for key, value in sorted(item.items(), key=lambda entry: str(entry[0])):
            if not isinstance(key, str):
                raise EmbedInputError(
                    "embedding_profile_invalid",
                    "Embedding profile keys must be strings.",
                )
            _validate_profile_string(key, "key")
            child_key_path = (*key_path, key)
            if _is_header_name_field(key):
                sanitized[key] = _sanitize_header_name_value(
                    value,
                    child_key_path,
                    active_containers,
                )
                continue
            if secret_header:
                if _is_header_metadata_field(key):
                    sanitized[key] = _sanitize_profile_value(
                        value,
                        child_key_path,
                        active_containers,
                    )
                else:
                    sanitized[key] = _REDACTED_PROFILE_SECRET
                continue
            if _is_header_value_field(key):
                if isinstance(value, str) and _is_secret_key(value):
                    sanitized[key] = _REDACTED_PROFILE_SECRET
                else:
                    sanitized[key] = _sanitize_profile_value(value, (), active_containers)
                continue
            if _is_secret_key(key) or _is_secret_key_path(child_key_path):
                continue
            sanitized[key] = _sanitize_profile_value(
                value,
                child_key_path,
                active_containers,
            )
        return sanitized
    finally:
        active_containers.remove(identity)


def _sanitize_header_mapping(
    item: dict[Any, Any],
    key_path: tuple[str, ...],
    active_containers: set[int],
) -> dict[str, Any]:
    identity = id(item)
    if identity in active_containers:
        raise EmbedInputError(
            "embedding_profile_invalid",
            "Embedding profile values must be portable JSON.",
        )
    active_containers.add(identity)
    sanitized: dict[str, Any] = {}
    try:
        for key, value in sorted(item.items(), key=lambda entry: str(entry[0])):
            if not isinstance(key, str):
                raise EmbedInputError(
                    "embedding_profile_invalid",
                    "Embedding profile keys must be strings.",
                )
            _validate_profile_string(key, "key")
            child_key_path = (*key_path, key)
            if _is_secret_key(key) or _is_secret_key_path(child_key_path):
                continue
            scalar_value = isinstance(value, (str, int, bool, float)) or value is None
            sanitized[key] = _sanitize_profile_value(
                value,
                child_key_path if scalar_value else key_path,
                active_containers,
            )
        return sanitized
    finally:
        active_containers.remove(identity)


def _header_object_names(item: dict[Any, Any]) -> list[str]:
    names: list[str] = []
    for key, value in item.items():
        if (
            isinstance(key, str)
            and _is_header_name_field(key)
            and isinstance(value, str)
        ):
            _validate_profile_string(value, "value")
            names.append(value)
    return names


def _sanitize_header_name_value(
    value: Any,
    key_path: tuple[str, ...],
    active_containers: set[int],
) -> Any:
    if isinstance(value, str):
        _validate_profile_string(value, "value")
        return value
    return _sanitize_profile_value(value, key_path, active_containers)


def _is_header_name_field(key: str) -> bool:
    return _normalize_profile_key(key) in _HEADER_NAME_FIELDS


def _is_header_metadata_field(key: str) -> bool:
    return _normalize_profile_key(key) in _HEADER_METADATA_FIELDS


def _is_header_value_field(key: str) -> bool:
    return _normalize_profile_key(key) in {
        "header_value",
        "header_values",
        "value",
        "values",
    }


def _sanitize_secret_header_extra(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    return _REDACTED_PROFILE_SECRET


def _should_redact_header_scalar(value: str, key_path: tuple[str, ...]) -> bool:
    if not _has_header_collection_context(key_path):
        return False
    if _is_header_value_path(key_path):
        return True
    return _is_secret_key(value) or _is_secret_header_line(value)


def _is_secret_header_line(value: str) -> bool:
    header_name, separator, _header_value = value.partition(":")
    return bool(separator) and _is_secret_key(header_name.strip())


def _has_header_collection_context(key_path: tuple[str, ...]) -> bool:
    return any(_is_header_collection_key(key) for key in key_path)


def _is_header_value_path(key_path: tuple[str, ...]) -> bool:
    return (
        _has_header_collection_context(key_path)
        and bool(key_path)
        and _is_header_value_field(key_path[-1])
    )


def _is_header_collection_path(key_path: tuple[str, ...]) -> bool:
    if not key_path:
        return False
    return _is_header_collection_key(key_path[-1])


def _is_header_collection_key(key: str) -> bool:
    normalized = _normalize_profile_key(key)
    parts = [part for part in normalized.split("_") if part]
    return bool(parts) and bool(set(parts) & {"header", "headers"})


def _validate_profile_string(value: str, field: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise EmbedInputError(
            "embedding_profile_invalid",
            f"Embedding profile {field} must be valid UTF-8.",
        ) from error


def _is_secret_key(key: str) -> bool:
    normalized = _normalize_profile_key(key)
    secret_parts = {
        "authentication",
        "authorization",
        "bearer",
        "cookie",
        "cookies",
        "credential",
        "credentials",
        "jwt",
        "password",
        "passwords",
        "secret",
        "secrets",
    }
    secret_suffixes = (
        "auth",
        "auth_header",
        "auth_headers",
        "api_key",
        "api_keys",
        "apikey",
        "apikeys",
        "access_key",
        "access_key_id",
        "access_key_ids",
        "access_keys",
        "private_key",
        "private_keys",
        "subscription_key",
        "subscription_keys",
        "token",
    )
    secret_owner_parts = {
        "anthropic",
        "api",
        "app",
        "application",
        "auth",
        "azure",
        "bearer",
        "client",
        "cohere",
        "consumer",
        "developer",
        "gemini",
        "google",
        "jina",
        "mistral",
        "nomic",
        "oauth",
        "openai",
        "refresh",
        "service",
        "voyage",
    }
    token_owner_parts = {*secret_owner_parts, "access", "id", "session"}
    for candidate in _secret_key_candidates(normalized):
        normalized_parts_list = [part for part in candidate.split("_") if part]
        normalized_parts = set(normalized_parts_list)
        if candidate in {"key", "keys", "tokens"}:
            return True
        if bool(normalized_parts & secret_parts) or any(
            candidate == suffix or candidate.endswith(f"_{suffix}")
            for suffix in secret_suffixes
        ):
            return True
        prefix_parts = normalized_parts_list[:-1]
        compact_prefix = "".join(prefix_parts)
        if (
            normalized_parts_list
            and normalized_parts_list[-1] in {"key", "keys"}
            and (
                bool(set(prefix_parts) & secret_owner_parts)
                or compact_prefix in secret_owner_parts
            )
        ):
            return True
        if (
            normalized_parts_list
            and normalized_parts_list[-1] == "tokens"
            and (
                bool(set(prefix_parts) & token_owner_parts)
                or compact_prefix in token_owner_parts
            )
        ):
            return True
    return False


def _is_secret_key_path(keys: Iterable[str]) -> bool:
    normalized_keys = [_normalize_profile_key(key) for key in keys]
    key_path_without_value_qualifier = list(normalized_keys)
    while key_path_without_value_qualifier and key_path_without_value_qualifier[
        -1
    ] in {
        "value",
        "values",
    }:
        key_path_without_value_qualifier.pop()
    if (
        key_path_without_value_qualifier
        and _is_non_secret_profile_key(key_path_without_value_qualifier[-1])
    ):
        return False
    normalized = "_".join(
        part
        for key in normalized_keys
        for part in key.split("_")
        if part
    )
    return bool(normalized) and _is_secret_key(normalized)


def _is_non_secret_profile_key(normalized_key: str) -> bool:
    return any(
        candidate in _NON_SECRET_PROFILE_KEYS
        for candidate in _secret_key_candidates(normalized_key)
    )


def _secret_key_candidates(normalized: str) -> Iterable[str]:
    yield normalized
    parts = [part for part in normalized.split("_") if part]
    while parts and parts[-1] in {"value", "values"}:
        parts = parts[:-1]
        if parts:
            yield "_".join(parts)
    without_header_parts = [part for part in parts if part not in {"header", "headers"}]
    if without_header_parts and without_header_parts != parts:
        yield "_".join(without_header_parts)


def _normalize_profile_key(key: str) -> str:
    separated = re.sub(r"[^0-9A-Za-z]+", "_", key)
    with_word_boundaries = _CAMEL_CASE_BOUNDARY.sub("_", separated)
    return "_".join(part for part in with_word_boundaries.lower().split("_") if part)


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
        raise EmbedInputError(
            "embedding_profile_invalid",
            f"Embedding profile must be portable JSON: {error}",
        ) from error


def _jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    try:
        return "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            for row in rows
        )
    except ValueError as error:
        raise EmbedInputError(
            "chunks_invalid_jsonl",
            f"Embedding artifacts must be portable JSONL with finite values: {error}",
        ) from error
