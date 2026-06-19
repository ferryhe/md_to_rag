from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from . import chunk as chunk_module
from . import embed as embed_module
from . import index as index_module
from . import ingest as ingest_module
from .manifest import (
    MANIFEST_FILENAME,
    ManifestReadError,
    _nearest_existing_ancestor,
    _read_manifest,
)
from .schemas import (
    ChunkResponseData,
    CommandError,
    CommandName,
    CommandStatus,
    DiffErrorData,
    DiffResponseData,
    DiffStageData,
    EmbedResponseData,
    IndexResponseData,
    IngestResponseData,
    ProjectManifest,
)


@dataclass(frozen=True)
class DiffProjectResult:
    status: CommandStatus
    message: str
    data: DiffResponseData | DiffErrorData
    artifact_path: str | None = None
    error: CommandError | None = None


@dataclass(frozen=True)
class _ProjectContext:
    project_root: Path
    manifest_path: Path
    manifest: ProjectManifest


class DiffInputError(Exception):
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


def diff_project(project: str | Path | None = None) -> DiffProjectResult:
    context_result = _resolve_context(project)
    if isinstance(context_result, DiffProjectResult):
        return context_result
    context = context_result

    stages: list[DiffStageData] = []
    upstream_rebuild_needed = False
    for command, checker in (
        (CommandName.INGEST, _diff_ingest),
        (CommandName.CHUNK, _diff_chunk),
        (CommandName.EMBED, _diff_embed),
        (CommandName.INDEX, _diff_index),
    ):
        stage = checker(context, upstream_rebuild_needed)
        stages.append(stage)
        upstream_rebuild_needed = upstream_rebuild_needed or stage.rebuild_needed

    data = DiffResponseData(
        project_root=str(context.project_root),
        manifest_path=str(context.manifest_path),
        rebuild_needed=any(stage.rebuild_needed for stage in stages),
        stages=stages,
        missing_stages=[
            stage.command for stage in stages if stage.missing
        ],
        stale_stages=[
            stage.command for stage in stages if stage.stale
        ],
        error_stages=[
            stage.command for stage in stages if stage.status == "error"
        ],
    )
    return DiffProjectResult(
        status=CommandStatus.OK,
        message="Artifact diff inspected.",
        data=data,
        artifact_path=str(context.manifest_path),
    )


def _resolve_context(project: str | Path | None) -> _ProjectContext | DiffProjectResult:
    path_error = _raw_project_path_error(project)
    if path_error is not None:
        return DiffProjectResult(
            status=path_error.status,
            message=path_error.message,
            data=DiffErrorData(),
            error=path_error.to_command_error(),
        )

    requested_path = _resolve_user_path(project) if project is not None else Path.cwd()
    if project is not None and not requested_path.exists():
        return DiffProjectResult(
            status=CommandStatus.MISSING_ARTIFACT,
            message=f"Project path does not exist: {requested_path}",
            data=DiffErrorData(project_path=str(requested_path)),
            error=CommandError(
                code="project_not_found",
                message=f"Project path does not exist: {requested_path}",
            ),
        )

    anchor = (
        requested_path
        if requested_path.exists()
        else _nearest_existing_ancestor(requested_path)
    )
    manifest_path = ingest_module._find_manifest_lexical(anchor)
    if manifest_path is None:
        return DiffProjectResult(
            status=CommandStatus.MISSING_ARTIFACT,
            message=f"No {MANIFEST_FILENAME} found for diff.",
            data=DiffErrorData(project_path=str(requested_path)),
            error=CommandError(
                code="manifest_not_found",
                message=f"No {MANIFEST_FILENAME} found for diff.",
            ),
        )

    try:
        manifest = _read_manifest(manifest_path)
    except ManifestReadError as error:
        project_root = manifest_path.parent.resolve()
        return DiffProjectResult(
            status=CommandStatus.ERROR,
            message=error.message,
            data=DiffErrorData(
                project_root=str(project_root),
                manifest_path=str(manifest_path.resolve()),
                project_path=str(requested_path),
            ),
            error=error.to_command_error(),
        )

    return _ProjectContext(
        project_root=manifest_path.parent.resolve(),
        manifest_path=manifest_path.resolve(),
        manifest=manifest,
    )


def _diff_ingest(
    context: _ProjectContext,
    upstream_rebuild_needed: bool,
) -> DiffStageData:
    try:
        source_relative = _ingest_source_relative(context)
    except DiffInputError as error:
        return _stage_from_step_result(
            CommandName.INGEST,
            error.status,
            error.message,
            {"source_path": ""},
            error.to_command_error(),
            upstream_rebuild_needed,
        )
    source_path = context.project_root / source_relative
    source_manifest_path = context.project_root / ingest_module.SOURCE_MANIFEST_PATH
    documents_path = context.project_root / ingest_module.DOCUMENTS_PATH
    artifact_paths = {
        "source_path": source_relative,
        "source_manifest_path": ingest_module.SOURCE_MANIFEST_PATH,
        "documents_path": ingest_module.DOCUMENTS_PATH,
    }
    context_result = ingest_module._resolve_context(source_path)
    if isinstance(context_result, ingest_module.IngestProjectResult):
        return _stage_from_step_result(
            CommandName.INGEST,
            context_result.status,
            context_result.message,
            artifact_paths,
            context_result.error,
            upstream_rebuild_needed,
        )
    if context_result.manifest_path != context.manifest_path:
        return _stage_from_step_result(
            CommandName.INGEST,
            CommandStatus.ERROR,
            "Ingest source resolves inside a nested initialized project.",
            artifact_paths,
            CommandError(
                code="source_nested_project",
                message="Ingest source resolves inside a nested initialized project.",
            ),
            upstream_rebuild_needed,
        )

    try:
        ingest_module._reject_generated_artifact_source(context_result)
        source_rows, document_rows = ingest_module._collect_rows(context_result)
        source_text = ingest_module._jsonl_text(source_rows)
        documents_text = ingest_module._jsonl_text(document_rows)
        expected_hashes = {
            "source_manifest_hash": _hash_text(source_text),
            "documents_hash": _hash_text(documents_text),
        }
        current_hashes = {
            "source_manifest_hash": _artifact_hash(
                context,
                source_manifest_path,
                ingest_module.SOURCE_MANIFEST_PATH,
                "Source manifest",
            ),
            "documents_hash": _artifact_hash(
                context,
                documents_path,
                ingest_module.DOCUMENTS_PATH,
                "Documents artifact",
            ),
        }
    except ingest_module.IngestInputError as error:
        return _stage_from_step_result(
            CommandName.INGEST,
            error.status,
            error.message,
            artifact_paths,
            error.to_command_error(),
            upstream_rebuild_needed,
        )
    except (OSError, DiffInputError) as error:
        return _error_stage(CommandName.INGEST, artifact_paths, error, upstream_rebuild_needed)

    data = IngestResponseData(
        project_root=str(context.project_root),
        manifest_path=str(context.manifest_path),
        source_path=context_result.source_path_relative,
        changed=False,
        source_count=len(source_rows),
        document_count=len(document_rows),
        source_manifest_path=ingest_module.SOURCE_MANIFEST_PATH,
        documents_path=ingest_module.DOCUMENTS_PATH,
        source_manifest_hash=expected_hashes["source_manifest_hash"],
        documents_hash=expected_hashes["documents_hash"],
    )
    manifest_matches = (
        ingest_module._manifest_status_matches(context.manifest, data)
        or _legacy_ingest_status_hashes_match(context.manifest, data)
    )
    return _stage_from_hashes(
        context,
        command=CommandName.INGEST,
        artifact_paths=artifact_paths,
        current_hashes=current_hashes,
        expected_hashes=expected_hashes,
        recorded_hash_keys=("source_manifest_hash", "documents_hash"),
        manifest_matches=manifest_matches,
        upstream_rebuild_needed=upstream_rebuild_needed,
    )


def _diff_chunk(
    context: _ProjectContext,
    upstream_rebuild_needed: bool,
) -> DiffStageData:
    try:
        documents_relative = _recorded_stage_input_path(
            context,
            CommandName.CHUNK,
            "documents_path",
            chunk_module.DOCUMENTS_PATH,
        )
    except DiffInputError as error:
        return _stage_from_step_result(
            CommandName.CHUNK,
            error.status,
            error.message,
            {"documents_path": "", "chunks_path": chunk_module.CHUNKS_PATH},
            error.to_command_error(),
            upstream_rebuild_needed,
        )
    documents_path = context.project_root / documents_relative
    chunks_path = context.project_root / chunk_module.CHUNKS_PATH
    artifact_paths = {
        "documents_path": documents_relative,
        "chunks_path": chunk_module.CHUNKS_PATH,
    }
    context_result = chunk_module._resolve_context(documents_path)
    if isinstance(context_result, chunk_module.ChunkProjectResult):
        return _stage_from_step_result(
            CommandName.CHUNK,
            context_result.status,
            context_result.message,
            artifact_paths,
            context_result.error,
            upstream_rebuild_needed,
        )

    try:
        documents_current_hash = _artifact_hash(
            context,
            context_result.documents_path,
            context_result.documents_path_relative,
            "Documents artifact",
        )
        document_rows, documents_hash = chunk_module._read_document_rows(
            context_result.documents_path
        )
        chunk_rows = chunk_module._chunk_rows(document_rows)
        chunks_text = chunk_module._jsonl_text(chunk_rows)
        expected_hashes = {
            "documents_hash": documents_hash,
            "chunks_hash": _hash_text(chunks_text),
        }
        current_hashes = {
            "documents_hash": documents_current_hash,
            "chunks_hash": _artifact_hash(
                context,
                chunks_path,
                chunk_module.CHUNKS_PATH,
                "Chunks artifact",
            ),
        }
    except chunk_module.ChunkInputError as error:
        return _stage_from_step_result(
            CommandName.CHUNK,
            error.status,
            error.message,
            artifact_paths,
            error.to_command_error(),
            upstream_rebuild_needed,
        )
    except (OSError, DiffInputError) as error:
        return _error_stage(CommandName.CHUNK, artifact_paths, error, upstream_rebuild_needed)

    data = ChunkResponseData(
        project_root=str(context.project_root),
        manifest_path=str(context.manifest_path),
        documents_path=context_result.documents_path_relative,
        changed=False,
        document_count=len(document_rows),
        chunk_count=len(chunk_rows),
        chunks_path=chunk_module.CHUNKS_PATH,
        documents_hash=documents_hash,
        chunks_hash=expected_hashes["chunks_hash"],
    )
    return _stage_from_hashes(
        context,
        command=CommandName.CHUNK,
        artifact_paths=artifact_paths,
        current_hashes=current_hashes,
        expected_hashes=expected_hashes,
        recorded_hash_keys=("documents_hash", "chunks_hash"),
        manifest_matches=chunk_module._manifest_status_matches(context.manifest, data),
        upstream_rebuild_needed=upstream_rebuild_needed,
    )


def _diff_embed(
    context: _ProjectContext,
    upstream_rebuild_needed: bool,
) -> DiffStageData:
    try:
        chunks_relative = _recorded_stage_input_path(
            context,
            CommandName.EMBED,
            "chunks_path",
            embed_module.CHUNKS_PATH,
        )
    except DiffInputError as error:
        return _stage_from_step_result(
            CommandName.EMBED,
            error.status,
            error.message,
            {"chunks_path": "", "embeddings_path": embed_module.EMBEDDINGS_PATH},
            error.to_command_error(),
            upstream_rebuild_needed,
        )
    chunks_path = context.project_root / chunks_relative
    embeddings_path = context.project_root / embed_module.EMBEDDINGS_PATH
    artifact_paths = {
        "chunks_path": chunks_relative,
        "embeddings_path": embed_module.EMBEDDINGS_PATH,
    }
    context_result = embed_module._resolve_context(chunks_path)
    if isinstance(context_result, embed_module.EmbedProjectResult):
        return _stage_from_step_result(
            CommandName.EMBED,
            context_result.status,
            context_result.message,
            artifact_paths,
            context_result.error,
            upstream_rebuild_needed,
        )

    try:
        chunks_current_hash = _artifact_hash(
            context,
            context_result.chunks_path,
            context_result.chunks_path_relative,
            "Chunks artifact",
        )
        embeddings_current_hash = _artifact_hash(
            context,
            embeddings_path,
            embed_module.EMBEDDINGS_PATH,
            "Embeddings artifact",
        )
        chunk_rows, chunks_hash = embed_module._read_chunk_rows(context_result.chunks_path)
        status_profile = _status_data_value(context.manifest, CommandName.EMBED, "profile")
        if isinstance(status_profile, dict):
            profile = status_profile
            provider = _deterministic_provider_from_profile(profile)
        else:
            provider = embed_module.DeterministicHashEmbeddingProvider()
            profile = embed_module._provider_profile(provider)
        embedding_count = len(chunk_rows)
        expected_embeddings_hash: str
        if embeddings_current_hash is None:
            if provider is None:
                expected_embeddings_hash = _recorded_hash(
                    context.manifest,
                    CommandName.EMBED,
                    "embeddings_hash",
                ) or ""
            else:
                profile = embed_module._provider_profile(provider)
                embedding_rows = embed_module._embedding_rows(
                    chunk_rows,
                    provider,
                    profile,
                    context_result,
                )
                expected_embeddings_hash = _hash_text(embed_module._jsonl_text(embedding_rows))
        else:
            embedding_artifact = index_module._read_embedding_rows(embeddings_path)
            profile = _profile_for_embedding_diff(context.manifest, embedding_artifact.profile)
            provider = _deterministic_provider_from_profile(profile)
            if provider is None:
                embedding_count = len(embedding_artifact.rows)
                expected_embeddings_hash = _recorded_hash(
                    context.manifest,
                    CommandName.EMBED,
                    "embeddings_hash",
                ) or embedding_artifact.artifact_hash
            else:
                profile = embed_module._provider_profile(provider)
                embedding_rows = embed_module._embedding_rows(
                    chunk_rows,
                    provider,
                    profile,
                    context_result,
                )
                embedding_count = len(embedding_rows)
                expected_embeddings_hash = _hash_text(embed_module._jsonl_text(embedding_rows))
        expected_hashes = {
            "chunks_hash": chunks_hash,
            "embeddings_hash": expected_embeddings_hash,
        }
        current_hashes = {
            "chunks_hash": chunks_current_hash,
            "embeddings_hash": embeddings_current_hash,
        }
    except embed_module.EmbedInputError as error:
        return _stage_from_step_result(
            CommandName.EMBED,
            error.status,
            error.message,
            artifact_paths,
            error.to_command_error(),
            upstream_rebuild_needed,
        )
    except index_module.IndexInputError as error:
        return _stage_from_step_result(
            CommandName.EMBED,
            error.status,
            error.message,
            artifact_paths,
            error.to_command_error(),
            upstream_rebuild_needed,
        )
    except (OSError, DiffInputError) as error:
        return _error_stage(CommandName.EMBED, artifact_paths, error, upstream_rebuild_needed)

    data = EmbedResponseData(
        project_root=str(context.project_root),
        manifest_path=str(context.manifest_path),
        chunks_path=context_result.chunks_path_relative,
        changed=False,
        chunk_count=len(chunk_rows),
        embedding_count=embedding_count,
        embeddings_path=embed_module.EMBEDDINGS_PATH,
        chunks_hash=chunks_hash,
        embeddings_hash=expected_hashes["embeddings_hash"],
        profile=profile,
    )
    return _stage_from_hashes(
        context,
        command=CommandName.EMBED,
        artifact_paths=artifact_paths,
        current_hashes=current_hashes,
        expected_hashes=expected_hashes,
        recorded_hash_keys=("chunks_hash", "embeddings_hash"),
        manifest_matches=embed_module._manifest_status_matches(context.manifest, data),
        upstream_rebuild_needed=upstream_rebuild_needed,
    )


def _diff_index(
    context: _ProjectContext,
    upstream_rebuild_needed: bool,
) -> DiffStageData:
    try:
        embeddings_relative = _recorded_stage_input_path(
            context,
            CommandName.INDEX,
            "embeddings_path",
            index_module.EMBEDDINGS_PATH,
        )
    except DiffInputError as error:
        return _stage_from_step_result(
            CommandName.INDEX,
            error.status,
            error.message,
            {
                "embeddings_path": "",
                "index_manifest_path": index_module.INDEX_MANIFEST_PATH,
                "index_path": index_module.INDEX_PATH,
            },
            error.to_command_error(),
            upstream_rebuild_needed,
        )
    embeddings_path = context.project_root / embeddings_relative
    index_manifest_path = context.project_root / index_module.INDEX_MANIFEST_PATH
    index_path = context.project_root / index_module.INDEX_PATH
    artifact_paths = {
        "embeddings_path": embeddings_relative,
        "index_manifest_path": index_module.INDEX_MANIFEST_PATH,
        "index_path": index_module.INDEX_PATH,
    }
    context_result = index_module._resolve_context(embeddings_path)
    if isinstance(context_result, index_module.IndexProjectResult):
        return _stage_from_step_result(
            CommandName.INDEX,
            context_result.status,
            context_result.message,
            artifact_paths,
            context_result.error,
            upstream_rebuild_needed,
        )

    try:
        embeddings_current_hash = _artifact_hash(
            context,
            context_result.embeddings_path,
            context_result.embeddings_path_relative,
            "Embeddings artifact",
        )
        current_hashes = {
            "embeddings_hash": embeddings_current_hash,
            "index_hash": _artifact_hash(
                context,
                index_path,
                index_module.INDEX_PATH,
                "Index artifact",
            ),
            "index_manifest_hash": _artifact_hash(
                context,
                index_manifest_path,
                index_module.INDEX_MANIFEST_PATH,
                "Index manifest",
            ),
        }
        if upstream_rebuild_needed:
            expected_hashes = _recorded_or_current_hashes(
                context.manifest,
                CommandName.INDEX,
                current_hashes,
            )
            return _stage_from_hashes(
                context,
                command=CommandName.INDEX,
                artifact_paths=artifact_paths,
                current_hashes=current_hashes,
                expected_hashes=expected_hashes,
                recorded_hash_keys=(
                    "embeddings_hash",
                    "index_hash",
                    "index_manifest_hash",
                ),
                manifest_matches=True,
                upstream_rebuild_needed=upstream_rebuild_needed,
            )
        embedding_artifact = index_module._read_embedding_rows(context_result.embeddings_path)
        embedding_artifact = index_module._ensure_chunk_provenance(
            context_result,
            embedding_artifact,
        )
        chunk_rows = index_module._chunk_rows_by_id(
            context_result,
            embedding_artifact.chunks_path,
        )
        index_rows = index_module._index_rows(
            embedding_artifact.rows,
            context_result,
            chunk_rows,
            chunks_recorded=embedding_artifact.chunks_path is not None,
        )
        index_text = index_module._jsonl_text(index_rows)
        index_hash = _hash_text(index_text)
        index_manifest = index_module._index_manifest_payload(
            context_result,
            embedding_artifact,
            index_hash,
        )
        index_manifest_text = index_module._json_text(index_manifest)
        index_manifest_hash = _hash_text(index_manifest_text)
        expected_hashes = {
            "embeddings_hash": embedding_artifact.artifact_hash,
            "index_hash": index_hash,
            "index_manifest_hash": index_manifest_hash,
        }
    except index_module.IndexInputError as error:
        return _stage_from_step_result(
            CommandName.INDEX,
            error.status,
            error.message,
            artifact_paths,
            error.to_command_error(),
            upstream_rebuild_needed,
        )
    except (OSError, DiffInputError) as error:
        return _error_stage(CommandName.INDEX, artifact_paths, error, upstream_rebuild_needed)

    data = IndexResponseData(
        project_root=str(context.project_root),
        manifest_path=str(context.manifest_path),
        embeddings_path=context_result.embeddings_path_relative,
        changed=False,
        embedding_count=len(embedding_artifact.rows),
        vector_count=len(index_rows),
        index_manifest_path=index_module.INDEX_MANIFEST_PATH,
        index_path=index_module.INDEX_PATH,
        embeddings_hash=embedding_artifact.artifact_hash,
        index_hash=index_hash,
        index_manifest_hash=index_manifest_hash,
        index_engine=index_module.INDEX_ENGINE,
        dimensions=embedding_artifact.dimensions,
        profile=embedding_artifact.profile,
        chunks_path=embedding_artifact.chunks_path,
    )
    return _stage_from_hashes(
        context,
        command=CommandName.INDEX,
        artifact_paths=artifact_paths,
        current_hashes=current_hashes,
        expected_hashes=expected_hashes,
        recorded_hash_keys=(
            "embeddings_hash",
            "index_hash",
            "index_manifest_hash",
        ),
        manifest_matches=index_module._manifest_status_matches(context.manifest, data),
        upstream_rebuild_needed=upstream_rebuild_needed,
    )


def _stage_from_hashes(
    context: _ProjectContext,
    *,
    command: CommandName,
    artifact_paths: dict[str, str],
    current_hashes: dict[str, str | None],
    expected_hashes: dict[str, str],
    recorded_hash_keys: tuple[str, ...],
    manifest_matches: bool,
    upstream_rebuild_needed: bool,
) -> DiffStageData:
    issues: list[str] = []
    missing = any(value is None for value in current_hashes.values())
    stale = False
    if missing:
        missing_keys = [
            key for key, value in current_hashes.items() if value is None
        ]
        issues.append(f"Missing artifact hash(es): {', '.join(missing_keys)}.")
    else:
        stale_keys = [
            key
            for key, expected in expected_hashes.items()
            if current_hashes.get(key) != expected
        ]
        if stale_keys:
            stale = True
            issues.append(f"Artifact content differs for: {', '.join(stale_keys)}.")
    if not manifest_matches:
        stale = True
        issues.append("Manifest status does not match current artifacts.")
    if upstream_rebuild_needed and not missing:
        stale = True
        issues.append("An upstream stage requires rebuild.")

    status = "fresh"
    if missing:
        status = "missing"
    elif stale:
        status = "stale"

    return DiffStageData(
        command=command,
        status=status,
        rebuild_needed=missing or stale,
        missing=missing,
        stale=stale and not missing,
        artifact_paths=artifact_paths,
        current_hashes=current_hashes,
        expected_hashes=expected_hashes,
        recorded_hashes=_recorded_hashes(context.manifest, command, recorded_hash_keys),
        issues=issues,
    )


def _stage_from_step_result(
    command: CommandName,
    status: CommandStatus,
    message: str,
    artifact_paths: dict[str, str],
    error: CommandError | None,
    upstream_rebuild_needed: bool,
) -> DiffStageData:
    stage_status = "missing" if status is CommandStatus.MISSING_ARTIFACT else "error"
    issues = [message]
    if upstream_rebuild_needed:
        issues.append("An upstream stage requires rebuild.")
    return DiffStageData(
        command=command,
        status=stage_status,
        rebuild_needed=True,
        missing=status is CommandStatus.MISSING_ARTIFACT,
        stale=False,
        artifact_paths=artifact_paths,
        issues=issues,
        error=error,
    )


def _error_stage(
    command: CommandName,
    artifact_paths: dict[str, str],
    error: Exception,
    upstream_rebuild_needed: bool,
) -> DiffStageData:
    command_error = (
        error.to_command_error()
        if isinstance(error, DiffInputError)
        else CommandError(code="diff_failed", message=str(error))
    )
    issues = [command_error.message]
    if upstream_rebuild_needed:
        issues.append("An upstream stage requires rebuild.")
    return DiffStageData(
        command=command,
        status="error",
        rebuild_needed=True,
        artifact_paths=artifact_paths,
        issues=issues,
        error=command_error,
    )


def _recorded_or_current_hashes(
    manifest: ProjectManifest,
    command: CommandName,
    current_hashes: dict[str, str | None],
) -> dict[str, str]:
    return {
        key: _recorded_hash(manifest, command, key) or current_hash or ""
        for key, current_hash in current_hashes.items()
    }


def _legacy_ingest_status_hashes_match(
    manifest: ProjectManifest,
    data: IngestResponseData,
) -> bool:
    for status in manifest.command_status:
        if status.command is not CommandName.INGEST:
            continue
        if status.status is not CommandStatus.OK or "source_path" in status.data:
            return False
        return (
            status.artifact_path == ingest_module.DOCUMENTS_PATH
            and status.data.get("source_count") == data.source_count
            and status.data.get("document_count") == data.document_count
            and status.data.get("source_manifest_path") == data.source_manifest_path
            and status.data.get("documents_path") == data.documents_path
            and status.data.get("source_manifest_hash") == data.source_manifest_hash
            and status.data.get("documents_hash") == data.documents_hash
        )
    return False


def _recorded_stage_input_path(
    context: _ProjectContext,
    command: CommandName,
    key: str,
    default_path: str,
) -> str:
    for status in context.manifest.command_status:
        if status.command is not command:
            continue
        if status.status is not CommandStatus.OK:
            break
        value = status.data.get(key)
        if value is None:
            break
        if not isinstance(value, str) or not value:
            raise DiffInputError(
                "stage_input_invalid",
                f"{command.value} manifest status records an invalid {key}.",
            )
        if _relative_project_path_error(value):
            raise DiffInputError(
                "stage_input_invalid",
                f"{command.value} manifest status records a non-portable {key}.",
            )
        return value
    return default_path


def _recorded_hashes(
    manifest: ProjectManifest,
    command: CommandName,
    keys: tuple[str, ...],
) -> dict[str, Any]:
    for status in manifest.command_status:
        if status.command is command:
            return {
                key: status.data.get(key)
                for key in keys
            }
    return {key: None for key in keys}


def _recorded_hash(
    manifest: ProjectManifest,
    command: CommandName,
    key: str,
) -> str | None:
    value = _status_data_value(manifest, command, key)
    return value if isinstance(value, str) else None


def _status_data_value(
    manifest: ProjectManifest,
    command: CommandName,
    key: str,
) -> Any:
    for status in manifest.command_status:
        if status.command is command:
            return status.data.get(key)
    return None


def _profile_for_embedding_diff(
    manifest: ProjectManifest,
    artifact_profile: dict[str, Any],
) -> dict[str, Any]:
    if artifact_profile:
        return artifact_profile
    status_profile = _status_data_value(manifest, CommandName.EMBED, "profile")
    return status_profile if isinstance(status_profile, dict) else artifact_profile


def _deterministic_provider_from_profile(
    profile: dict[str, Any],
) -> embed_module.DeterministicHashEmbeddingProvider | None:
    default_provider_name = embed_module.DeterministicHashEmbeddingProvider().provider
    if profile.get("provider") != default_provider_name:
        return None
    allowed_keys = {"provider", "model", "dimensions", "version", "options"}
    if set(profile) - allowed_keys:
        return None
    provider = profile.get("provider")
    model = profile.get("model")
    dimensions = profile.get("dimensions")
    version = profile.get("version")
    options = profile.get("options", {})
    if (
        not isinstance(provider, str)
        or not isinstance(model, str)
        or not isinstance(dimensions, int)
        or isinstance(dimensions, bool)
        or not isinstance(version, str)
        or not isinstance(options, dict)
    ):
        return None
    return embed_module._RecordedProfileDeterministicHashEmbeddingProvider(profile)


def _ingest_source_relative(context: _ProjectContext) -> str:
    manifest = context.manifest
    default_source = manifest.artifact_directories.get("source", "source")
    if not default_source or _relative_project_path_error(default_source):
        raise DiffInputError(
            "ingest_source_invalid",
            "Manifest source artifact directory must be a portable project-relative path.",
        )
    for status in manifest.command_status:
        if status.command is not CommandName.INGEST:
            continue
        if status.status is not CommandStatus.OK:
            break
        if "source_path" not in status.data:
            source_error = _ingest_source_project_error(
                context,
                context.project_root / default_source,
            )
            if source_error is not None:
                raise source_error
            legacy_source = _legacy_source_replay_relative(context, default_source, status)
            if legacy_source is None:
                raise DiffInputError(
                    "ingest_source_unavailable",
                    "Ingest manifest status does not record a replayable source_path.",
                )
            return legacy_source
        source_path = status.data.get("source_path")
        if not isinstance(source_path, str) or not source_path:
            raise DiffInputError(
                "ingest_source_unavailable",
                "Ingest manifest status does not record a replayable source_path.",
            )
        if _relative_project_path_error(source_path):
            raise DiffInputError(
                "ingest_source_invalid",
                "Ingest manifest status records a non-portable source_path.",
            )
        return source_path
    return default_source


def _legacy_source_replay_relative(
    context: _ProjectContext,
    default_source: str,
    ingest_status,
) -> str | None:
    source_rows = _legacy_source_manifest_rows(context)
    if source_rows is not None and _legacy_source_manifest_hash_matches(
        context,
        ingest_status,
    ):
        source_from_rows = _legacy_source_from_rows(
            context,
            source_rows,
            default_source,
        )
        if source_from_rows is not None:
            return source_from_rows

    if _legacy_default_source_matches_recorded_hashes(
        context,
        default_source,
        ingest_status,
    ):
        return default_source

    return None


def _legacy_source_manifest_hash_matches(
    context: _ProjectContext,
    ingest_status,
) -> bool:
    recorded_hash = ingest_status.data.get("source_manifest_hash")
    if not isinstance(recorded_hash, str):
        return False
    source_manifest_path = context.project_root / ingest_module.SOURCE_MANIFEST_PATH
    try:
        _reject_owned_artifact_path(
            context,
            source_manifest_path,
            ingest_module.SOURCE_MANIFEST_PATH,
            "Source manifest",
        )
        current_hash = ingest_module._hash_text(
            source_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, DiffInputError):
        return False
    return current_hash == recorded_hash


def _legacy_default_source_matches_recorded_hashes(
    context: _ProjectContext,
    default_source: str,
    ingest_status,
) -> bool:
    source_path = context.project_root / default_source
    if _ingest_source_project_error(context, source_path) is not None:
        return False
    recorded_source_hash = ingest_status.data.get("source_manifest_hash")
    recorded_documents_hash = ingest_status.data.get("documents_hash")
    if not isinstance(recorded_source_hash, str) or not isinstance(recorded_documents_hash, str):
        return False
    context_result = ingest_module._resolve_context(source_path)
    if isinstance(context_result, ingest_module.IngestProjectResult):
        return False
    try:
        source_rows, document_rows = ingest_module._collect_rows(context_result)
    except (OSError, ingest_module.IngestInputError):
        return False
    source_hash = ingest_module._hash_text(ingest_module._jsonl_text(source_rows))
    documents_hash = ingest_module._hash_text(ingest_module._jsonl_text(document_rows))
    return source_hash == recorded_source_hash and documents_hash == recorded_documents_hash


def _ingest_source_project_error(
    context: _ProjectContext,
    source_path: Path,
) -> DiffInputError | None:
    try:
        resolved_source = source_path.resolve()
    except (OSError, RuntimeError, ValueError):
        return DiffInputError(
            "ingest_source_invalid",
            "Ingest source path cannot be resolved safely.",
        )
    nested_manifest = ingest_module._nearest_nested_manifest(
        resolved_source,
        context.project_root,
        context.manifest_path,
    )
    if nested_manifest is not None:
        return DiffInputError(
            "source_nested_project",
            "Ingest source resolves inside a nested initialized project.",
        )
    return None


def _legacy_source_manifest_rows(
    context: _ProjectContext,
) -> list[dict[str, object]] | None:
    source_manifest_path = context.project_root / ingest_module.SOURCE_MANIFEST_PATH
    rows: list[dict[str, object]] = []
    try:
        _reject_owned_artifact_path(
            context,
            source_manifest_path,
            ingest_module.SOURCE_MANIFEST_PATH,
            "Source manifest",
        )
        text = source_manifest_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError, DiffInputError):
        return None
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(row, dict):
            return None
        rows.append(row)
    return rows


def _legacy_source_from_rows(
    context: _ProjectContext,
    rows: list[dict[str, object]],
    default_source: str,
) -> str | None:
    if not rows:
        return None

    default_prefix = default_source.rstrip("/")
    doc_to_md_manifest_paths: set[str] = set()
    markdown_source_paths: list[str] = []
    for row in rows:
        source_path = row.get("source_path")
        if not isinstance(source_path, str):
            return None
        if _relative_project_path_error(source_path):
            return None

        source_type = row.get("source_type")
        has_doc_to_md_metadata = (
            source_type == "doc_to_md_manifest"
            or "manifest_path" in row
            or "manifest_row_index" in row
            or "upstream_source_path" in row
            or "upstream_document_id" in row
        )
        if has_doc_to_md_metadata:
            manifest_path = row.get("manifest_path")
            if (
                not isinstance(manifest_path, str)
                or not manifest_path
                or _relative_project_path_error(manifest_path)
            ):
                return None
            doc_to_md_manifest_paths.add(manifest_path)
            continue

        if source_path != default_prefix and not source_path.startswith(f"{default_prefix}/"):
            return None
        if source_type != "markdown":
            return None
        markdown_source_paths.append(source_path)

    if doc_to_md_manifest_paths:
        if markdown_source_paths or len(doc_to_md_manifest_paths) != 1:
            return None
        return next(iter(doc_to_md_manifest_paths))

    if markdown_source_paths:
        return _legacy_markdown_source_candidate(
            context,
            default_source,
            markdown_source_paths,
        )

    return None


def _legacy_markdown_source_candidate(
    context: _ProjectContext,
    default_source: str,
    source_paths: list[str],
) -> str | None:
    unique_paths = sorted(set(source_paths))
    if len(unique_paths) != len(source_paths):
        return None
    if len(unique_paths) == 1:
        return unique_paths[0]

    parents = [PurePosixPath(path).parent for path in unique_paths]
    common_parts = list(parents[0].parts)
    for parent in parents[1:]:
        parent_parts = parent.parts
        while common_parts and tuple(common_parts) != parent_parts[:len(common_parts)]:
            common_parts.pop()
    if not common_parts:
        return None

    candidate = PurePosixPath(*common_parts).as_posix()
    if candidate in {"", "."}:
        return None
    if not _current_markdown_paths_match(context, candidate, unique_paths):
        return None
    return candidate


def _current_markdown_paths_match(
    context: _ProjectContext,
    source_relative: str,
    expected_paths: list[str],
) -> bool:
    source_path = context.project_root / source_relative
    if not source_path.exists() or not source_path.is_dir():
        return False
    try:
        current_paths = {
            path.relative_to(context.project_root).as_posix()
            for path in source_path.rglob("*")
            if path.is_file() and path.suffix.lower() in ingest_module.MARKDOWN_SUFFIXES
        }
    except (OSError, ValueError):
        return False
    return current_paths == set(expected_paths)


def _relative_project_path_error(path: str) -> bool:
    try:
        path.encode("utf-8")
    except UnicodeEncodeError:
        return True
    if "\\" in path:
        return True
    project_path = Path(path)
    windows_path = PureWindowsPath(path)
    if (
        project_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or windows_path.root
    ):
        return True
    return ".." in project_path.parts or ".." in windows_path.parts


def _artifact_hash(
    context: _ProjectContext,
    path: Path,
    relative_path: str,
    artifact_label: str,
) -> str | None:
    _reject_owned_artifact_path(context, path, relative_path, artifact_label)
    if not path.exists():
        return None
    try:
        return _hash_bytes(path.read_bytes())
    except OSError as error:
        raise DiffInputError(
            "artifact_read_failed",
            f"Could not read {artifact_label.lower()} {relative_path}: {error}",
        ) from error


def _reject_owned_artifact_path(
    context: _ProjectContext,
    path: Path,
    relative_path: str,
    artifact_label: str,
) -> None:
    try:
        resolved_path = path.resolve()
        project_root = context.project_root.resolve()
    except (OSError, RuntimeError, ValueError) as error:
        raise DiffInputError(
            "artifact_path_collision",
            f"{artifact_label} path cannot be resolved safely: {relative_path}",
        ) from error
    try:
        resolved_path.relative_to(project_root)
    except ValueError as error:
        raise DiffInputError(
            "artifact_path_outside_project",
            f"{artifact_label} path must stay inside the initialized project: {relative_path}",
        ) from error
    nested_manifest_path = ingest_module._nearest_nested_manifest(
        resolved_path,
        context.project_root,
        context.manifest_path,
    )
    if nested_manifest_path is not None:
        raise DiffInputError(
            "artifact_path_nested_project",
            f"{artifact_label} path resolves inside a nested initialized project.",
        )
    if resolved_path != path.absolute():
        raise DiffInputError(
            "artifact_path_collision",
            f"{artifact_label} path cannot be a symlink or linked path: {relative_path}",
        )
    if path.exists() and path.stat().st_nlink > 1:
        raise DiffInputError(
            "artifact_path_collision",
            f"{artifact_label} path cannot be a hard-linked path: {relative_path}",
        )


def _resolve_user_path(path: str | Path) -> Path:
    user_path = Path(path).expanduser()
    if user_path.is_absolute():
        return user_path
    return Path.cwd() / user_path


def _raw_project_path_error(path: str | Path | None) -> DiffInputError | None:
    if path is None:
        return None
    path_text = str(path)
    try:
        path_text.encode("utf-8")
    except UnicodeEncodeError:
        return DiffInputError(
            "project_path_not_portable",
            "Project path must be valid UTF-8.",
        )
    return None


def _hash_text(text: str) -> str:
    return _hash_bytes(text.encode("utf-8"))


def _hash_bytes(data: bytes) -> str:
    return f"sha256:{sha256(data).hexdigest()}"
