from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from . import ingest as ingest_module
from .chunk import CHUNKS_PATH, DOCUMENTS_PATH, chunk_project
from .embed import (
    EMBEDDINGS_PATH,
    DeterministicHashEmbeddingProvider,
    EmbedInputError,
    _RecordedProfileDeterministicHashEmbeddingProvider,
    embed_project,
)
from .index import INDEX_MANIFEST_PATH, INDEX_PATH, index_project
from .ingest import _find_manifest_lexical, ingest_project
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
    ManifestCommandStatus,
    ProjectManifest,
    RebuildErrorData,
    RebuildResponseData,
    RebuildStepData,
)


@dataclass(frozen=True)
class RebuildProjectResult:
    status: CommandStatus
    message: str
    data: RebuildResponseData | RebuildErrorData
    artifact_path: str | None = None
    error: CommandError | None = None


@dataclass(frozen=True)
class _ProjectContext:
    project_root: Path
    manifest_path: Path
    manifest: ProjectManifest


def rebuild_project(project: str | Path | None = None) -> RebuildProjectResult:
    context_result = _resolve_context(project)
    if isinstance(context_result, RebuildProjectResult):
        return context_result
    context = context_result

    steps: list[RebuildStepData] = []
    steps_changed = False
    source_path_result = _ingest_source_path(context)
    embed_provider_result = _embed_provider_from_manifest(context.manifest)
    if isinstance(source_path_result, RebuildInputError):
        failed_command = CommandName.INGEST
        failed_step = RebuildStepData(
            command=failed_command,
            status=source_path_result.status.value,
            message=source_path_result.message,
            error=source_path_result.to_command_error(),
        )
        steps.append(failed_step)
        for skipped_command in (CommandName.CHUNK, CommandName.EMBED, CommandName.INDEX):
            steps.append(_skipped_step(skipped_command, failed_command))
        return RebuildProjectResult(
            status=source_path_result.status,
            message=f"Rebuild stopped at ingest: {source_path_result.message}",
            data=RebuildErrorData(
                project_root=str(context.project_root),
                manifest_path=str(context.manifest_path),
                changed=False,
                completed=False,
                steps=steps,
                stopped_at=failed_command,
            ),
            error=source_path_result.to_command_error(),
        )

    source_project_error = _ingest_source_project_error(context, source_path_result)
    if source_project_error is not None:
        failed_command = CommandName.INGEST
        failed_step = RebuildStepData(
            command=failed_command,
            status=source_project_error.status.value,
            message=source_project_error.message,
            error=source_project_error.to_command_error(),
        )
        steps.append(failed_step)
        for skipped_command in (CommandName.CHUNK, CommandName.EMBED, CommandName.INDEX):
            steps.append(_skipped_step(skipped_command, failed_command))
        return RebuildProjectResult(
            status=source_project_error.status,
            message=f"Rebuild stopped at ingest: {source_project_error.message}",
            data=RebuildErrorData(
                project_root=str(context.project_root),
                manifest_path=str(context.manifest_path),
                changed=False,
                completed=False,
                steps=steps,
                stopped_at=failed_command,
            ),
            error=source_project_error.to_command_error(),
        )

    stage_specs = [
        (
            CommandName.INGEST,
            lambda: ingest_project(source_path_result),
        ),
        (
            CommandName.CHUNK,
            lambda: chunk_project(context.project_root / DOCUMENTS_PATH),
        ),
        (
            CommandName.EMBED,
            lambda: _embed_step(context, embed_provider_result),
        ),
        (
            CommandName.INDEX,
            lambda: index_project(context.project_root / EMBEDDINGS_PATH),
        ),
    ]

    for index, (command, run_step) in enumerate(stage_specs):
        result = run_step()
        step = _step_from_result(command, result)
        steps.append(step)
        steps_changed = steps_changed or step.changed is True
        if result.status is CommandStatus.OK:
            continue

        failed_command = command
        for skipped_command, _ in stage_specs[index + 1:]:
            steps.append(_skipped_step(skipped_command, failed_command))
        data = RebuildErrorData(
            project_root=str(context.project_root),
            manifest_path=str(context.manifest_path),
            changed=steps_changed,
            completed=False,
            steps=steps,
            stopped_at=failed_command,
        )
        return RebuildProjectResult(
            status=result.status,
            message=f"Rebuild stopped at {failed_command.value}: {result.message}",
            data=data,
            artifact_path=result.artifact_path,
            error=result.error,
        )

    try:
        current_manifest = _read_manifest(context.manifest_path)
    except ManifestReadError as error:
        data = RebuildErrorData(
            project_root=str(context.project_root),
            manifest_path=str(context.manifest_path),
            changed=steps_changed,
            completed=False,
            steps=steps,
        )
        return RebuildProjectResult(
            status=CommandStatus.ERROR,
            message=error.message,
            data=data,
            artifact_path=str(context.manifest_path),
            error=error.to_command_error(),
        )

    manifest_status_changed = steps_changed or not _manifest_status_matches(current_manifest)
    changed = steps_changed or manifest_status_changed
    data = RebuildResponseData(
        project_root=str(context.project_root),
        manifest_path=str(context.manifest_path),
        changed=changed,
        completed=True,
        steps=steps,
    )

    if manifest_status_changed:
        try:
            _update_manifest_status(context.manifest_path, current_manifest)
        except ManifestWriteError as error:
            return RebuildProjectResult(
                status=CommandStatus.ERROR,
                message=error.message,
                data=data,
                artifact_path=str(context.manifest_path),
                error=error.to_command_error(),
            )

    return RebuildProjectResult(
        status=CommandStatus.OK,
        message="Rebuild completed.",
        data=data,
        artifact_path=str((context.project_root / INDEX_MANIFEST_PATH).resolve()),
    )


def _resolve_context(
    project: str | Path | None,
) -> _ProjectContext | RebuildProjectResult:
    path_error = _raw_project_path_error(project)
    if path_error is not None:
        return RebuildProjectResult(
            status=path_error.status,
            message=path_error.message,
            data=RebuildErrorData(),
            error=path_error.to_command_error(),
        )

    requested_path = _resolve_user_path(project) if project is not None else Path.cwd()
    if project is not None and not requested_path.exists():
        return RebuildProjectResult(
            status=CommandStatus.MISSING_ARTIFACT,
            message=f"Project path does not exist: {requested_path}",
            data=RebuildErrorData(project_path=str(requested_path)),
            error=CommandError(
                code="project_not_found",
                message=f"Project path does not exist: {requested_path}",
            ),
        )

    if project is None:
        manifest_path = _find_manifest_lexical(
            requested_path
            if requested_path.exists()
            else _nearest_existing_ancestor(requested_path)
        )
    elif requested_path.is_file() and requested_path.name == MANIFEST_FILENAME:
        manifest_path = requested_path
    elif requested_path.is_dir() and (requested_path / MANIFEST_FILENAME).exists():
        manifest_path = requested_path / MANIFEST_FILENAME
    else:
        return RebuildProjectResult(
            status=CommandStatus.ERROR,
            message=f"Project path is not an initialized project: {requested_path}",
            data=RebuildErrorData(project_path=str(requested_path)),
            error=CommandError(
                code="project_path_not_project",
                message=f"Project path is not an initialized project: {requested_path}",
            ),
        )
    if manifest_path is None:
        return RebuildProjectResult(
            status=CommandStatus.MISSING_ARTIFACT,
            message=f"No {MANIFEST_FILENAME} found for rebuild.",
            data=RebuildErrorData(project_path=str(requested_path)),
            error=CommandError(
                code="manifest_not_found",
                message=f"No {MANIFEST_FILENAME} found for rebuild.",
            ),
        )

    try:
        manifest = _read_manifest(manifest_path)
    except ManifestReadError as error:
        project_root = manifest_path.parent.resolve()
        return RebuildProjectResult(
            status=CommandStatus.ERROR,
            message=error.message,
            data=RebuildErrorData(
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


def _ingest_source_project_error(
    context: _ProjectContext,
    source_path: Path,
) -> RebuildInputError | None:
    try:
        resolved_source = source_path.resolve()
    except (OSError, RuntimeError, ValueError):
        return RebuildInputError(
            "ingest_source_invalid",
            "Ingest source path cannot be resolved safely.",
        )
    nested_manifest = ingest_module._nearest_nested_manifest(
        resolved_source,
        context.project_root,
        context.manifest_path,
    )
    if nested_manifest is not None:
        return RebuildInputError(
            "source_nested_project",
            "Ingest source resolves inside a nested initialized project.",
        )
    return None


def _step_from_result(command: CommandName, result: object) -> RebuildStepData:
    result_status = getattr(result, "status")
    result_data = getattr(result, "data")
    changed = getattr(result_data, "changed", None)
    return RebuildStepData(
        command=command,
        status=result_status.value,
        message=getattr(result, "message"),
        changed=changed if isinstance(changed, bool) else None,
        artifact_path=getattr(result, "artifact_path"),
        error=getattr(result, "error"),
    )


def _skipped_step(command: CommandName, failed_command: CommandName) -> RebuildStepData:
    return RebuildStepData(
        command=command,
        status="skipped",
        message=f"Skipped because {failed_command.value} did not complete.",
        skipped=True,
    )


def _embed_step(
    context: _ProjectContext,
    provider: DeterministicHashEmbeddingProvider | RebuildInputError | None,
) -> object:
    if isinstance(provider, RebuildInputError):
        return RebuildProjectResult(
            status=provider.status,
            message=provider.message,
            data=RebuildErrorData(
                project_root=str(context.project_root),
                manifest_path=str(context.manifest_path),
                changed=False,
                completed=False,
            ),
            error=provider.to_command_error(),
        )
    return embed_project(context.project_root / CHUNKS_PATH, provider=provider)


def _manifest_status_matches(manifest: ProjectManifest) -> bool:
    for existing_status in manifest.command_status:
        if existing_status.command is not CommandName.REBUILD:
            continue
        return (
            existing_status.status is CommandStatus.OK
            and existing_status.artifact_path == INDEX_MANIFEST_PATH
            and existing_status.data.get("completed_steps") == [
                command.value
                for command in (
                    CommandName.INGEST,
                    CommandName.CHUNK,
                    CommandName.EMBED,
                    CommandName.INDEX,
                )
            ]
            and existing_status.data.get("index_manifest_path") == INDEX_MANIFEST_PATH
            and existing_status.data.get("index_path") == INDEX_PATH
        )
    return False


def _update_manifest_status(
    manifest_path: Path,
    manifest: ProjectManifest,
) -> None:
    status = ManifestCommandStatus(
        command=CommandName.REBUILD,
        status=CommandStatus.OK,
        message="Rebuild completed.",
        artifact_path=INDEX_MANIFEST_PATH,
        updated_at=_utc_now(),
        data={
            "completed_steps": [
                CommandName.INGEST.value,
                CommandName.CHUNK.value,
                CommandName.EMBED.value,
                CommandName.INDEX.value,
            ],
            "index_manifest_path": INDEX_MANIFEST_PATH,
            "index_path": INDEX_PATH,
        },
    )
    command_status = []
    replaced = False
    for existing_status in manifest.command_status:
        if existing_status.command is CommandName.REBUILD:
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


def _resolve_user_path(path: str | Path) -> Path:
    user_path = Path(path).expanduser()
    if user_path.is_absolute():
        return user_path
    return Path.cwd() / user_path


class RebuildInputError(Exception):
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


def _raw_project_path_error(path: str | Path | None) -> RebuildInputError | None:
    if path is None:
        return None
    path_text = str(path)
    try:
        path_text.encode("utf-8")
    except UnicodeEncodeError:
        return RebuildInputError(
            "project_path_not_portable",
            "Project path must be valid UTF-8.",
        )
    return None


def _ingest_source_path(context: _ProjectContext) -> Path | RebuildInputError:
    source_relative = context.manifest.artifact_directories.get("source", "source")
    if not source_relative or _relative_project_path_error(source_relative):
        return RebuildInputError(
            "ingest_source_invalid",
            "Manifest source artifact directory must be a portable project-relative path.",
        )
    for status in context.manifest.command_status:
        if status.command is not CommandName.INGEST:
            continue
        if status.status is not CommandStatus.OK:
            break
        if "source_path" not in status.data:
            legacy_source = _legacy_source_replay_relative(
                context,
                source_relative,
                status,
            )
            if legacy_source is None:
                return RebuildInputError(
                    "ingest_source_unavailable",
                    "Ingest manifest status does not record a replayable source_path.",
                )
            source_relative = legacy_source
            break
        source_path = status.data.get("source_path")
        if not isinstance(source_path, str) or not source_path:
            return RebuildInputError(
                "ingest_source_unavailable",
                "Ingest manifest status does not record a replayable source_path.",
            )
        if _relative_project_path_error(source_path):
            return RebuildInputError(
                "ingest_source_invalid",
                "Ingest manifest status records a non-portable source_path.",
            )
        source_relative = source_path
        break
    return context.project_root / source_relative


def _legacy_source_replay_relative(
    context: _ProjectContext,
    default_source: str,
    ingest_status: ManifestCommandStatus,
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
    ingest_status: ManifestCommandStatus,
) -> bool:
    recorded_hash = ingest_status.data.get("source_manifest_hash")
    if not isinstance(recorded_hash, str):
        return False
    source_manifest_path = context.project_root / ingest_module.SOURCE_MANIFEST_PATH
    if not _owned_artifact_path_is_safe(
        context,
        source_manifest_path,
        ingest_module.SOURCE_MANIFEST_PATH,
    ):
        return False
    try:
        current_hash = ingest_module._hash_text(
            source_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError):
        return False
    return current_hash == recorded_hash


def _legacy_default_source_matches_recorded_hashes(
    context: _ProjectContext,
    default_source: str,
    ingest_status: ManifestCommandStatus,
) -> bool:
    source_path = context.project_root / default_source
    recorded_source_hash = ingest_status.data.get("source_manifest_hash")
    recorded_documents_hash = ingest_status.data.get("documents_hash")
    if not isinstance(recorded_source_hash, str) or not isinstance(recorded_documents_hash, str):
        return False
    if _ingest_source_project_error(context, source_path) is not None:
        return False
    context_result = ingest_module._resolve_context(source_path)
    if isinstance(context_result, ingest_module.IngestProjectResult):
        return False
    if context_result.manifest_path != context.manifest_path:
        return False
    try:
        source_rows, document_rows = ingest_module._collect_rows(context_result)
    except (OSError, ingest_module.IngestInputError):
        return False
    source_hash = ingest_module._hash_text(ingest_module._jsonl_text(source_rows))
    documents_hash = ingest_module._hash_text(ingest_module._jsonl_text(document_rows))
    return source_hash == recorded_source_hash and documents_hash == recorded_documents_hash


def _legacy_source_manifest_rows(
    context: _ProjectContext,
) -> list[dict[str, object]] | None:
    source_manifest_path = context.project_root / ingest_module.SOURCE_MANIFEST_PATH
    rows: list[dict[str, object]] = []
    if not _owned_artifact_path_is_safe(
        context,
        source_manifest_path,
        ingest_module.SOURCE_MANIFEST_PATH,
    ):
        return None
    try:
        text = source_manifest_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
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


def _owned_artifact_path_is_safe(
    context: _ProjectContext,
    path: Path,
    relative_path: str,
) -> bool:
    try:
        resolved_path = path.resolve()
        project_root = context.project_root.resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    try:
        resolved_path.relative_to(project_root)
    except ValueError:
        return False
    nested_manifest = ingest_module._nearest_nested_manifest(
        resolved_path,
        context.project_root,
        context.manifest_path,
    )
    if nested_manifest is not None:
        return False
    if resolved_path != path.absolute():
        return False
    try:
        return not path.exists() or path.stat().st_nlink <= 1
    except OSError:
        return False


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


def _embed_provider_from_manifest(
    manifest: ProjectManifest,
) -> DeterministicHashEmbeddingProvider | RebuildInputError | None:
    for status in manifest.command_status:
        if status.command is not CommandName.EMBED:
            continue
        if status.status is not CommandStatus.OK:
            return None
        profile = status.data.get("profile")
        if profile is None:
            return None
        if not isinstance(profile, dict):
            return RebuildInputError(
                "embedding_profile_invalid",
                "Embed manifest status has invalid profile provenance.",
            )
        try:
            provider = _deterministic_provider_from_profile(profile)
        except EmbedInputError:
            return RebuildInputError(
                "embedding_profile_invalid",
                "Embed manifest status has invalid profile provenance.",
            )
        if provider is None:
            return RebuildInputError(
                "embedding_profile_unsupported",
                "Rebuild cannot replay the recorded embedding provider profile.",
            )
        return provider
    return None


def _deterministic_provider_from_profile(
    profile: dict[str, object],
) -> DeterministicHashEmbeddingProvider | None:
    default_provider_name = DeterministicHashEmbeddingProvider().provider
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
    return _RecordedProfileDeterministicHashEmbeddingProvider(profile)
