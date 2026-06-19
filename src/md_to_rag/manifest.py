from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from md_to_rag import __version__

from .schemas import (
    CommandError,
    CommandName,
    CommandStatus,
    InitResponseData,
    InspectResponseData,
    ManifestCommandStatus,
    ProjectManifest,
)


MANIFEST_FILENAME = "corpus_manifest.json"
ARTIFACT_DIRECTORIES: dict[str, str] = {
    "source": "source",
    "documents": "documents",
    "chunks": "chunks",
    "embeddings": "embeddings",
    "indexes": "indexes",
    "reports": "reports",
}


class ManifestError(Exception):
    code = "manifest_error"

    def __init__(self, message: str, path: Path | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.path = path

    def to_command_error(self) -> CommandError:
        return CommandError(code=self.code, message=self.message)


class ProjectPathIsFileError(ManifestError):
    code = "project_path_is_file"


class ProjectCreateError(ManifestError):
    code = "project_create_failed"


class ArtifactPathIsFileError(ManifestError):
    code = "artifact_path_is_file"


class ArtifactCreateError(ManifestError):
    code = "artifact_create_failed"


class ManifestReadError(ManifestError):
    code = "manifest_invalid"


class ManifestWriteError(ManifestError):
    code = "manifest_write_failed"


@dataclass(frozen=True)
class InitProjectResult:
    data: InitResponseData
    message: str


@dataclass(frozen=True)
class InspectProjectResult:
    status: CommandStatus
    message: str
    data: InspectResponseData
    error: CommandError | None = None


def initialize_project(project: str | Path) -> InitProjectResult:
    project_root = Path(project).expanduser()
    if project_root.exists() and not project_root.is_dir():
        raise ProjectPathIsFileError(
            f"Project path is a file, not a directory: {project_root}",
            project_root,
        )

    try:
        project_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ProjectCreateError(
            f"Could not create project directory {project_root}: {error}",
            project_root,
        ) from error

    layout_changed = False
    for relative_path in ARTIFACT_DIRECTORIES.values():
        artifact_path = project_root / relative_path
        if artifact_path.exists() and not artifact_path.is_dir():
            raise ArtifactPathIsFileError(
                f"Artifact path is a file, not a directory: {artifact_path}",
                artifact_path,
            )
        if not artifact_path.exists():
            layout_changed = True
        try:
            artifact_path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ArtifactCreateError(
                f"Could not create artifact directory {artifact_path}: {error}",
                artifact_path,
            ) from error

    manifest_path = project_root / MANIFEST_FILENAME
    now = _utc_now()
    existing_manifest = _read_manifest(manifest_path) if manifest_path.exists() else None
    manifest = _normalized_manifest(existing_manifest, now)
    manifest_changed = True
    created = existing_manifest is None

    if existing_manifest is not None:
        existing_data = existing_manifest.model_dump(mode="json")
        candidate_data = manifest.model_dump(mode="json")
        manifest_changed = existing_data != candidate_data
        if manifest_changed:
            manifest = manifest.model_copy(update={"updated_at": now})

    if manifest_changed:
        _write_manifest(manifest_path, manifest)

    changed = manifest_changed or layout_changed
    if created:
        message = "Project initialized."
    elif changed:
        message = "Project updated."
    else:
        message = "Project already initialized."
    data = InitResponseData(
        project_root=str(project_root.resolve()),
        manifest_path=str(manifest_path.resolve()),
        created=created,
        changed=changed,
        directories=ARTIFACT_DIRECTORIES,
        manifest=manifest,
    )
    return InitProjectResult(data=data, message=message)


def inspect_project(artifact: str | Path | None = None) -> InspectProjectResult:
    target = Path.cwd() if artifact is None else Path(artifact).expanduser()
    target_resolved = target.resolve()
    artifact_exists = target.exists()

    if not artifact_exists:
        anchor = _nearest_existing_ancestor(target_resolved)
        manifest_path = _find_manifest(anchor)
        manifest = None
        manifest_error = None
        if manifest_path is not None:
            try:
                manifest = _read_manifest(manifest_path)
            except ManifestReadError as error:
                manifest_error = error

        issues = ["Artifact does not exist."]
        if manifest_error is not None:
            issues.append(manifest_error.message)

        data = InspectResponseData(
            artifact=str(target_resolved),
            artifact_exists=False,
            artifact_type="missing",
            project_root=str(manifest_path.parent.resolve()) if manifest_path else None,
            manifest_path=str(manifest_path.resolve()) if manifest_path else None,
            manifest_exists=manifest_path is not None,
            manifest=manifest,
            issues=issues,
        )
        if manifest_error is not None:
            return InspectProjectResult(
                status=CommandStatus.ERROR,
                message=manifest_error.message,
                data=data,
                error=manifest_error.to_command_error(),
            )

        return InspectProjectResult(
            status=CommandStatus.MISSING_ARTIFACT,
            message="Artifact does not exist.",
            data=data,
            error=CommandError(
                code="artifact_not_found",
                message=f"Artifact does not exist: {target}",
            ),
        )

    manifest_path, artifact_type = _manifest_for_existing_target(target)
    if manifest_path is None:
        data = InspectResponseData(
            artifact=str(target_resolved),
            artifact_exists=True,
            artifact_type=artifact_type,
            manifest_exists=False,
            issues=["No corpus_manifest.json found for artifact."],
        )
        return InspectProjectResult(
            status=CommandStatus.MISSING_ARTIFACT,
            message="No corpus_manifest.json found for artifact.",
            data=data,
            error=CommandError(
                code="manifest_not_found",
                message=f"No {MANIFEST_FILENAME} found for: {target}",
            ),
        )

    try:
        manifest = _read_manifest(manifest_path)
    except ManifestReadError as error:
        data = InspectResponseData(
            artifact=str(target_resolved),
            artifact_exists=True,
            artifact_type=artifact_type,
            project_root=str(manifest_path.parent.resolve()),
            manifest_path=str(manifest_path.resolve()),
            manifest_exists=True,
            issues=[error.message],
        )
        return InspectProjectResult(
            status=CommandStatus.ERROR,
            message=error.message,
            data=data,
            error=error.to_command_error(),
        )

    data = InspectResponseData(
        artifact=str(target_resolved),
        artifact_exists=True,
        artifact_type=artifact_type,
        project_root=str(manifest_path.parent.resolve()),
        manifest_path=str(manifest_path.resolve()),
        manifest_exists=True,
        manifest=manifest,
    )
    return InspectProjectResult(
        status=CommandStatus.OK,
        message="Artifact inspected.",
        data=data,
    )


def _normalized_manifest(
    existing_manifest: ProjectManifest | None,
    now: str,
) -> ProjectManifest:
    created_at = existing_manifest.created_at if existing_manifest else now
    updated_at = existing_manifest.updated_at if existing_manifest else now
    existing_statuses = {
        status.command: status
        for status in existing_manifest.command_status
    } if existing_manifest else {}

    command_status = [
        _normalized_command_status(command, existing_statuses.get(command), now)
        for command in CommandName
    ]
    return ProjectManifest(
        md_to_rag_version=__version__,
        created_at=created_at,
        updated_at=updated_at,
        artifact_directories=ARTIFACT_DIRECTORIES,
        command_status=command_status,
    )


def _normalized_command_status(
    command: CommandName,
    existing_status: ManifestCommandStatus | None,
    default_updated_at: str,
) -> ManifestCommandStatus:
    if command is CommandName.INIT:
        return ManifestCommandStatus(
            command=command,
            status=CommandStatus.OK,
            message="Project initialized.",
            artifact_path=MANIFEST_FILENAME,
            updated_at=existing_status.updated_at if existing_status else default_updated_at,
            data=existing_status.data if existing_status else {},
        )

    if command in {CommandName.INSPECT, CommandName.DIFF}:
        command_title = command.value.title()
        return ManifestCommandStatus(
            command=command,
            status=CommandStatus.OK,
            message=f"{command_title} available.",
            artifact_path=MANIFEST_FILENAME,
            updated_at=existing_status.updated_at if existing_status else default_updated_at,
            data=existing_status.data if existing_status else {},
        )

    if existing_status is not None:
        return existing_status

    return ManifestCommandStatus(
        command=command,
        status=CommandStatus.NOT_IMPLEMENTED,
        message=f"{command.value} is defined but not implemented yet.",
    )


def _manifest_for_existing_target(target: Path) -> tuple[Path | None, str]:
    if target.is_dir():
        direct_manifest = target / MANIFEST_FILENAME
        if direct_manifest.exists():
            return direct_manifest, "project"
        return _find_manifest(target), "artifact"

    if target.name == MANIFEST_FILENAME:
        return target, "manifest"

    return _find_manifest(target.parent), "artifact"


def _nearest_existing_ancestor(target: Path) -> Path:
    candidate = target if target.exists() else target.parent
    if candidate.exists():
        return candidate

    for parent in candidate.parents:
        if parent.exists():
            return parent

    return Path(candidate.anchor) if candidate.anchor else Path.cwd()


def _find_manifest(start: Path) -> Path | None:
    start = start.resolve()
    candidates = [start] if start.is_dir() else [start.parent]
    candidates.extend(candidates[0].parents)
    for directory in candidates:
        manifest_path = directory / MANIFEST_FILENAME
        if manifest_path.exists():
            return manifest_path
    return None


def _read_manifest(manifest_path: Path) -> ProjectManifest:
    try:
        raw_data: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
        _require_schema_markers(raw_data, manifest_path)
        return ProjectManifest.model_validate(raw_data)
    except (OSError, ValueError, json.JSONDecodeError, ValidationError) as error:
        raise ManifestReadError(
            f"Could not read a valid md_to_rag manifest at {manifest_path}: {error}",
            manifest_path,
        ) from error


def _require_schema_markers(raw_data: Any, manifest_path: Path) -> None:
    if not isinstance(raw_data, dict):
        raise ValueError(f"Manifest is not a JSON object: {manifest_path}")

    missing_markers = [
        field for field in ("schema_name", "schema_version") if field not in raw_data
    ]
    if missing_markers:
        markers = ", ".join(missing_markers)
        raise ValueError(f"Manifest is missing schema marker(s): {markers}")


def _write_manifest(manifest_path: Path, manifest: ProjectManifest) -> None:
    manifest_data = manifest.model_dump(mode="json")
    manifest_text = json.dumps(manifest_data, indent=2, sort_keys=True) + "\n"
    try:
        manifest_path.write_text(manifest_text, encoding="utf-8")
    except OSError as error:
        raise ManifestWriteError(
            f"Could not write md_to_rag manifest at {manifest_path}: {error}",
            manifest_path,
        ) from error


def _utc_now() -> str:
    timestamp = datetime.now(timezone.utc).replace(microsecond=0)
    return timestamp.isoformat().replace("+00:00", "Z")
