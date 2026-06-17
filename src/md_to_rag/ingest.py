from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable

from .manifest import (
    MANIFEST_FILENAME,
    ManifestReadError,
    ManifestWriteError,
    _find_manifest,
    _nearest_existing_ancestor,
    _read_manifest,
    _utc_now,
    _write_manifest,
)
from .schemas import (
    CommandError,
    CommandName,
    CommandStatus,
    IngestErrorData,
    IngestResponseData,
    ManifestCommandStatus,
    ProjectManifest,
)


SOURCE_MANIFEST_PATH = "source/source_manifest.jsonl"
DOCUMENTS_PATH = "documents/documents.jsonl"
MARKDOWN_SUFFIXES = {".md", ".markdown"}


@dataclass(frozen=True)
class IngestProjectResult:
    status: CommandStatus
    message: str
    data: IngestResponseData | IngestErrorData
    artifact_path: str | None = None
    error: CommandError | None = None


@dataclass(frozen=True)
class _ProjectContext:
    project_root: Path
    manifest_path: Path
    manifest: ProjectManifest
    source_path: Path
    source_path_relative: str


class IngestInputError(Exception):
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


def ingest_project(source: str | Path | None = None) -> IngestProjectResult:
    context_result = _resolve_context(source)
    if isinstance(context_result, IngestProjectResult):
        return context_result
    context = context_result

    try:
        _reject_generated_artifact_source(context)
        source_rows, document_rows = _collect_rows(context.source_path, context.project_root)
        source_text = _jsonl_text(source_rows)
        documents_text = _jsonl_text(document_rows)
        source_manifest_path = context.project_root / SOURCE_MANIFEST_PATH
        documents_path = context.project_root / DOCUMENTS_PATH
        source_changed = _write_if_changed(source_manifest_path, source_text)
        documents_changed = _write_if_changed(documents_path, documents_text)
    except IngestInputError as error:
        return _input_error_result(error, context)
    except OSError as error:
        ingest_error = IngestInputError(
            "ingest_io_failed",
            f"Could not generate ingest artifacts: {error}",
        )
        return _input_error_result(ingest_error, context)

    source_manifest_hash = _hash_text(source_text)
    documents_hash = _hash_text(documents_text)
    data = IngestResponseData(
        project_root=str(context.project_root),
        manifest_path=str(context.manifest_path),
        source_path=context.source_path_relative,
        changed=False,
        source_count=len(source_rows),
        document_count=len(document_rows),
        source_manifest_path=SOURCE_MANIFEST_PATH,
        documents_path=DOCUMENTS_PATH,
        source_manifest_hash=source_manifest_hash,
        documents_hash=documents_hash,
    )
    artifacts_changed = source_changed or documents_changed
    manifest_status_changed = not _manifest_status_matches(context.manifest, data)
    changed = artifacts_changed or manifest_status_changed
    data = data.model_copy(update={"changed": changed})

    if changed:
        try:
            _update_manifest_status(context.manifest_path, context.manifest, data)
        except ManifestWriteError as error:
            return IngestProjectResult(
                status=CommandStatus.ERROR,
                message=error.message,
                data=data,
                artifact_path=str(documents_path.resolve()),
                error=error.to_command_error(),
            )
        message = "Ingest artifacts generated."
    else:
        message = "Ingest artifacts unchanged."

    return IngestProjectResult(
        status=CommandStatus.OK,
        message=message,
        data=data,
        artifact_path=str(documents_path.resolve()),
    )


def _resolve_context(source: str | Path | None) -> _ProjectContext | IngestProjectResult:
    if source is None:
        manifest_path = _find_manifest(Path.cwd())
        source_path: Path | None = None
    else:
        requested_source = _resolve_user_path(source)
        anchor = requested_source if requested_source.exists() else _nearest_existing_ancestor(requested_source)
        manifest_path = _find_manifest(anchor)
        source_path = requested_source

    if manifest_path is None:
        return IngestProjectResult(
            status=CommandStatus.MISSING_ARTIFACT,
            message=f"No {MANIFEST_FILENAME} found for ingest source.",
            data=IngestErrorData(
                source_path=str(_resolve_user_path(source)) if source is not None else None,
            ),
            error=CommandError(
                code="manifest_not_found",
                message=f"No {MANIFEST_FILENAME} found for ingest source.",
            ),
        )

    try:
        manifest = _read_manifest(manifest_path)
    except ManifestReadError as error:
        project_root = manifest_path.parent.resolve()
        return IngestProjectResult(
            status=CommandStatus.ERROR,
            message=error.message,
            data=IngestErrorData(
                project_root=str(project_root),
                manifest_path=str(manifest_path.resolve()),
                source_path=str(source) if source is not None else None,
            ),
            error=error.to_command_error(),
        )

    project_root = manifest_path.parent.resolve()
    if source_path is None:
        source_path = project_root / manifest.artifact_directories.get("source", "source")

    source_resolved = source_path.resolve()
    source_relative_result = _relative_to_project(source_resolved, project_root)
    if isinstance(source_relative_result, IngestInputError):
        return _input_error_result(
            source_relative_result,
            _ProjectContext(
                project_root=project_root,
                manifest_path=manifest_path.resolve(),
                manifest=manifest,
                source_path=source_resolved,
                source_path_relative=str(source_resolved),
            ),
        )

    if not source_resolved.exists():
        return IngestProjectResult(
            status=CommandStatus.MISSING_ARTIFACT,
            message=f"Ingest source does not exist: {source_relative_result}",
            data=IngestErrorData(
                project_root=str(project_root),
                manifest_path=str(manifest_path.resolve()),
                source_path=source_relative_result,
            ),
            error=CommandError(
                code="source_not_found",
                message=f"Ingest source does not exist: {source_relative_result}",
            ),
        )

    return _ProjectContext(
        project_root=project_root,
        manifest_path=manifest_path.resolve(),
        manifest=manifest,
        source_path=source_resolved,
        source_path_relative=source_relative_result,
    )


def _collect_rows(source_path: Path, project_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if source_path.is_dir():
        markdown_paths = sorted(
            (
                path
                for path in source_path.rglob("*")
                if path.is_file() and path.suffix.lower() in MARKDOWN_SUFFIXES
            ),
            key=lambda path: _relative_to_project_or_raise(path.resolve(), project_root),
        )
        records = [
            _record_from_markdown(path, project_root)
            for path in markdown_paths
        ]
        return _rows_from_records(records)

    if source_path.is_file() and source_path.suffix.lower() in MARKDOWN_SUFFIXES:
        return _rows_from_records([_record_from_markdown(source_path, project_root)])

    if source_path.is_file() and source_path.suffix.lower() in {".json", ".jsonl"}:
        records = [
            _record_from_manifest_row(source_path, project_root, row, index)
            for index, row in enumerate(_read_doc_to_md_rows(source_path))
        ]
        records.sort(key=lambda record: record["source_path"])
        return _rows_from_records(records)

    raise IngestInputError(
        "unsupported_source",
        f"Ingest source must be a Markdown file, Markdown directory, JSON manifest, or JSONL manifest: {source_path}",
    )


def _reject_generated_artifact_source(context: _ProjectContext) -> None:
    generated_artifacts = {
        (context.project_root / SOURCE_MANIFEST_PATH).resolve(): SOURCE_MANIFEST_PATH,
        (context.project_root / DOCUMENTS_PATH).resolve(): DOCUMENTS_PATH,
    }
    artifact_path = generated_artifacts.get(context.source_path.resolve())
    if artifact_path is not None:
        raise IngestInputError(
            "source_artifact_collision",
            f"Ingest source cannot be a generated md_to_rag artifact: {artifact_path}",
        )


def _record_from_markdown(path: Path, project_root: Path) -> dict[str, Any]:
    source_path = _relative_to_project(path.resolve(), project_root)
    if isinstance(source_path, IngestInputError):
        raise source_path
    content = _read_markdown(path)
    content_hash = _hash_bytes(content.encode("utf-8"))
    title = _title_from_markdown(content, path)
    provenance = {
        "kind": "markdown",
        "source_path": source_path,
    }
    return {
        "source_path": source_path,
        "source_type": "markdown",
        "content": content,
        "content_hash": content_hash,
        "metadata": {
            "source_extension": path.suffix.lower(),
            "title": title,
        },
        "provenance": provenance,
    }


def _record_from_manifest_row(
    manifest_path: Path,
    project_root: Path,
    row: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    markdown_value = _first_value(
        row,
        ("markdown_path", "md_path", "output_path", "content_path", "document_path", "path"),
    )
    if markdown_value is None:
        raise IngestInputError(
            "manifest_row_missing_markdown_path",
            f"doc_to_md manifest row {index} is missing a Markdown path.",
        )

    markdown_path = _project_path_from_manifest_value(markdown_value, project_root)
    if not markdown_path.exists():
        markdown_relative = _relative_to_project(markdown_path, project_root)
        if isinstance(markdown_relative, IngestInputError):
            raise markdown_relative
        raise IngestInputError(
            "source_not_found",
            f"Manifest row {index} references missing Markdown source: {markdown_relative}",
            status=CommandStatus.MISSING_ARTIFACT,
        )

    source_path = _relative_to_project(markdown_path.resolve(), project_root)
    manifest_relative = _relative_to_project(manifest_path.resolve(), project_root)
    if isinstance(source_path, IngestInputError):
        raise source_path
    if isinstance(manifest_relative, IngestInputError):
        raise manifest_relative

    upstream_source = _portable_upstream_path(
        _first_value(row, ("source_path", "input_path", "original_path"))
    )
    upstream_document_id = _upstream_document_id(row)
    content = _read_markdown(markdown_path)
    content_hash = _hash_bytes(content.encode("utf-8"))
    metadata = _metadata_from_manifest_row(row, content, markdown_path)
    provenance: dict[str, Any] = {
        "kind": "doc_to_md_manifest",
        "manifest_path": manifest_relative,
        "manifest_row_index": index,
    }
    if upstream_source is not None:
        provenance["source_path"] = upstream_source
    if upstream_document_id is not None:
        provenance["upstream_document_id"] = upstream_document_id

    return {
        "source_path": source_path,
        "source_type": "doc_to_md_manifest",
        "identity_key": _doc_to_md_identity_key(
            manifest_relative,
            source_path,
            upstream_source,
            upstream_document_id,
        ),
        "content": content,
        "content_hash": content_hash,
        "metadata": metadata,
        "provenance": provenance,
        "manifest_path": manifest_relative,
        "manifest_row_index": index,
        "upstream_source_path": upstream_source,
        "upstream_document_id": upstream_document_id,
    }


def _rows_from_records(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_rows: list[dict[str, Any]] = []
    document_rows: list[dict[str, Any]] = []
    seen_source_keys: set[str] = set()
    for record in records:
        source_key = _source_key(record)
        if source_key in seen_source_keys:
            raise IngestInputError(
                "duplicate_document_identity",
                f"doc_to_md manifest rows produce duplicate document identity: {record['source_path']}",
            )
        seen_source_keys.add(source_key)
        source_id = _stable_id("src", source_key)
        doc_id = _stable_id("doc", source_key)
        source_hash = _hash_text(
            "\n".join(
                [
                    record["source_type"],
                    record["source_path"],
                    record["content_hash"],
                    json.dumps(
                        _stable_provenance_for_hash(record["provenance"]),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ]
            )
        )
        source_row = {
            "schema_name": "md_to_rag.source",
            "schema_version": "1.0",
            "source_id": source_id,
            "source_type": record["source_type"],
            "source_path": record["source_path"],
            "source_hash": source_hash,
            "content_hash": record["content_hash"],
            "metadata": record["metadata"],
            "provenance": record["provenance"],
        }
        if record.get("manifest_path") is not None:
            source_row["manifest_path"] = record["manifest_path"]
        if record.get("manifest_row_index") is not None:
            source_row["manifest_row_index"] = record["manifest_row_index"]
        if record.get("upstream_source_path") is not None:
            source_row["upstream_source_path"] = record["upstream_source_path"]
        if record.get("upstream_document_id") is not None:
            source_row["upstream_document_id"] = record["upstream_document_id"]

        document_row = {
            "schema_name": "md_to_rag.document",
            "schema_version": "1.0",
            "doc_id": doc_id,
            "source_id": source_id,
            "source_path": record["source_path"],
            "source_hash": source_hash,
            "content_hash": record["content_hash"],
            "content": record["content"],
            "line_count": len(record["content"].splitlines()),
            "metadata": record["metadata"],
            "provenance": record["provenance"],
        }
        source_rows.append(source_row)
        document_rows.append(document_row)

    return source_rows, document_rows


def _read_doc_to_md_rows(manifest_path: Path) -> list[dict[str, Any]]:
    try:
        if manifest_path.suffix.lower() == ".jsonl":
            rows = [
                json.loads(line)
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                rows = raw
            elif isinstance(raw, dict):
                rows = _first_list(raw, ("documents", "files", "items", "records"))
                if rows is None:
                    rows = [raw]
            else:
                raise ValueError("manifest root must be an object, array, or JSONL objects")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise IngestInputError(
            "source_manifest_invalid",
            f"Could not read a valid doc_to_md manifest at {manifest_path}: {error}",
        ) from error

    invalid_indexes = [
        index for index, row in enumerate(rows) if not isinstance(row, dict)
    ]
    if invalid_indexes:
        raise IngestInputError(
            "source_manifest_invalid",
            f"doc_to_md manifest rows must be JSON objects; invalid row indexes: {invalid_indexes}",
        )
    return rows


def _metadata_from_manifest_row(row: dict[str, Any], content: str, path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    row_metadata = row.get("metadata")
    if isinstance(row_metadata, dict):
        metadata.update(row_metadata)

    title = row.get("title")
    if not isinstance(title, str) or not title.strip():
        title = _title_from_markdown(content, path)
    metadata["title"] = title.strip()
    return metadata


def _title_from_markdown(content: str, path: Path) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                return title
    return path.stem


def _read_markdown(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise IngestInputError(
            "source_decode_failed",
            f"Markdown source is not valid UTF-8: {path}",
        ) from error
    except OSError as error:
        raise IngestInputError(
            "source_read_failed",
            f"Could not read Markdown source {path}: {error}",
        ) from error


def _update_manifest_status(
    manifest_path: Path,
    manifest: ProjectManifest,
    data: IngestResponseData,
) -> None:
    status = ManifestCommandStatus(
        command=CommandName.INGEST,
        status=CommandStatus.OK,
        message="Ingest artifacts generated.",
        artifact_path=DOCUMENTS_PATH,
        updated_at=_utc_now(),
        data={
            "document_count": data.document_count,
            "source_count": data.source_count,
            "source_manifest_path": data.source_manifest_path,
            "documents_path": data.documents_path,
            "source_manifest_hash": data.source_manifest_hash,
            "documents_hash": data.documents_hash,
        },
    )
    command_status = []
    replaced = False
    for existing_status in manifest.command_status:
        if existing_status.command is CommandName.INGEST:
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


def _manifest_status_matches(manifest: ProjectManifest, data: IngestResponseData) -> bool:
    for existing_status in manifest.command_status:
        if existing_status.command is not CommandName.INGEST:
            continue
        return (
            existing_status.status is CommandStatus.OK
            and existing_status.artifact_path == DOCUMENTS_PATH
            and existing_status.data.get("document_count") == data.document_count
            and existing_status.data.get("source_count") == data.source_count
            and existing_status.data.get("source_manifest_path") == data.source_manifest_path
            and existing_status.data.get("documents_path") == data.documents_path
            and existing_status.data.get("source_manifest_hash") == data.source_manifest_hash
            and existing_status.data.get("documents_hash") == data.documents_hash
        )
    return False


def _input_error_result(error: IngestInputError, context: _ProjectContext) -> IngestProjectResult:
    return IngestProjectResult(
        status=error.status,
        message=error.message,
        data=IngestErrorData(
            project_root=str(context.project_root),
            manifest_path=str(context.manifest_path),
            source_path=context.source_path_relative,
        ),
        error=error.to_command_error(),
    )


def _resolve_user_path(path: str | Path) -> Path:
    user_path = Path(path).expanduser()
    if user_path.is_absolute():
        return user_path.resolve()
    return (Path.cwd() / user_path).resolve()


def _relative_to_project(path: Path, project_root: Path) -> str | IngestInputError:
    try:
        relative = path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return IngestInputError(
            "source_outside_project",
            f"Ingest source must be inside the initialized project: {path}",
        )
    return relative.as_posix()


def _relative_to_project_or_raise(path: Path, project_root: Path) -> str:
    relative = _relative_to_project(path, project_root)
    if isinstance(relative, IngestInputError):
        raise relative
    return relative


def _project_path_from_manifest_value(value: Any, project_root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise IngestInputError(
            "manifest_row_missing_markdown_path",
            "doc_to_md manifest Markdown path must be a non-empty string.",
    )

    relative = _portable_project_relative_markdown_path(value)
    path = (project_root / relative).resolve()
    return path


def _portable_project_relative_markdown_path(value: str) -> str:
    normalized_text = value.replace("\\", "/").strip()
    posix_path = PurePosixPath(normalized_text)
    windows_path = PureWindowsPath(normalized_text)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in posix_path.parts
    ):
        raise IngestInputError(
            "manifest_path_not_portable",
            f"Manifest row Markdown path must be project-relative and portable: {value}",
        )
    relative = posix_path.as_posix()
    if relative in {SOURCE_MANIFEST_PATH, DOCUMENTS_PATH}:
        raise IngestInputError(
            "source_artifact_collision",
            f"doc_to_md manifest row cannot point to a generated md_to_rag artifact: {relative}",
        )
    if posix_path.suffix.lower() not in MARKDOWN_SUFFIXES:
        raise IngestInputError(
            "unsupported_source",
            f"Manifest row Markdown path must point to a Markdown file: {value}",
        )
    return relative


def _portable_upstream_path(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    normalized_text = value.replace("\\", "/").strip()
    posix_path = PurePosixPath(normalized_text)
    windows_path = PureWindowsPath(normalized_text)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in posix_path.parts
    ):
        raise IngestInputError(
            "manifest_path_not_portable",
            f"Manifest upstream source path must be relative and portable: {value}",
        )
    return posix_path.as_posix()


def _first_value(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def _first_list(row: dict[str, Any], keys: tuple[str, ...]) -> list[Any] | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, list):
            return value
    return None


def _source_key(record: dict[str, Any]) -> str:
    if record["source_type"] == "doc_to_md_manifest":
        return record["identity_key"]
    return record["source_path"]


def _stable_provenance_for_hash(provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in provenance.items()
        if key != "manifest_row_index"
    }


def _doc_to_md_identity_key(
    manifest_path: str,
    markdown_path: str,
    upstream_source: str | None,
    upstream_document_id: str | None,
) -> str:
    identity_parts = [
        manifest_path,
        markdown_path,
        upstream_source or "",
        upstream_document_id or "",
    ]
    return "\n".join(identity_parts)


def _upstream_document_id(row: dict[str, Any]) -> str | None:
    value = _first_value(
        row,
        ("document_id", "doc_id", "source_id", "id", "original_id"),
    )
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}_{sha256(value.encode('utf-8')).hexdigest()[:16]}"


def _hash_text(text: str) -> str:
    return _hash_bytes(text.encode("utf-8"))


def _hash_bytes(data: bytes) -> str:
    return f"sha256:{sha256(data).hexdigest()}"


def _jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )


def _write_if_changed(path: Path, text: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    if path.exists() and path.read_bytes() == data:
        return False
    path.write_bytes(data)
    return True
