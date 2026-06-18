from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable

from .manifest import (
    MANIFEST_FILENAME,
    ManifestReadError,
    ManifestWriteError,
    _nearest_existing_ancestor,
    _read_manifest,
    _utc_now,
    _write_manifest,
)
from .ingest import (
    WINDOWS_RESERVED_BASENAMES,
    _find_manifest_lexical,
    _nearest_nested_manifest,
)
from .schemas import (
    ChunkErrorData,
    ChunkResponseData,
    CommandError,
    CommandName,
    CommandStatus,
    ManifestCommandStatus,
    ProjectManifest,
)


DOCUMENTS_PATH = "documents/documents.jsonl"
CHUNKS_PATH = "chunks/chunks.jsonl"
MAX_CHUNK_CHARS = 2000


@dataclass(frozen=True)
class ChunkProjectResult:
    status: CommandStatus
    message: str
    data: ChunkResponseData | ChunkErrorData
    artifact_path: str | None = None
    error: CommandError | None = None


@dataclass(frozen=True)
class _ProjectContext:
    project_root: Path
    manifest_path: Path
    manifest: ProjectManifest
    documents_path: Path
    documents_path_relative: str


@dataclass(frozen=True)
class _ChunkBlock:
    content: str
    line_start: int
    line_end: int
    heading_path: list[str]


class ChunkInputError(Exception):
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


def chunk_project(manifest: str | Path | None = None) -> ChunkProjectResult:
    context_result = _resolve_context(manifest)
    if isinstance(context_result, ChunkProjectResult):
        return context_result
    context = context_result

    try:
        chunks_path = context.project_root / CHUNKS_PATH
        _reject_output_path_outside_project(context, chunks_path, CHUNKS_PATH)
        document_rows, documents_hash = _read_document_rows(context.documents_path)
        chunk_rows = _chunk_rows(document_rows)
        chunks_text = _jsonl_text(chunk_rows)
        chunks_changed = _write_if_changed(chunks_path, chunks_text)
    except ChunkInputError as error:
        return _input_error_result(error, context)
    except OSError as error:
        chunk_error = ChunkInputError(
            "chunk_io_failed",
            f"Could not generate chunk artifacts: {error}",
        )
        return _input_error_result(chunk_error, context)

    data = ChunkResponseData(
        project_root=str(context.project_root),
        manifest_path=str(context.manifest_path),
        documents_path=context.documents_path_relative,
        changed=False,
        document_count=len(document_rows),
        chunk_count=len(chunk_rows),
        chunks_path=CHUNKS_PATH,
        documents_hash=documents_hash,
        chunks_hash=_hash_text(chunks_text),
    )
    manifest_status_changed = not _manifest_status_matches(context.manifest, data)
    changed = chunks_changed or manifest_status_changed
    data = data.model_copy(update={"changed": changed})

    if changed:
        try:
            _update_manifest_status(context.manifest_path, context.manifest, data)
        except ManifestWriteError as error:
            return ChunkProjectResult(
                status=CommandStatus.ERROR,
                message=error.message,
                data=data,
                artifact_path=str(chunks_path.resolve()),
                error=error.to_command_error(),
            )
        message = "Chunk artifacts generated."
    else:
        message = "Chunk artifacts unchanged."

    return ChunkProjectResult(
        status=CommandStatus.OK,
        message=message,
        data=data,
        artifact_path=str(chunks_path.resolve()),
    )


def _resolve_context(manifest: str | Path | None) -> _ProjectContext | ChunkProjectResult:
    if manifest is None:
        manifest_path = _find_manifest_lexical(Path.cwd())
        requested_documents_path: Path | None = None
    else:
        requested_documents_path = _resolve_user_path(manifest)
        anchor = (
            requested_documents_path
            if requested_documents_path.exists()
            else _nearest_existing_ancestor(requested_documents_path)
        )
        manifest_path = _find_manifest_lexical(anchor)

    if manifest_path is None:
        return ChunkProjectResult(
            status=CommandStatus.MISSING_ARTIFACT,
            message=f"No {MANIFEST_FILENAME} found for documents artifact.",
            data=ChunkErrorData(
                documents_path=str(_resolve_user_path(manifest)) if manifest is not None else None,
            ),
            error=CommandError(
                code="manifest_not_found",
                message=f"No {MANIFEST_FILENAME} found for documents artifact.",
            ),
        )

    try:
        project_manifest = _read_manifest(manifest_path)
    except ManifestReadError as error:
        project_root = manifest_path.parent.resolve()
        return ChunkProjectResult(
            status=CommandStatus.ERROR,
            message=error.message,
            data=ChunkErrorData(
                project_root=str(project_root),
                manifest_path=str(manifest_path.resolve()),
                documents_path=str(manifest) if manifest is not None else None,
            ),
            error=error.to_command_error(),
        )

    project_root = manifest_path.parent.resolve()
    if requested_documents_path is None:
        requested_documents_path = project_root / DOCUMENTS_PATH

    try:
        documents_path = requested_documents_path.resolve()
    except (OSError, RuntimeError, ValueError) as error:
        message = f"Could not resolve documents artifact: {requested_documents_path}"
        return ChunkProjectResult(
            status=CommandStatus.ERROR,
            message=message,
            data=ChunkErrorData(
                project_root=str(project_root),
                manifest_path=str(manifest_path.resolve()),
                documents_path=str(requested_documents_path),
            ),
            error=CommandError(
                code="documents_path_unresolvable",
                message=f"{message}: {error}",
            ),
        )

    documents_relative = _relative_to_project(documents_path, project_root)
    if isinstance(documents_relative, ChunkInputError):
        return _input_error_result(
            documents_relative,
            _ProjectContext(
                project_root=project_root,
                manifest_path=manifest_path.resolve(),
                manifest=project_manifest,
                documents_path=documents_path,
                documents_path_relative=str(documents_path),
            ),
        )
    portability_error = _documents_artifact_path_error(documents_relative)
    if portability_error is not None:
        return _input_error_result(
            portability_error,
            _ProjectContext(
                project_root=project_root,
                manifest_path=manifest_path.resolve(),
                manifest=project_manifest,
                documents_path=documents_path,
                documents_path_relative=documents_relative,
            ),
        )
    if documents_relative == CHUNKS_PATH:
        return _input_error_result(
            ChunkInputError(
                "documents_artifact_collision",
                f"Documents artifact cannot be a generated md_to_rag chunk artifact: {documents_relative}",
            ),
            _ProjectContext(
                project_root=project_root,
                manifest_path=manifest_path.resolve(),
                manifest=project_manifest,
                documents_path=documents_path,
                documents_path_relative=documents_relative,
            ),
        )
    nested_manifest_path = _nearest_nested_manifest(
        documents_path,
        project_root,
        manifest_path,
    )
    if nested_manifest_path is not None:
        return _input_error_result(
            ChunkInputError(
                "documents_nested_project",
                "Documents artifact resolves inside a nested initialized project; "
                "use that project path directly.",
            ),
            _ProjectContext(
                project_root=project_root,
                manifest_path=manifest_path.resolve(),
                manifest=project_manifest,
                documents_path=documents_path,
                documents_path_relative=documents_relative,
            ),
        )

    if not documents_path.exists():
        return ChunkProjectResult(
            status=CommandStatus.MISSING_ARTIFACT,
            message=f"Documents artifact does not exist: {documents_relative}",
            data=ChunkErrorData(
                project_root=str(project_root),
                manifest_path=str(manifest_path.resolve()),
                documents_path=documents_relative,
            ),
            error=CommandError(
                code="documents_not_found",
                message=f"Documents artifact does not exist: {documents_relative}",
            ),
        )

    return _ProjectContext(
        project_root=project_root,
        manifest_path=manifest_path.resolve(),
        manifest=project_manifest,
        documents_path=documents_path,
        documents_path_relative=documents_relative,
    )


def _read_document_rows(documents_path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        data = documents_path.read_bytes()
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ChunkInputError(
            "documents_invalid_jsonl",
            f"Documents artifact is not valid UTF-8: {documents_path}",
        ) from error
    except OSError as error:
        raise ChunkInputError(
            "documents_read_failed",
            f"Could not read documents artifact {documents_path}: {error}",
        ) from error

    rows: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = _json_loads_strict(line)
        except (json.JSONDecodeError, ValueError) as error:
            raise ChunkInputError(
                "documents_invalid_jsonl",
                f"Documents artifact contains invalid JSONL at line {line_number}: {error}",
            ) from error
        if not isinstance(row, dict):
            raise ChunkInputError(
                "documents_invalid_jsonl",
                f"Documents artifact row {line_number} must be a JSON object.",
            )
        _validate_document_row(row, line_number)
        doc_id = row["doc_id"]
        if doc_id in seen_doc_ids:
            raise ChunkInputError(
                "duplicate_document_id",
                f"Documents artifact contains duplicate doc_id at line {line_number}: {doc_id}",
            )
        seen_doc_ids.add(doc_id)
        rows.append(row)
    return rows, _hash_bytes(data)


def _validate_document_row(row: dict[str, Any], line_number: int) -> None:
    if row.get("schema_name") != "md_to_rag.document" or row.get("schema_version") != "1.0":
        raise ChunkInputError(
            "document_schema_invalid",
            f"Documents artifact row {line_number} is not an md_to_rag.document v1.0 row.",
        )

    required_strings = (
        "doc_id",
        "source_id",
        "source_path",
        "source_hash",
        "content_hash",
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
        raise ChunkInputError(
            "document_schema_invalid",
            f"Documents artifact row {line_number} has invalid string field(s): {fields}.",
        )
    for field in required_strings:
        _validate_utf8_string(row[field], field, line_number)
    _validate_document_source_path(row["source_path"], line_number)
    line_count = row.get("line_count")
    expected_line_count = len(row["content"].splitlines())
    if not isinstance(line_count, int) or isinstance(line_count, bool) or line_count < 0:
        raise ChunkInputError(
            "document_schema_invalid",
            f"Documents artifact row {line_number} has invalid line_count.",
        )
    if line_count != expected_line_count:
        raise ChunkInputError(
            "document_schema_invalid",
            f"Documents artifact row {line_number} line_count does not match content.",
        )
    if row["content_hash"] != _hash_text(row["content"]):
        raise ChunkInputError(
            "document_schema_invalid",
            f"Documents artifact row {line_number} content_hash does not match content.",
        )
    if not isinstance(row.get("metadata"), dict):
        raise ChunkInputError(
            "document_schema_invalid",
            f"Documents artifact row {line_number} has invalid metadata.",
        )
    if not isinstance(row.get("provenance"), dict):
        raise ChunkInputError(
            "document_schema_invalid",
            f"Documents artifact row {line_number} has invalid provenance.",
        )


def _validate_utf8_string(value: str, field: str, line_number: int) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ChunkInputError(
            "document_schema_invalid",
            f"Documents artifact row {line_number} has invalid UTF-8 string field: {field}.",
        ) from error


def _validate_document_source_path(value: str, line_number: int) -> None:
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
        raise ChunkInputError(
            "document_schema_invalid",
            f"Documents artifact row {line_number} has non-portable source_path.",
        )


def _documents_artifact_path_error(value: str) -> ChunkInputError | None:
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
        return ChunkInputError(
            "documents_path_not_portable",
            f"Documents artifact path must be project-relative and portable: {value}",
        )
    return None


def _has_windows_reserved_path_component(part: str) -> bool:
    normalized = part.rstrip(" .")
    basename = normalized.split(".", 1)[0].upper()
    return (
        any(character in part for character in '<>:"|?*')
        or any(ord(character) < 32 for character in part)
        or normalized != part
        or basename in WINDOWS_RESERVED_BASENAMES
    )


def _chunk_rows(document_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document_row in document_rows:
        blocks = _chunk_content(document_row["content"])
        for chunk_index, block in enumerate(blocks):
            content_hash = _hash_text(block.content)
            chunk_id = _stable_chunk_id(
                document_row["doc_id"],
                document_row["content_hash"],
                chunk_index,
                content_hash,
            )
            rows.append(
                {
                    "schema_name": "md_to_rag.chunk",
                    "schema_version": "1.0",
                    "chunk_id": chunk_id,
                    "doc_id": document_row["doc_id"],
                    "source_id": document_row["source_id"],
                    "source_path": document_row["source_path"],
                    "source_hash": document_row["source_hash"],
                    "content_hash": content_hash,
                    "document_content_hash": document_row["content_hash"],
                    "chunk_index": chunk_index,
                    "content": block.content,
                    "line_start": block.line_start,
                    "line_end": block.line_end,
                    "heading_path": block.heading_path,
                    "metadata": document_row["metadata"],
                    "provenance": document_row["provenance"],
                }
            )
    return rows


def _chunk_content(content: str) -> list[_ChunkBlock]:
    blocks: list[_ChunkBlock] = []
    current_lines: list[tuple[int, str]] = []
    heading_paths = _heading_paths_by_line(content)
    fence_marker: tuple[str, int] | None = None

    def flush() -> None:
        nonlocal current_lines
        if not current_lines:
            return
        blocks.extend(_split_large_block(current_lines, heading_paths))
        current_lines = []

    for line_number, line in enumerate(content.splitlines(), start=1):
        if fence_marker is not None:
            current_lines.append((line_number, line))
            if _closes_fence(_closing_fence_marker(line), fence_marker):
                fence_marker = None
                flush()
            continue

        marker = _opening_fence_marker(line)
        if marker is not None:
            flush()
            fence_marker = marker
            current_lines.append((line_number, line))
            continue

        if (
            _markdown_setext_underline(line) is not None
            and _setext_heading_title(current_lines) is not None
        ):
            current_lines.append((line_number, line))
            flush()
            continue

        if not line.strip():
            flush()
            continue

        if _markdown_heading(line) is not None:
            flush()
            current_lines.append((line_number, line))
            flush()
            continue

        current_lines.append((line_number, line))
    flush()
    return blocks


def _heading_paths_by_line(content: str) -> dict[int, list[str]]:
    paths: dict[int, list[str]] = {}
    heading_stack: list[str] = []
    fence_marker: tuple[str, int] | None = None
    pending_lines: list[tuple[int, str]] = []

    def flush_pending_lines() -> None:
        nonlocal pending_lines
        for pending_line_number, _ in pending_lines:
            paths[pending_line_number] = list(heading_stack)
        pending_lines = []

    for line_number, line in enumerate(content.splitlines(), start=1):
        if fence_marker is not None:
            if _closes_fence(_closing_fence_marker(line), fence_marker):
                fence_marker = None
            paths[line_number] = list(heading_stack)
            continue

        marker = _opening_fence_marker(line)
        if marker is not None:
            flush_pending_lines()
            fence_marker = marker
            paths[line_number] = list(heading_stack)
            continue

        setext_level = _markdown_setext_underline(line)
        setext_title = (
            _setext_heading_title(pending_lines)
            if setext_level is not None
            else None
        )
        if setext_level is not None and setext_title is not None:
            heading_stack = heading_stack[: setext_level - 1]
            heading_stack.append(setext_title)
            for pending_line_number, _ in pending_lines:
                paths[pending_line_number] = list(heading_stack)
            paths[line_number] = list(heading_stack)
            pending_lines = []
            continue

        if not line.strip():
            flush_pending_lines()
            paths[line_number] = list(heading_stack)
            continue

        heading = _markdown_heading(line)
        if heading is not None:
            flush_pending_lines()
            level, title = heading
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)
            paths[line_number] = list(heading_stack)
            continue

        pending_lines.append((line_number, line))
    flush_pending_lines()
    return paths


def _opening_fence_marker(line: str) -> tuple[str, int] | None:
    marker = _raw_fence_marker(line)
    if marker is None:
        return None
    marker_character, marker_length, rest = marker
    if marker_character == "`" and "`" in rest:
        return None
    return marker_character, marker_length


def _closing_fence_marker(line: str) -> tuple[str, int] | None:
    marker = _raw_fence_marker(line)
    if marker is None:
        return None
    marker_character, marker_length, rest = marker
    if rest.strip():
        return None
    return marker_character, marker_length


def _raw_fence_marker(line: str) -> tuple[str, int, str] | None:
    leading_whitespace = line[: len(line) - len(line.lstrip(" \t"))]
    if "\t" in leading_whitespace or len(leading_whitespace) > 3:
        return None
    stripped = line[len(leading_whitespace):]
    backtick_count = len(stripped) - len(stripped.lstrip("`"))
    if backtick_count >= 3:
        return ("`", backtick_count, stripped[backtick_count:])
    tilde_count = len(stripped) - len(stripped.lstrip("~"))
    if tilde_count >= 3:
        return ("~", tilde_count, stripped[tilde_count:])
    return None


def _closes_fence(
    marker: tuple[str, int] | None,
    opening_marker: tuple[str, int] | None,
) -> bool:
    if marker is None or opening_marker is None:
        return False
    marker_character, marker_length = marker
    opening_character, opening_length = opening_marker
    return marker_character == opening_character and marker_length >= opening_length


def _markdown_heading(line: str) -> tuple[int, str] | None:
    if line.startswith("\t"):
        return None
    leading_spaces = len(line) - len(line.lstrip(" "))
    if leading_spaces > 3:
        return None
    stripped = line[leading_spaces:]
    level = len(stripped) - len(stripped.lstrip("#"))
    if level < 1 or level > 6:
        return None
    heading_text = stripped[level:]
    if heading_text and not heading_text[0].isspace():
        return None
    title = _strip_atx_closing_sequence(heading_text.strip())
    if not title:
        return None
    return level, title


def _markdown_setext_underline(line: str) -> int | None:
    leading_whitespace = line[: len(line) - len(line.lstrip(" \t"))]
    if "\t" in leading_whitespace or len(leading_whitespace) > 3:
        return None
    marker = line[len(leading_whitespace):].strip()
    if marker and all(character == "=" for character in marker):
        return 1
    if marker and all(character == "-" for character in marker):
        return 2
    return None


def _setext_heading_title(lines: list[tuple[int, str]]) -> str | None:
    if not lines:
        return None
    title_lines: list[str] = []
    for _, line in lines:
        leading_whitespace = line[: len(line) - len(line.lstrip(" \t"))]
        if "\t" in leading_whitespace or len(leading_whitespace) > 3:
            return None
        title = line[len(leading_whitespace):].strip()
        if not title:
            return None
        title_lines.append(title)
    return " ".join(title_lines)


def _strip_atx_closing_sequence(title: str) -> str:
    stripped = title.rstrip()
    hash_start = len(stripped)
    while hash_start > 0 and stripped[hash_start - 1] == "#":
        hash_start -= 1
    if hash_start > 0 and hash_start < len(stripped) and stripped[hash_start - 1].isspace():
        return stripped[:hash_start].rstrip()
    return title


def _split_large_block(
    lines: list[tuple[int, str]],
    heading_paths: dict[int, list[str]],
) -> list[_ChunkBlock]:
    if _opening_fence_marker(lines[0][1]) is not None:
        return _split_fenced_block(lines, heading_paths)

    return _split_line_limited_block(lines, heading_paths)


def _split_line_limited_block(
    lines: list[tuple[int, str]],
    heading_paths: dict[int, list[str]],
) -> list[_ChunkBlock]:
    chunks: list[_ChunkBlock] = []
    current: list[tuple[int, str]] = []
    current_length = 0

    def flush_current() -> None:
        nonlocal current, current_length
        if current:
            chunks.append(_block_from_lines(current, heading_paths))
            current = []
            current_length = 0

    for line_number, line in lines:
        if len(line) > MAX_CHUNK_CHARS:
            flush_current()
            for start in range(0, len(line), MAX_CHUNK_CHARS):
                chunks.append(
                    _ChunkBlock(
                        content=line[start:start + MAX_CHUNK_CHARS],
                        line_start=line_number,
                        line_end=line_number,
                        heading_path=heading_paths.get(line_number, []),
                    )
                )
            continue
        line_length = len(line)
        next_length = line_length if not current else current_length + 1 + line_length
        if current and next_length > MAX_CHUNK_CHARS:
            flush_current()
            next_length = line_length
        current.append((line_number, line))
        current_length = next_length
    flush_current()
    return chunks


def _split_fenced_block(
    lines: list[tuple[int, str]],
    heading_paths: dict[int, list[str]],
) -> list[_ChunkBlock]:
    full_block = _block_from_lines(lines, heading_paths)
    if len(full_block.content) <= MAX_CHUNK_CHARS:
        return [full_block]

    opening_marker = _opening_fence_marker(lines[0][1])
    if opening_marker is None:
        return _split_line_limited_block(lines, heading_paths)

    has_closing_fence = len(lines) > 1 and _closes_fence(
        _closing_fence_marker(lines[-1][1]),
        opening_marker,
    )
    if not has_closing_fence:
        return _split_line_limited_block(lines, heading_paths)

    opening_line_number, opening_text = lines[0]
    closing_line_number = lines[-1][0]
    closing_text = lines[-1][1]
    inner_lines = lines[1:-1]
    inner_budget = MAX_CHUNK_CHARS - len(opening_text) - len(closing_text) - 2
    if inner_budget < 1:
        return _split_line_limited_block(lines, heading_paths)

    chunks: list[_ChunkBlock] = []
    current: list[tuple[int, str]] = []
    current_length = 0

    def append_chunk(chunk_lines: list[tuple[int, str]]) -> None:
        inner_content = "\n".join(line for _, line in chunk_lines)
        chunks.append(
            _ChunkBlock(
                content=f"{opening_text}\n{inner_content}\n{closing_text}",
                line_start=opening_line_number,
                line_end=closing_line_number,
                heading_path=heading_paths.get(opening_line_number, []),
            )
        )

    def flush_current() -> None:
        nonlocal current, current_length
        if current:
            append_chunk(current)
            current = []
            current_length = 0

    for line_number, line in inner_lines:
        if len(line) > inner_budget:
            flush_current()
            for start in range(0, len(line), inner_budget):
                append_chunk([(line_number, line[start:start + inner_budget])])
            continue
        next_length = len(line) if not current else current_length + 1 + len(line)
        if current and next_length > inner_budget:
            flush_current()
            next_length = len(line)
        current.append((line_number, line))
        current_length = next_length
    flush_current()

    if not chunks:
        return [full_block]
    return chunks


def _block_from_lines(
    lines: list[tuple[int, str]],
    heading_paths: dict[int, list[str]],
) -> _ChunkBlock:
    return _ChunkBlock(
        content="\n".join(line for _, line in lines),
        line_start=lines[0][0],
        line_end=lines[-1][0],
        heading_path=heading_paths.get(lines[0][0], []),
    )


def _update_manifest_status(
    manifest_path: Path,
    manifest: ProjectManifest,
    data: ChunkResponseData,
) -> None:
    status = ManifestCommandStatus(
        command=CommandName.CHUNK,
        status=CommandStatus.OK,
        message="Chunk artifacts generated.",
        artifact_path=CHUNKS_PATH,
        updated_at=_utc_now(),
        data={
            "document_count": data.document_count,
            "chunk_count": data.chunk_count,
            "documents_path": data.documents_path,
            "chunks_path": data.chunks_path,
            "documents_hash": data.documents_hash,
            "chunks_hash": data.chunks_hash,
        },
    )
    command_status = []
    replaced = False
    for existing_status in manifest.command_status:
        if existing_status.command is CommandName.CHUNK:
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


def _manifest_status_matches(manifest: ProjectManifest, data: ChunkResponseData) -> bool:
    for existing_status in manifest.command_status:
        if existing_status.command is not CommandName.CHUNK:
            continue
        return (
            existing_status.status is CommandStatus.OK
            and existing_status.artifact_path == CHUNKS_PATH
            and existing_status.data.get("document_count") == data.document_count
            and existing_status.data.get("chunk_count") == data.chunk_count
            and existing_status.data.get("documents_path") == data.documents_path
            and existing_status.data.get("chunks_path") == data.chunks_path
            and existing_status.data.get("documents_hash") == data.documents_hash
            and existing_status.data.get("chunks_hash") == data.chunks_hash
        )
    return False


def _input_error_result(error: ChunkInputError, context: _ProjectContext) -> ChunkProjectResult:
    return ChunkProjectResult(
        status=error.status,
        message=error.message,
        data=ChunkErrorData(
            project_root=str(context.project_root),
            manifest_path=str(context.manifest_path),
            documents_path=context.documents_path_relative,
        ),
        error=error.to_command_error(),
    )


def _resolve_user_path(path: str | Path) -> Path:
    user_path = Path(path).expanduser()
    if user_path.is_absolute():
        return user_path
    return Path.cwd() / user_path


def _relative_to_project(path: Path, project_root: Path) -> str | ChunkInputError:
    try:
        relative = path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return ChunkInputError(
            "documents_outside_project",
            f"Documents artifact must be inside the initialized project: {path}",
        )
    return relative.as_posix()


def _reject_output_path_outside_project(
    context: _ProjectContext,
    output_path: Path,
    artifact_path: str,
) -> None:
    try:
        resolved_output = output_path.resolve()
    except (OSError, RuntimeError, ValueError) as error:
        raise ChunkInputError(
            "artifact_path_collision",
            f"Generated artifact path cannot be resolved safely: {artifact_path}",
        ) from error
    relative = _relative_to_project(resolved_output, context.project_root)
    if isinstance(relative, ChunkInputError):
        raise ChunkInputError(
            "artifact_path_outside_project",
            f"Generated artifact path must stay inside the initialized project: {artifact_path}",
        )
    if resolved_output != output_path.absolute():
        raise ChunkInputError(
            "artifact_path_collision",
            f"Generated artifact path cannot be a symlink or linked path: {artifact_path}",
        )
    if output_path.exists() and output_path.stat().st_nlink > 1:
        raise ChunkInputError(
            "artifact_path_collision",
            f"Generated artifact path cannot be a hard-linked path: {artifact_path}",
        )


def _stable_chunk_id(
    doc_id: str,
    document_content_hash: str,
    chunk_index: int,
    content_hash: str,
) -> str:
    value = "\n".join([doc_id, document_content_hash, str(chunk_index), content_hash])
    return f"chk_{sha256(value.encode('utf-8')).hexdigest()[:16]}"


def _hash_text(text: str) -> str:
    return _hash_bytes(text.encode("utf-8"))


def _hash_bytes(data: bytes) -> str:
    return f"sha256:{sha256(data).hexdigest()}"


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


def _jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    try:
        return "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            for row in rows
        )
    except ValueError as error:
        raise ChunkInputError(
            "documents_invalid_jsonl",
            f"Chunk artifacts must be portable JSONL with finite values: {error}",
        ) from error


def _write_if_changed(path: Path, text: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    if path.exists() and path.read_bytes() == data:
        return False
    path.write_bytes(data)
    return True
