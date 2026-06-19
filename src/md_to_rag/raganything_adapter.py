from __future__ import annotations

import importlib
import inspect
import json
import re
from dataclasses import dataclass, field
from math import isfinite
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, JsonValue


_UPSTREAM_MODULE_NAME = "raganything"
_MANAGED_CONFIG_KEYS = {
    "working_dir",
    "parser",
    "parse_method",
    "enable_image_processing",
    "enable_table_processing",
    "enable_equation_processing",
    "llm_model_func",
    "embedding_func",
}
_SECRET_KEY_PATTERN = re.compile(
    r"(^|_)(api_?key|auth_?header|authorization|bearer|cookie|credentials?|password|"
    r"private_?key|secret|subscription_?key|access_?token|refresh_?token)($|_)",
    re.IGNORECASE,
)
_SECRET_COMPACT_KEYS = {
    "apikey",
    "auth",
    "authheaders",
    "authheader",
    "authentication",
    "authmethod",
    "authorization",
    "bearer",
    "cookie",
    "cookies",
    "credential",
    "credentials",
    "jwt",
    "key",
    "keyid",
    "keyids",
    "keys",
    "pat",
    "pats",
    "password",
    "passwords",
    "privatekey",
    "secret",
    "secrets",
    "subscriptionkey",
    "token",
    "tokens",
}
_SECRET_SUFFIXES = (
    "auth",
    "authentication",
    "key",
    "keyid",
    "keyids",
    "keys",
    "pat",
    "pats",
    "token",
    "tokens",
    "secret",
    "secrets",
    "password",
    "passwords",
)
_NON_SECRET_COMPACT_KEYS = {
    "chunkmaxtokens",
    "chunkoverlaptokens",
    "maxcontexttokens",
    "maxentitytokens",
    "maxrelationtokens",
    "maxtokens",
    "maxtotaltokens",
    "summarymaxtokens",
    "tokencount",
    "tokenizer",
    "tokenlimit",
    "tokensize",
}
_ALLOWED_LIGHTRAG_OPTION_KEYS = {
    "chunk_overlap_token_size",
    "chunk_token_size",
    "max_entity_tokens",
    "max_relation_tokens",
    "max_total_tokens",
    "summary_context_size",
    "summary_max_tokens",
    "tiktoken_model_name",
}
_LIGHTRAG_POSITIVE_INT_OPTION_KEYS = {
    "chunk_token_size",
    "max_entity_tokens",
    "max_relation_tokens",
    "max_total_tokens",
    "summary_context_size",
    "summary_max_tokens",
}
_LIGHTRAG_NON_NEGATIVE_INT_OPTION_KEYS = {"chunk_overlap_token_size"}
_LIGHTRAG_STRING_OPTION_KEYS = {"tiktoken_model_name"}
_SECRET_QUALIFIER_SUFFIXES = {
    "file",
    "filename",
    "filenames",
    "files",
    "header",
    "headers",
    "name",
    "names",
    "path",
    "paths",
    "value",
    "values",
}
_SECRET_HEADER_NAME_PARTS = {
    "auth",
    "authentication",
    "authorization",
    "bearer",
    "cookie",
    "cookies",
    "credential",
    "credentials",
    "jwt",
    "key",
    "keys",
    "password",
    "passwords",
    "secret",
    "secrets",
    "token",
    "tokens",
}
_CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


class RAGAnythingAdapterError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def __str__(self) -> str:
        return self.message


class RAGAnythingConfigError(RAGAnythingAdapterError):
    pass


class RAGAnythingDependencyError(RAGAnythingAdapterError):
    pass


class RAGAnythingRuntimeError(RAGAnythingAdapterError):
    pass


class RAGAnythingInsertResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inserted: bool
    doc_id: str
    file_path: str


class RAGAnythingQueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    mode: str


@dataclass(frozen=True)
class RAGAnythingAdapterConfig:
    """Internal md_to_rag config for the optional RAG-Anything adapter.

    Upstream touchpoints are intentionally limited to RAGAnythingConfig,
    insert_content_list(...), and aquery(...). This config is internal and is
    not part of the public CLI/API/MCP/artifact contract.
    """

    working_dir: str | Path
    parser: str = "mineru"
    parse_method: str = "auto"
    query_mode: str = "hybrid"
    enable_image_processing: bool = False
    enable_table_processing: bool = True
    enable_equation_processing: bool = True
    display_stats: bool = False
    config_options: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.working_dir, (str, PathLike)):
            _raise_config("RAG-Anything working_dir must be a non-empty path.")
        if isinstance(self.working_dir, str) and not self.working_dir.strip():
            _raise_config("RAG-Anything working_dir must be a non-empty path.")
        working_dir = Path(self.working_dir)
        _validate_string(str(working_dir), "working_dir")
        object.__setattr__(self, "working_dir", working_dir)

        for field_name in ("parser", "parse_method", "query_mode"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                _raise_config(f"RAG-Anything {field_name} must be a non-empty string.")
            _validate_string(value, field_name)

        for field_name in (
            "enable_image_processing",
            "enable_table_processing",
            "enable_equation_processing",
            "display_stats",
        ):
            if not isinstance(getattr(self, field_name), bool):
                _raise_config(f"RAG-Anything {field_name} must be a boolean.")

        object.__setattr__(
            self,
            "config_options",
            MappingProxyType(_validated_lightrag_options(self.config_options)),
        )

    def to_raganything_config_kwargs(self) -> dict[str, JsonValue]:
        return {
            "working_dir": str(self.working_dir),
            "parser": self.parser,
            "parse_method": self.parse_method,
            "enable_image_processing": self.enable_image_processing,
            "enable_table_processing": self.enable_table_processing,
            "enable_equation_processing": self.enable_equation_processing,
        }

    def to_lightrag_kwargs(self) -> dict[str, JsonValue]:
        return _validated_lightrag_options(self.config_options)


class RAGAnythingBackend:
    def __init__(self, rag: Any, config: RAGAnythingAdapterConfig) -> None:
        self._rag = rag
        self._config = config
        self._require_callable("insert_content_list")
        self._require_callable("aquery")

    async def insert_content_list(
        self,
        *,
        content_list: Sequence[Mapping[str, JsonValue]],
        file_path: str,
        doc_id: str,
    ) -> RAGAnythingInsertResult:
        validated_content_list = _validated_content_list(content_list)
        _validate_non_empty_string(file_path, "file_path")
        _validate_non_empty_string(doc_id, "doc_id")
        try:
            result = self._rag.insert_content_list(
                content_list=validated_content_list,
                file_path=file_path,
                doc_id=doc_id,
                display_stats=self._config.display_stats,
            )
            if inspect.isawaitable(result):
                await result
        except RAGAnythingAdapterError:
            raise
        except SystemExit:
            raise RAGAnythingRuntimeError(
                "raganything_insert_failed",
                "Could not insert content into optional RAG-Anything backend.",
            ) from None
        except Exception:
            raise RAGAnythingRuntimeError(
                "raganything_insert_failed",
                "Could not insert content into optional RAG-Anything backend.",
            ) from None
        return RAGAnythingInsertResult(inserted=True, doc_id=doc_id, file_path=file_path)

    async def aquery(self, question: str, *, mode: str | None = None) -> RAGAnythingQueryResult:
        _validate_non_empty_string(question, "question")
        query_mode = self._config.query_mode if mode is None else mode
        _validate_non_empty_string(query_mode, "mode")
        try:
            await self._ensure_query_ready()
            result = self._rag.aquery(question, mode=query_mode)
            if inspect.isawaitable(result):
                result = await result
            answer = _normalized_query_answer(result)
        except RAGAnythingAdapterError:
            raise
        except SystemExit:
            raise RAGAnythingRuntimeError(
                "raganything_query_failed",
                "Could not query optional RAG-Anything backend.",
            ) from None
        except Exception:
            raise RAGAnythingRuntimeError(
                "raganything_query_failed",
                "Could not query optional RAG-Anything backend.",
            ) from None
        return RAGAnythingQueryResult(answer=answer, mode=query_mode)

    async def _ensure_query_ready(self) -> None:
        ensure = getattr(self._rag, "_ensure_lightrag_initialized", None)
        if not callable(ensure):
            return
        try:
            result = ensure()
            if inspect.isawaitable(result):
                result = await result
        except RAGAnythingAdapterError:
            raise
        except SystemExit:
            raise RAGAnythingRuntimeError(
                "raganything_initialization_failed",
                "Could not initialize optional RAG-Anything backend.",
            ) from None
        except Exception:
            raise RAGAnythingRuntimeError(
                "raganything_initialization_failed",
                "Could not initialize optional RAG-Anything backend.",
            ) from None
        if result is False or (isinstance(result, Mapping) and result.get("success") is False):
            raise RAGAnythingRuntimeError(
                "raganything_initialization_failed",
                "Could not initialize optional RAG-Anything backend.",
            )

    def _require_callable(self, name: str) -> None:
        try:
            value = getattr(self._rag, name, None)
        except SystemExit:
            raise RAGAnythingDependencyError(
                "raganything_interface_invalid",
                f"RAG-Anything backend must expose {name}(...).",
            ) from None
        except Exception:
            raise RAGAnythingDependencyError(
                "raganything_interface_invalid",
                f"RAG-Anything backend must expose {name}(...).",
            ) from None
        if not callable(value):
            raise RAGAnythingDependencyError(
                "raganything_interface_invalid",
                f"RAG-Anything backend must expose {name}(...).",
            )


def create_raganything_backend(
    config: RAGAnythingAdapterConfig,
    *,
    raganything_module: Any | None = None,
    llm_model_func: Any | None = None,
    embedding_func: Any | None = None,
) -> RAGAnythingBackend:
    module = _load_raganything_module(raganything_module)
    config_cls = _required_attribute(module, "RAGAnythingConfig")
    rag_cls = _required_attribute(module, "RAGAnything")
    _validate_required_callable(llm_model_func, "llm_model_func")
    _validate_required_embedding_func(embedding_func)
    try:
        upstream_config = config_cls(**config.to_raganything_config_kwargs())
        rag_kwargs = {"config": upstream_config}
        lightrag_kwargs = config.to_lightrag_kwargs()
        if lightrag_kwargs:
            rag_kwargs["lightrag_kwargs"] = lightrag_kwargs
        rag_kwargs["llm_model_func"] = llm_model_func
        rag_kwargs["embedding_func"] = embedding_func
        rag = rag_cls(**rag_kwargs)
    except RAGAnythingAdapterError:
        raise
    except SystemExit:
        raise RAGAnythingRuntimeError(
            "raganything_initialization_failed",
            "Could not initialize optional RAG-Anything backend.",
        ) from None
    except Exception:
        raise RAGAnythingRuntimeError(
            "raganything_initialization_failed",
            "Could not initialize optional RAG-Anything backend.",
        ) from None
    return RAGAnythingBackend(rag, config)


def _load_raganything_module(raganything_module: Any | None) -> Any:
    if raganything_module is not None:
        return raganything_module
    try:
        return importlib.import_module(_UPSTREAM_MODULE_NAME)
    except ImportError:
        raise RAGAnythingDependencyError(
            "raganything_unavailable",
            "Optional RAG-Anything dependency is unavailable. "
            "Install md-to-rag[raganything] to enable the internal adapter.",
        ) from None
    except SystemExit:
        raise RAGAnythingDependencyError(
            "raganything_unavailable",
            "Optional RAG-Anything dependency could not be loaded. "
            "Install md-to-rag[raganything] to enable the internal adapter.",
        ) from None
    except Exception:
        raise RAGAnythingDependencyError(
            "raganything_unavailable",
            "Optional RAG-Anything dependency could not be loaded. "
            "Install md-to-rag[raganything] to enable the internal adapter.",
        ) from None


def _required_attribute(module: Any, name: str) -> Any:
    try:
        value = getattr(module, name, None)
    except SystemExit:
        raise RAGAnythingDependencyError(
            "raganything_interface_invalid",
            "Optional RAG-Anything dependency does not expose the required adapter interface.",
        ) from None
    except Exception:
        raise RAGAnythingDependencyError(
            "raganything_interface_invalid",
            "Optional RAG-Anything dependency does not expose the required adapter interface.",
        ) from None
    if value is None:
        raise RAGAnythingDependencyError(
            "raganything_interface_invalid",
            f"Optional RAG-Anything dependency does not expose {name}.",
        ) from None
    return value


def _validate_required_callable(value: Any | None, field_name: str) -> None:
    if not callable(value):
        _raise_config(f"RAG-Anything {field_name} must be callable.")


def _validate_required_embedding_func(value: Any | None) -> None:
    _validate_required_callable(value, "embedding_func")
    try:
        func = getattr(value, "func")
        embedding_dim = getattr(value, "embedding_dim")
        max_token_size = getattr(value, "max_token_size", None)
    except Exception:
        _raise_config("RAG-Anything embedding_func must be EmbeddingFunc-compatible.")
    if not callable(func):
        _raise_config("RAG-Anything embedding_func.func must be callable.")
    if (
        not isinstance(embedding_dim, int)
        or isinstance(embedding_dim, bool)
        or embedding_dim <= 0
    ):
        _raise_config("RAG-Anything embedding_func.embedding_dim must be a positive integer.")
    if max_token_size is not None and (
        not isinstance(max_token_size, int)
        or isinstance(max_token_size, bool)
        or max_token_size <= 0
    ):
        _raise_config(
            "RAG-Anything embedding_func.max_token_size must be a positive integer."
        )


def _validated_content_list(
    content_list: Sequence[Mapping[str, JsonValue]],
) -> list[dict[str, JsonValue]]:
    if isinstance(content_list, (str, bytes)) or not isinstance(content_list, Sequence):
        _raise_runtime("RAG-Anything content_list must be a sequence of JSON objects.")
    if not content_list:
        _raise_runtime("RAG-Anything content_list must not be empty.")
    validated: list[dict[str, JsonValue]] = []
    for index, item in enumerate(content_list):
        if not isinstance(item, Mapping):
            _raise_runtime(f"RAG-Anything content_list item {index} must be a JSON object.")
        validated.append(
            _validated_json_mapping(
                item,
                f"content_list[{index}]",
                enforce_config_key_policy=False,
            )
        )
    return validated


def _normalized_query_answer(value: Any) -> str:
    if isinstance(value, str):
        _validate_query_answer_string(value, "answer")
        return value
    if isinstance(value, Mapping):
        for key in ("answer", "response", "result", "content", "text"):
            answer = value.get(key)
            if isinstance(answer, str):
                _validate_query_answer_string(answer, key)
                return answer
        try:
            serialized = json.dumps(
                _validated_json_value(
                    value,
                    "query_result",
                    enforce_config_key_policy=False,
                ),
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
        except (RAGAnythingConfigError, TypeError, ValueError):
            raise RAGAnythingRuntimeError(
                "raganything_query_result_invalid",
                "RAG-Anything query result could not be normalized to owned data.",
            ) from None
        return serialized
    if value is None or isinstance(value, (bool, int)):
        return str(value)
    if isinstance(value, float) and isfinite(value):
        return str(value)
    raise RAGAnythingRuntimeError(
        "raganything_query_result_invalid",
        "RAG-Anything query result could not be normalized to owned data.",
    )


def _validate_query_answer_string(value: str, field_name: str) -> None:
    try:
        _validate_string(value, field_name)
    except RAGAnythingConfigError:
        raise RAGAnythingRuntimeError(
            "raganything_query_result_invalid",
            "RAG-Anything query result could not be normalized to owned data.",
        ) from None


def _validated_lightrag_options(value: Mapping[Any, Any]) -> dict[str, JsonValue]:
    validated = _validated_json_mapping(value, "config_options")
    canonical: dict[str, JsonValue] = {}
    for key in validated:
        normalized_key = _normalized_key(key)
        if normalized_key not in _ALLOWED_LIGHTRAG_OPTION_KEYS:
            _raise_config(f"RAG-Anything config option is not allowed: {key}.")
        _validate_lightrag_option_value(normalized_key, validated[key], key)
        if normalized_key in canonical:
            _raise_config(f"RAG-Anything config option is duplicated: {key}.")
        canonical[normalized_key] = validated[key]
    return canonical


def _validate_lightrag_option_value(
    normalized_key: str, value: JsonValue, original_key: str
) -> None:
    if normalized_key in _LIGHTRAG_POSITIVE_INT_OPTION_KEYS:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            _raise_config(f"RAG-Anything config option must be a positive integer: {original_key}.")
        return
    if normalized_key in _LIGHTRAG_NON_NEGATIVE_INT_OPTION_KEYS:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            _raise_config(
                f"RAG-Anything config option must be a non-negative integer: {original_key}."
            )
        return
    if normalized_key in _LIGHTRAG_STRING_OPTION_KEYS:
        if not isinstance(value, str) or not value.strip():
            _raise_config(f"RAG-Anything config option must be a non-empty string: {original_key}.")
        _validate_string(value, f"config_options.{original_key}")
        return
    _raise_config(f"RAG-Anything config option is not allowed: {original_key}.")


def _validated_json_mapping(
    value: Mapping[Any, Any],
    field_name: str,
    *,
    enforce_config_key_policy: bool = True,
    _active_ids: set[int] | None = None,
) -> dict[str, JsonValue]:
    if not isinstance(value, Mapping):
        _raise_config(f"RAG-Anything {field_name} must be a JSON object.")
    active_ids = set() if _active_ids is None else _active_ids
    value_id = id(value)
    if value_id in active_ids:
        _raise_config(f"RAG-Anything {field_name} must be portable JSON.")
    active_ids.add(value_id)
    validated: dict[str, JsonValue] = {}
    try:
        for key, item in value.items():
            if not isinstance(key, str):
                _raise_config(f"RAG-Anything {field_name} keys must be strings.")
            _validate_string(key, f"{field_name} key")
            normalized_key: str | None = None
            if enforce_config_key_policy:
                normalized_key = _normalized_key(key)
                if normalized_key in _MANAGED_CONFIG_KEYS:
                    _raise_config(
                        f"RAG-Anything config option cannot override managed field: {key}."
                    )
                if _is_secret_config_key(normalized_key):
                    _raise_config("RAG-Anything config options must not include secrets.")
            is_header_container = (
                enforce_config_key_policy
                and normalized_key is not None
                and _is_header_config_key(normalized_key)
            )
            validated_item = _validated_json_value(
                item,
                f"{field_name}.{key}",
                enforce_config_key_policy=enforce_config_key_policy and not is_header_container,
                _active_ids=active_ids,
            )
            if is_header_container and _contains_secret_header_value(validated_item):
                _raise_config("RAG-Anything config options must not include secrets.")
            validated[key] = validated_item
        if enforce_config_key_policy and _contains_secret_header_descriptor(validated):
            _raise_config("RAG-Anything config options must not include secrets.")
        try:
            return json.loads(json.dumps(validated, ensure_ascii=False, allow_nan=False))
        except (TypeError, ValueError):
            raise RAGAnythingConfigError(
                "raganything_config_invalid",
                f"RAG-Anything {field_name} must be portable JSON.",
            ) from None
    finally:
        active_ids.remove(value_id)


def _validated_json_value(
    value: Any,
    field_name: str,
    *,
    enforce_config_key_policy: bool = True,
    _active_ids: set[int] | None = None,
) -> JsonValue:
    if isinstance(value, str):
        _validate_string(value, field_name)
        return value
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return value
    if isinstance(value, float) and isfinite(value):
        return value
    if isinstance(value, list):
        active_ids = set() if _active_ids is None else _active_ids
        value_id = id(value)
        if value_id in active_ids:
            _raise_config(f"RAG-Anything {field_name} must be portable JSON.")
        active_ids.add(value_id)
        try:
            return [
                _validated_json_value(
                    item,
                    f"{field_name}[]",
                    enforce_config_key_policy=enforce_config_key_policy,
                    _active_ids=active_ids,
                )
                for item in value
            ]
        finally:
            active_ids.remove(value_id)
    if isinstance(value, tuple):
        active_ids = set() if _active_ids is None else _active_ids
        value_id = id(value)
        if value_id in active_ids:
            _raise_config(f"RAG-Anything {field_name} must be portable JSON.")
        active_ids.add(value_id)
        try:
            return [
                _validated_json_value(
                    item,
                    f"{field_name}[]",
                    enforce_config_key_policy=enforce_config_key_policy,
                    _active_ids=active_ids,
                )
                for item in value
            ]
        finally:
            active_ids.remove(value_id)
    if isinstance(value, Mapping):
        return _validated_json_mapping(
            value,
            field_name,
            enforce_config_key_policy=enforce_config_key_policy,
            _active_ids=_active_ids,
        )
    _raise_config(f"RAG-Anything {field_name} must be portable JSON.")


def _validate_non_empty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        _raise_runtime(f"RAG-Anything {field_name} must be a non-empty string.")
    _validate_string(value, field_name)


def _validate_string(value: str, field_name: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise RAGAnythingConfigError(
            "raganything_config_invalid",
            f"RAG-Anything {field_name} must be valid UTF-8.",
        ) from None


def _normalized_key(value: str) -> str:
    separated = re.sub(r"[^0-9A-Za-z]+", "_", value)
    with_boundaries = _CAMEL_CASE_BOUNDARY.sub("_", separated)
    return "_".join(part for part in with_boundaries.lower().split("_") if part)


def _is_secret_config_key(normalized_key: str) -> bool:
    for candidate in _secret_key_candidates(normalized_key):
        if _SECRET_KEY_PATTERN.search(candidate):
            return True
        compact_key = candidate.replace("_", "")
        if compact_key in _NON_SECRET_COMPACT_KEYS:
            continue
        if compact_key in _SECRET_COMPACT_KEYS or compact_key in _SECRET_SUFFIXES:
            return True
        parts = [part for part in candidate.split("_") if part]
        if parts and parts[-1] in _SECRET_SUFFIXES:
            if len(parts) > 1:
                return True
        for suffix in _SECRET_SUFFIXES:
            if not compact_key.endswith(suffix):
                continue
            prefix = compact_key[: -len(suffix)]
            if prefix:
                return True
    return False


def _is_header_config_key(normalized_key: str) -> bool:
    return any(part in {"header", "headers"} for part in normalized_key.split("_"))


def _contains_secret_header_descriptor(value: JsonValue) -> bool:
    if isinstance(value, list):
        return any(_contains_secret_header_descriptor(item) for item in value)
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = _normalized_key(key)
            if normalized_key in {
                "header",
                "header_name",
                "header_names",
                "headers",
                "headers_name",
                "headers_names",
                "key",
                "name",
            } and isinstance(item, str):
                if _is_secret_header_name(item):
                    return True
            if _contains_secret_header_descriptor(item):
                return True
    return False


def _contains_secret_header_value(value: JsonValue) -> bool:
    if isinstance(value, str):
        return _looks_like_secret_header_string(value)
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and _is_secret_header_name(item):
                return True
        return any(_contains_secret_header_value(item) for item in value)
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = _normalized_key(key)
            if normalized_key in {"header", "header_name", "key", "name"} and isinstance(item, str):
                if _is_secret_header_name(item):
                    return True
            elif _is_secret_header_name(key):
                return True
            if _contains_secret_header_value(item):
                return True
    return False


def _looks_like_secret_header_string(value: str) -> bool:
    for line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        normalized_line = line.strip()
        if not normalized_line:
            continue
        header_name, separator, _header_value = normalized_line.partition(":")
        if separator and _is_secret_header_name(header_name):
            return True
        if normalized_line.lower().startswith(("bearer ", "basic ", "jwt ")):
            return True
    return False


def _is_secret_header_name(value: str) -> bool:
    normalized_key = _normalized_key(value.strip())
    if _is_secret_config_key(normalized_key):
        return True
    return any(part in _SECRET_HEADER_NAME_PARTS for part in normalized_key.split("_"))


def _secret_key_candidates(normalized_key: str) -> list[str]:
    candidates = [normalized_key]
    parts = [part for part in normalized_key.split("_") if part]
    while parts and parts[-1] in _SECRET_QUALIFIER_SUFFIXES:
        parts = parts[:-1]
        if parts:
            candidates.append("_".join(parts))
    return candidates


def _raise_config(message: str) -> None:
    raise RAGAnythingConfigError("raganything_config_invalid", message)


def _raise_runtime(message: str) -> None:
    raise RAGAnythingRuntimeError("raganything_adapter_invalid_input", message)
