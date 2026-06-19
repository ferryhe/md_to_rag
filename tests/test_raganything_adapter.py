from __future__ import annotations

import asyncio
import json
import traceback
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from md_to_rag import api, mcp
from md_to_rag.cli import app
from md_to_rag.schemas import CommandName, CommandStatus


runner = CliRunner()
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_optional_dependency_extra_targets_raganything_range() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())

    assert pyproject["project"]["optional-dependencies"]["raganything"] == [
        "raganything>=1.3.1,<2.0"
    ]


def test_adapter_reports_unavailable_dependency_without_import_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from md_to_rag import raganything_adapter

    def missing_import(name: str) -> Any:
        assert name == "raganything"
        raise ModuleNotFoundError("No module named 'raganything'")

    monkeypatch.setattr(raganything_adapter.importlib, "import_module", missing_import)

    with pytest.raises(raganything_adapter.RAGAnythingDependencyError) as exc_info:
        raganything_adapter.create_raganything_backend(
            raganything_adapter.RAGAnythingAdapterConfig(
                working_dir=tmp_path / "raganything-storage"
            )
        )

    error = exc_info.value
    assert error.code == "raganything_unavailable"
    assert "Install md-to-rag[raganything]" in error.message
    assert "Traceback" not in error.message
    assert error.__cause__ is None
    formatted_error = "".join(traceback.format_exception(error))
    assert "ModuleNotFoundError" not in formatted_error


def test_adapter_wraps_broken_dependency_loading_without_backend_messages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from md_to_rag import raganything_adapter

    def broken_import(name: str) -> Any:
        assert name == "raganything"
        raise RuntimeError("upstream import leaked secret")

    monkeypatch.setattr(raganything_adapter.importlib, "import_module", broken_import)

    with pytest.raises(raganything_adapter.RAGAnythingDependencyError) as import_exc:
        raganything_adapter.create_raganything_backend(
            raganything_adapter.RAGAnythingAdapterConfig(
                working_dir=tmp_path / "raganything-storage"
            )
        )

    assert import_exc.value.code == "raganything_unavailable"
    assert "secret" not in import_exc.value.message
    assert import_exc.value.__cause__ is None
    formatted_import_error = "".join(traceback.format_exception(import_exc.value))
    assert "RuntimeError" not in formatted_import_error
    assert "upstream import" not in formatted_import_error

    class BrokenModule:
        @property
        def RAGAnythingConfig(self) -> object:
            raise RuntimeError("lazy attribute leaked secret")

        class RAGAnything:
            pass

    with pytest.raises(raganything_adapter.RAGAnythingDependencyError) as attr_exc:
        raganything_adapter.create_raganything_backend(
            raganything_adapter.RAGAnythingAdapterConfig(
                working_dir=tmp_path / "raganything-storage"
            ),
            raganything_module=BrokenModule(),
        )

    assert attr_exc.value.code == "raganything_interface_invalid"
    assert "secret" not in attr_exc.value.message
    assert attr_exc.value.__cause__ is None
    formatted_attr_error = "".join(traceback.format_exception(attr_exc.value))
    assert "RuntimeError" not in formatted_attr_error
    assert "lazy attribute" not in formatted_attr_error


def test_adapter_config_validates_owned_internal_settings(tmp_path: Path) -> None:
    from md_to_rag import raganything_adapter

    config = raganything_adapter.RAGAnythingAdapterConfig(
        working_dir=tmp_path / "raganything-storage",
        parser="mineru",
        parse_method="auto",
        query_mode="hybrid",
        enable_image_processing=False,
        enable_table_processing=True,
        enable_equation_processing=True,
        display_stats=False,
        config_options={"max_entity_tokens": 256},
    )

    assert config.to_raganything_config_kwargs() == {
        "working_dir": str(tmp_path / "raganything-storage"),
        "parser": "mineru",
        "parse_method": "auto",
        "enable_image_processing": False,
        "enable_table_processing": True,
        "enable_equation_processing": True,
    }
    assert config.to_lightrag_kwargs() == {"max_entity_tokens": 256}

    invalid_configs = [
        {"working_dir": ""},
        {"working_dir": None},
        {"working_dir": b"rag"},
        {"working_dir": 123},
        {"working_dir": tmp_path / "rag", "parser": ""},
        {"working_dir": tmp_path / "rag", "parse_method": ""},
        {"working_dir": tmp_path / "rag", "query_mode": ""},
        {"working_dir": tmp_path / "rag", "display_stats": "yes"},
        {"working_dir": tmp_path / "rag", "config_options": {"llm_model_func": "bad"}},
        {"working_dir": tmp_path / "rag", "config_options": {"embedding_func": "bad"}},
        {"working_dir": tmp_path / "rag", "config_options": {"llmModelFunc": "bad"}},
        {"working_dir": tmp_path / "rag", "config_options": {"embeddingFunc": "bad"}},
        {"working_dir": tmp_path / "rag", "config_options": {"accessKeyId": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"aws_access_key_id": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"api_key": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"openaiApiKey": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"clientSecret": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"clientSecrets": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"passwords": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"authHeaderValue": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"authHeaders": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"auth": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"authentication": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"auth_method": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"cookies": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"githubToken": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"github_token": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"githubPat": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"GitHubPAT": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"slackBotToken": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"AzureOpenAIKey": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"openaiKey": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"OpenAIKey": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"openaiToken": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"oauthToken": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"apiTokenValue": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"openaiApiToken": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"openaiTokenValue": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"authTokenFile": "secret.txt"}},
        {"working_dir": tmp_path / "rag", "config_options": {"openaiKeyFile": "secret.txt"}},
        {"working_dir": tmp_path / "rag", "config_options": {"cookies_file": "secret.txt"}},
        {"working_dir": tmp_path / "rag", "config_options": {"authHeadersFile": "secret.txt"}},
        {"working_dir": tmp_path / "rag", "config_options": {"githubPatFileName": "secret.txt"}},
        {"working_dir": tmp_path / "rag", "config_options": {"tokenHeaderValue": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"authTokenHeaderValue": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"idTokenHeaderValue": "secret"}},
        {
            "working_dir": tmp_path / "rag",
            "config_options": {"llm_model_kwargs": {"cookies": "session=secret"}},
        },
        {
            "working_dir": tmp_path / "rag",
            "config_options": {"llm_model_kwargs": {"authHeaders": "Bearer secret"}},
        },
        {
            "working_dir": tmp_path / "rag",
            "config_options": {"llm_model_kwargs": {"authentication": "Bearer secret"}},
        },
        {
            "working_dir": tmp_path / "rag",
            "config_options": {
                "llm_model_kwargs": {"headers": "Authorization: Bearer secret"}
            },
        },
        {
            "working_dir": tmp_path / "rag",
            "config_options": {
                "headers": "Accept: application/json\nAuthorization: Bearer secret"
            },
        },
        {
            "working_dir": tmp_path / "rag",
            "config_options": {"headers": [["X-Api-Key", "secret"]]},
        },
        {
            "working_dir": tmp_path / "rag",
            "config_options": {"headers": [["X-Csrf-Token", "secret"]]},
        },
        {
            "working_dir": tmp_path / "rag",
            "config_options": {"headers": {"Authorization": "Bearer secret"}},
        },
        {
            "working_dir": tmp_path / "rag",
            "config_options": {"headers": {"X-Amz-Security-Token": "secret"}},
        },
        {
            "working_dir": tmp_path / "rag",
            "config_options": {
                "llm_model_kwargs": {"headers": [["Authorization", "Bearer secret"]]}
            },
        },
        {
            "working_dir": tmp_path / "rag",
            "config_options": {
                "llm_model_kwargs": {
                    "headers": [{"name": "Authorization", "value": "Bearer secret"}]
                }
            },
        },
        {
            "working_dir": tmp_path / "rag",
            "config_options": {"headers": [{"Name": "X-Api-Key", "Value": "secret"}]},
        },
        {
            "working_dir": tmp_path / "rag",
            "config_options": {"headers": [{"key": "X-Api-Key", "value": "secret"}]},
        },
        {
            "working_dir": tmp_path / "rag",
            "config_options": {
                "headers": [{"headerName": "X-Api-Key", "headerValue": "secret"}]
            },
        },
        {"working_dir": tmp_path / "rag", "config_options": {"apiToken": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"authToken": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"idToken": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"sessionToken": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"max_api_token": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"max_auth_token": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"summary_session_tokens": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"securityToken": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"csrfToken": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"amzSecurityToken": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"jwt": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"token": "secret"}},
        {"working_dir": tmp_path / "rag", "config_options": {"bad": object()}},
    ]
    for kwargs in invalid_configs:
        with pytest.raises(raganything_adapter.RAGAnythingConfigError):
            raganything_adapter.RAGAnythingAdapterConfig(**kwargs)

    portable_token_counts = raganything_adapter.RAGAnythingAdapterConfig(
        working_dir=tmp_path / "rag",
        config_options={
            "headers": {"Accept": "application/json"},
            "max_tokens": 1024,
            "max_entity_tokens": 256,
            "max_relation_tokens": 128,
            "max_total_tokens": 2048,
            "summary_max_tokens": 512,
            "tokenizer": "portable",
        },
    )
    assert portable_token_counts.to_lightrag_kwargs()["max_tokens"] == 1024
    assert portable_token_counts.to_lightrag_kwargs()["max_entity_tokens"] == 256
    assert portable_token_counts.to_lightrag_kwargs()["max_relation_tokens"] == 128
    assert portable_token_counts.to_lightrag_kwargs()["max_total_tokens"] == 2048
    assert portable_token_counts.to_lightrag_kwargs()["summary_max_tokens"] == 512

    portable_headers = raganything_adapter.RAGAnythingAdapterConfig(
        working_dir=tmp_path / "rag",
        config_options={
            "headers": [
                {"key": "Accept", "value": "application/json"},
                {"headerName": "X-Trace-Id", "headerValue": "trace-1"},
            ],
        },
    )
    assert portable_headers.to_lightrag_kwargs()["headers"] == [
        {"key": "Accept", "value": "application/json"},
        {"headerName": "X-Trace-Id", "headerValue": "trace-1"},
    ]

    portable_header_block = raganything_adapter.RAGAnythingAdapterConfig(
        working_dir=tmp_path / "rag",
        config_options={"headers": "Accept: application/json\nX-Trace-Id: trace-1"},
    )
    assert (
        portable_header_block.to_lightrag_kwargs()["headers"]
        == "Accept: application/json\nX-Trace-Id: trace-1"
    )


def test_adapter_config_options_are_frozen_and_revalidated(tmp_path: Path) -> None:
    from md_to_rag import raganything_adapter

    config = raganything_adapter.RAGAnythingAdapterConfig(
        working_dir=tmp_path / "raganything-storage",
        config_options={"nested": {"max_entity_tokens": 256}},
    )

    with pytest.raises(TypeError):
        config.config_options["api_key"] = "secret"  # type: ignore[index]

    nested = config.config_options["nested"]
    assert isinstance(nested, dict)
    nested["apiToken"] = "secret"

    with pytest.raises(raganything_adapter.RAGAnythingConfigError):
        config.to_lightrag_kwargs()


def test_adapter_uses_raganything_touchpoints_and_normalizes_results(tmp_path: Path) -> None:
    from md_to_rag import raganything_adapter

    class FakeRAGAnythingConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeRAGAnything:
        last_instance: "FakeRAGAnything | None" = None

        def __init__(
            self,
            *,
            config: FakeRAGAnythingConfig,
            llm_model_func: Any | None = None,
            embedding_func: Any | None = None,
        ) -> None:
            self.config = config
            self.llm_model_func = llm_model_func
            self.embedding_func = embedding_func
            self.insert_calls: list[dict[str, Any]] = []
            self.query_calls: list[dict[str, Any]] = []
            FakeRAGAnything.last_instance = self

        async def insert_content_list(self, **kwargs: Any) -> object:
            self.insert_calls.append(kwargs)
            return object()

        async def aquery(self, question: str, *, mode: str) -> dict[str, Any]:
            self.query_calls.append({"question": question, "mode": mode})
            return {"answer": "Alpha answer", "backend_object": object()}

    backend = raganything_adapter.create_raganything_backend(
        raganything_adapter.RAGAnythingAdapterConfig(
            working_dir=tmp_path / "raganything-storage",
            display_stats=False,
        ),
        raganything_module=SimpleNamespace(
            RAGAnything=FakeRAGAnything,
            RAGAnythingConfig=FakeRAGAnythingConfig,
        ),
        llm_model_func=lambda *_args, **_kwargs: "llm",
        embedding_func=lambda *_args, **_kwargs: [0.1],
    )
    content_list = [
        {
            "type": "text",
            "text": "Alpha content",
            "page_idx": 0,
            "headers": "Authorization: Bearer content-token",
            "key": "block-alpha",
            "metadata": {"tokens": 2, "key": "section-alpha"},
        }
    ]

    insert_result = asyncio.run(
        backend.insert_content_list(
            content_list=content_list,
            file_path="source/alpha.md",
            doc_id="doc_alpha",
        )
    )
    query_result = asyncio.run(backend.aquery("What is alpha?"))

    fake_rag = FakeRAGAnything.last_instance
    assert fake_rag is not None
    assert fake_rag.config.kwargs["working_dir"] == str(tmp_path / "raganything-storage")
    assert fake_rag.insert_calls == [
        {
            "content_list": content_list,
            "file_path": "source/alpha.md",
            "doc_id": "doc_alpha",
            "display_stats": False,
        }
    ]
    assert fake_rag.query_calls == [{"question": "What is alpha?", "mode": "hybrid"}]
    assert insert_result.doc_id == "doc_alpha"
    assert insert_result.file_path == "source/alpha.md"
    assert insert_result.inserted is True
    assert query_result.answer == "Alpha answer"
    assert query_result.mode == "hybrid"
    assert "backend_object" not in query_result.model_dump()
    assert "raganything" not in query_result.model_dump_json().lower()


def test_adapter_routes_config_options_to_lightrag_kwargs(tmp_path: Path) -> None:
    from md_to_rag import raganything_adapter

    class FakeRAGAnythingConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeRAGAnything:
        last_instance: "FakeRAGAnything | None" = None

        def __init__(
            self,
            *,
            config: FakeRAGAnythingConfig,
            lightrag_kwargs: dict[str, Any] | None = None,
            llm_model_func: Any | None = None,
            embedding_func: Any | None = None,
        ) -> None:
            self.config = config
            self.lightrag_kwargs = lightrag_kwargs
            self.llm_model_func = llm_model_func
            self.embedding_func = embedding_func
            FakeRAGAnything.last_instance = self

        async def insert_content_list(self, **_kwargs: Any) -> object:
            return object()

        async def aquery(self, _question: str, *, mode: str) -> str:
            return mode

    raganything_adapter.create_raganything_backend(
        raganything_adapter.RAGAnythingAdapterConfig(
            working_dir=tmp_path / "raganything-storage",
            config_options={"max_entity_tokens": 256, "max_tokens": 1024},
        ),
        raganything_module=SimpleNamespace(
            RAGAnything=FakeRAGAnything,
            RAGAnythingConfig=FakeRAGAnythingConfig,
        ),
        llm_model_func=lambda *_args, **_kwargs: "llm",
        embedding_func=lambda *_args, **_kwargs: [0.1],
    )

    fake_rag = FakeRAGAnything.last_instance
    assert fake_rag is not None
    assert "max_entity_tokens" not in fake_rag.config.kwargs
    assert "max_tokens" not in fake_rag.config.kwargs
    assert fake_rag.lightrag_kwargs == {
        "max_entity_tokens": 256,
        "max_tokens": 1024,
    }


def test_adapter_rejects_non_callable_model_function_inputs(tmp_path: Path) -> None:
    from md_to_rag import raganything_adapter

    class FakeRAGAnythingConfig:
        def __init__(self, **_kwargs: Any) -> None:
            pass

    class FakeRAGAnything:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def insert_content_list(self, **_kwargs: Any) -> object:
            return object()

        async def aquery(self, _question: str, *, mode: str) -> str:
            return mode

    fake_module = SimpleNamespace(
        RAGAnything=FakeRAGAnything,
        RAGAnythingConfig=FakeRAGAnythingConfig,
    )
    config = raganything_adapter.RAGAnythingAdapterConfig(
        working_dir=tmp_path / "raganything-storage"
    )

    invalid_args = [
        {},
        {"llm_model_func": "token string"},
        {"embedding_func": "token string"},
    ]
    for kwargs in invalid_args:
        with pytest.raises(raganything_adapter.RAGAnythingConfigError) as exc_info:
            raganything_adapter.create_raganything_backend(
                config,
                raganything_module=fake_module,
                **kwargs,
            )

        assert exc_info.value.code == "raganything_config_invalid"
        assert "token string" not in exc_info.value.message


def test_adapter_initializes_raganything_before_query(tmp_path: Path) -> None:
    from md_to_rag import raganything_adapter

    class FakeRAGAnythingConfig:
        def __init__(self, **_kwargs: Any) -> None:
            pass

    class FakeRAGAnything:
        last_instance: "FakeRAGAnything | None" = None

        def __init__(
            self,
            *,
            config: FakeRAGAnythingConfig,
            llm_model_func: Any | None = None,
            embedding_func: Any | None = None,
        ) -> None:
            self.config = config
            self.llm_model_func = llm_model_func
            self.embedding_func = embedding_func
            self.initialized = False
            self.ensure_calls = 0
            FakeRAGAnything.last_instance = self

        async def _ensure_lightrag_initialized(self) -> dict[str, bool]:
            self.ensure_calls += 1
            self.initialized = True
            return {"success": True}

        async def insert_content_list(self, **_kwargs: Any) -> object:
            return object()

        async def aquery(self, _question: str, *, mode: str) -> str:
            if not self.initialized:
                raise RuntimeError("upstream LightRAG not initialized")
            return f"ready:{mode}"

    backend = raganything_adapter.create_raganything_backend(
        raganything_adapter.RAGAnythingAdapterConfig(
            working_dir=tmp_path / "raganything-storage"
        ),
        raganything_module=SimpleNamespace(
            RAGAnything=FakeRAGAnything,
            RAGAnythingConfig=FakeRAGAnythingConfig,
        ),
        llm_model_func=lambda *_args, **_kwargs: "llm",
        embedding_func=lambda *_args, **_kwargs: [0.1],
    )

    query_result = asyncio.run(backend.aquery("What exists?"))

    fake_rag = FakeRAGAnything.last_instance
    assert fake_rag is not None
    assert fake_rag.ensure_calls == 1
    assert query_result.answer == "ready:hybrid"


def test_adapter_wraps_json_round_trip_validation_failures(tmp_path: Path) -> None:
    from md_to_rag import raganything_adapter

    huge_integer = 10**5000
    recursive_config: dict[str, Any] = {}
    recursive_config["self"] = recursive_config

    with pytest.raises(raganything_adapter.RAGAnythingConfigError) as config_exc:
        raganything_adapter.RAGAnythingAdapterConfig(
            working_dir=tmp_path / "raganything-storage",
            config_options={"max_entity_tokens": huge_integer},
        )

    assert config_exc.value.code == "raganything_config_invalid"
    formatted_config_error = "".join(traceback.format_exception(config_exc.value))
    assert "ValueError" not in formatted_config_error
    assert "Exceeds the limit" not in formatted_config_error

    with pytest.raises(raganything_adapter.RAGAnythingConfigError) as recursive_config_exc:
        raganything_adapter.RAGAnythingAdapterConfig(
            working_dir=tmp_path / "raganything-storage",
            config_options=recursive_config,
        )

    assert recursive_config_exc.value.code == "raganything_config_invalid"
    formatted_recursive_config_error = "".join(
        traceback.format_exception(recursive_config_exc.value)
    )
    assert "RecursionError" not in formatted_recursive_config_error

    class FakeRAGAnything:
        async def insert_content_list(self, **_kwargs: Any) -> object:
            return object()

        async def aquery(self, _question: str, *, mode: str) -> str:
            return mode

    backend = raganything_adapter.RAGAnythingBackend(
        FakeRAGAnything(),
        raganything_adapter.RAGAnythingAdapterConfig(
            working_dir=tmp_path / "raganything-storage"
        ),
    )

    with pytest.raises(raganything_adapter.RAGAnythingConfigError) as content_exc:
        asyncio.run(
            backend.insert_content_list(
                content_list=[{"type": "text", "value": huge_integer}],
                file_path="source/alpha.md",
                doc_id="doc_alpha",
            )
        )

    assert content_exc.value.code == "raganything_config_invalid"
    formatted_content_error = "".join(traceback.format_exception(content_exc.value))
    assert "ValueError" not in formatted_content_error
    assert "Exceeds the limit" not in formatted_content_error

    recursive_content: dict[str, Any] = {"type": "text"}
    recursive_content["self"] = recursive_content
    with pytest.raises(raganything_adapter.RAGAnythingConfigError) as recursive_content_exc:
        asyncio.run(
            backend.insert_content_list(
                content_list=[recursive_content],
                file_path="source/alpha.md",
                doc_id="doc_alpha",
            )
        )

    assert recursive_content_exc.value.code == "raganything_config_invalid"
    formatted_recursive_content_error = "".join(
        traceback.format_exception(recursive_content_exc.value)
    )
    assert "RecursionError" not in formatted_recursive_content_error


def test_adapter_suppresses_invalid_utf8_validation_causes(tmp_path: Path) -> None:
    from md_to_rag import raganything_adapter

    invalid_text = "\ud800"

    with pytest.raises(raganything_adapter.RAGAnythingConfigError) as config_exc:
        raganything_adapter.RAGAnythingAdapterConfig(
            working_dir=tmp_path / "raganything-storage",
            config_options={"max_entity_tokens": invalid_text},
        )

    assert config_exc.value.code == "raganything_config_invalid"
    assert config_exc.value.__cause__ is None
    assert "UnicodeEncodeError" not in "".join(
        traceback.format_exception(config_exc.value)
    )

    class FakeRAGAnything:
        async def insert_content_list(self, **_kwargs: Any) -> object:
            return object()

        async def aquery(self, _question: str, *, mode: str) -> str:
            return mode

    backend = raganything_adapter.RAGAnythingBackend(
        FakeRAGAnything(),
        raganything_adapter.RAGAnythingAdapterConfig(
            working_dir=tmp_path / "raganything-storage"
        ),
    )

    with pytest.raises(raganything_adapter.RAGAnythingConfigError) as content_exc:
        asyncio.run(
            backend.insert_content_list(
                content_list=[{"type": "text", "value": invalid_text}],
                file_path="source/alpha.md",
                doc_id="doc_alpha",
            )
        )

    assert content_exc.value.code == "raganything_config_invalid"
    assert content_exc.value.__cause__ is None
    assert "UnicodeEncodeError" not in "".join(
        traceback.format_exception(content_exc.value)
    )

    with pytest.raises(raganything_adapter.RAGAnythingConfigError) as query_exc:
        asyncio.run(backend.aquery(invalid_text))

    assert query_exc.value.code == "raganything_config_invalid"
    assert query_exc.value.__cause__ is None
    assert "UnicodeEncodeError" not in "".join(
        traceback.format_exception(query_exc.value)
    )


def test_adapter_wraps_query_result_serialization_failures(tmp_path: Path) -> None:
    from md_to_rag import raganything_adapter

    huge_integer = 10**5000

    class BadQueryRAGAnything:
        async def insert_content_list(self, **_kwargs: Any) -> object:
            return object()

        async def aquery(self, _question: str, *, mode: str) -> dict[str, Any]:
            return {"value": huge_integer, "mode": mode}

    backend = raganything_adapter.RAGAnythingBackend(
        BadQueryRAGAnything(),
        raganything_adapter.RAGAnythingAdapterConfig(
            working_dir=tmp_path / "raganything-storage"
        ),
    )

    with pytest.raises(raganything_adapter.RAGAnythingRuntimeError) as exc_info:
        asyncio.run(backend.aquery("What exists?"))

    assert exc_info.value.code == "raganything_query_result_invalid"
    assert exc_info.value.__cause__ is None
    formatted_error = "".join(traceback.format_exception(exc_info.value))
    assert "ValueError" not in formatted_error
    assert "Exceeds the limit" not in formatted_error


def test_adapter_suppresses_query_result_validation_causes(tmp_path: Path) -> None:
    from md_to_rag import raganything_adapter

    class BadQueryRAGAnything:
        async def insert_content_list(self, **_kwargs: Any) -> object:
            return object()

        async def aquery(self, _question: str, *, mode: str) -> dict[str, Any]:
            return {"backend_object": object()}

    backend = raganything_adapter.RAGAnythingBackend(
        BadQueryRAGAnything(),
        raganything_adapter.RAGAnythingAdapterConfig(
            working_dir=tmp_path / "raganything-storage"
        ),
    )

    with pytest.raises(raganything_adapter.RAGAnythingRuntimeError) as exc_info:
        asyncio.run(backend.aquery("What exists?"))

    assert exc_info.value.code == "raganything_query_result_invalid"
    assert exc_info.value.__cause__ is None
    formatted_error = "".join(traceback.format_exception(exc_info.value))
    assert "backend_object" not in formatted_error


def test_adapter_wraps_malformed_string_answers_as_query_result_errors(
    tmp_path: Path,
) -> None:
    from md_to_rag import raganything_adapter

    class BadStringAnswerRAGAnything:
        async def insert_content_list(self, **_kwargs: Any) -> object:
            return object()

        async def aquery(self, _question: str, *, mode: str) -> dict[str, str]:
            return {"answer": "\ud800"}

    backend = raganything_adapter.RAGAnythingBackend(
        BadStringAnswerRAGAnything(),
        raganything_adapter.RAGAnythingAdapterConfig(
            working_dir=tmp_path / "raganything-storage"
        ),
    )

    with pytest.raises(raganything_adapter.RAGAnythingRuntimeError) as exc_info:
        asyncio.run(backend.aquery("What exists?"))

    assert exc_info.value.code == "raganything_query_result_invalid"
    assert exc_info.value.__cause__ is None
    formatted_error = "".join(traceback.format_exception(exc_info.value))
    assert "UnicodeEncodeError" not in formatted_error


def test_adapter_wraps_upstream_config_constructor_errors(tmp_path: Path) -> None:
    from md_to_rag import raganything_adapter

    class RejectingRAGAnythingConfig:
        def __init__(self, **_kwargs: Any) -> None:
            raise TypeError("unexpected keyword argument 'max_entity_tokens'")

    class UnusedRAGAnything:
        pass

    with pytest.raises(raganything_adapter.RAGAnythingRuntimeError) as exc_info:
        raganything_adapter.create_raganything_backend(
            raganything_adapter.RAGAnythingAdapterConfig(
                working_dir=tmp_path / "raganything-storage",
                config_options={"max_entity_tokens": 256},
            ),
            raganything_module=SimpleNamespace(
                RAGAnything=UnusedRAGAnything,
                RAGAnythingConfig=RejectingRAGAnythingConfig,
            ),
            llm_model_func=lambda *_args, **_kwargs: "llm",
            embedding_func=lambda *_args, **_kwargs: [0.1],
        )

    assert exc_info.value.code == "raganything_initialization_failed"
    assert "unexpected keyword" not in exc_info.value.message


def test_adapter_wraps_upstream_system_exit_during_construction(
    tmp_path: Path,
) -> None:
    from md_to_rag import raganything_adapter

    class PlainRAGAnythingConfig:
        def __init__(self, **_kwargs: Any) -> None:
            pass

    class ExitingRAGAnything:
        def __init__(self, **_kwargs: Any) -> None:
            raise SystemExit("ENTITY_TYPES leaked backend exit")

    with pytest.raises(raganything_adapter.RAGAnythingRuntimeError) as exc_info:
        raganything_adapter.create_raganything_backend(
            raganything_adapter.RAGAnythingAdapterConfig(
                working_dir=tmp_path / "raganything-storage"
            ),
            raganything_module=SimpleNamespace(
                RAGAnything=ExitingRAGAnything,
                RAGAnythingConfig=PlainRAGAnythingConfig,
            ),
            llm_model_func=lambda *_args, **_kwargs: "llm",
            embedding_func=lambda *_args, **_kwargs: [0.1],
        )

    assert exc_info.value.code == "raganything_initialization_failed"
    assert exc_info.value.__cause__ is None
    formatted_error = "".join(traceback.format_exception(exc_info.value))
    assert "SystemExit" not in formatted_error
    assert "ENTITY_TYPES" not in formatted_error


def test_adapter_wraps_backend_method_lookup_failures(tmp_path: Path) -> None:
    from md_to_rag import raganything_adapter

    class BrokenRAGAnything:
        @property
        def insert_content_list(self) -> object:
            raise RuntimeError("upstream descriptor leaked secret")

        async def aquery(self, _question: str, *, mode: str) -> str:
            return mode

    config = raganything_adapter.RAGAnythingAdapterConfig(
        working_dir=tmp_path / "raganything-storage"
    )

    with pytest.raises(raganything_adapter.RAGAnythingDependencyError) as exc_info:
        raganything_adapter.RAGAnythingBackend(BrokenRAGAnything(), config)

    assert exc_info.value.code == "raganything_interface_invalid"
    assert exc_info.value.__cause__ is None
    formatted_error = "".join(traceback.format_exception(exc_info.value))
    assert "secret" not in formatted_error


def test_adapter_wraps_upstream_system_exit_during_query_initialization(
    tmp_path: Path,
) -> None:
    from md_to_rag import raganything_adapter

    class ExitingInitializerRAGAnything:
        async def insert_content_list(self, **_kwargs: Any) -> object:
            return object()

        def _ensure_lightrag_initialized(self) -> None:
            raise SystemExit("ENTITY_TYPES leaked backend exit")

        async def aquery(self, _question: str, *, mode: str) -> str:
            return mode

    backend = raganything_adapter.RAGAnythingBackend(
        ExitingInitializerRAGAnything(),
        raganything_adapter.RAGAnythingAdapterConfig(
            working_dir=tmp_path / "raganything-storage"
        ),
    )

    with pytest.raises(raganything_adapter.RAGAnythingRuntimeError) as exc_info:
        asyncio.run(backend.aquery("What exists?"))

    assert exc_info.value.code == "raganything_initialization_failed"
    assert exc_info.value.__cause__ is None
    formatted_error = "".join(traceback.format_exception(exc_info.value))
    assert "SystemExit" not in formatted_error
    assert "ENTITY_TYPES" not in formatted_error


def test_adapter_wraps_upstream_failures_without_exposing_backend_messages(
    tmp_path: Path,
) -> None:
    from md_to_rag import raganything_adapter

    class FailingRAGAnythingConfig:
        def __init__(self, **_kwargs: Any) -> None:
            raise ValueError("upstream config leaked secret")

    class PlainRAGAnythingConfig:
        def __init__(self, **_kwargs: Any) -> None:
            pass

    class FailingRAGAnything:
        def __init__(
            self,
            *,
            config: PlainRAGAnythingConfig,
            llm_model_func: Any | None = None,
            embedding_func: Any | None = None,
        ) -> None:
            self.config = config
            self.llm_model_func = llm_model_func
            self.embedding_func = embedding_func

        def insert_content_list(self, **_kwargs: Any) -> None:
            raise RuntimeError("upstream insert leaked secret")

        async def aquery(self, _question: str, *, mode: str) -> str:
            raise RuntimeError(f"upstream query leaked secret in {mode}")

    config = raganything_adapter.RAGAnythingAdapterConfig(
        working_dir=tmp_path / "raganything-storage"
    )

    with pytest.raises(raganything_adapter.RAGAnythingRuntimeError) as init_exc:
        raganything_adapter.create_raganything_backend(
            config,
            raganything_module=SimpleNamespace(
                RAGAnything=FailingRAGAnything,
                RAGAnythingConfig=FailingRAGAnythingConfig,
            ),
            llm_model_func=lambda *_args, **_kwargs: "llm",
            embedding_func=lambda *_args, **_kwargs: [0.1],
        )
    assert init_exc.value.code == "raganything_initialization_failed"
    assert "secret" not in init_exc.value.message
    assert init_exc.value.__cause__ is None
    assert "secret" not in "".join(traceback.format_exception(init_exc.value))

    backend = raganything_adapter.create_raganything_backend(
        config,
        raganything_module=SimpleNamespace(
            RAGAnything=FailingRAGAnything,
            RAGAnythingConfig=PlainRAGAnythingConfig,
        ),
        llm_model_func=lambda *_args, **_kwargs: "llm",
        embedding_func=lambda *_args, **_kwargs: [0.1],
    )
    with pytest.raises(raganything_adapter.RAGAnythingRuntimeError) as insert_exc:
        asyncio.run(
            backend.insert_content_list(
                content_list=[{"type": "text", "text": "Alpha"}],
                file_path="source/alpha.md",
                doc_id="doc_alpha",
            )
        )
    assert insert_exc.value.code == "raganything_insert_failed"
    assert "secret" not in insert_exc.value.message
    assert insert_exc.value.__cause__ is None
    assert "secret" not in "".join(traceback.format_exception(insert_exc.value))

    with pytest.raises(raganything_adapter.RAGAnythingRuntimeError) as query_exc:
        asyncio.run(backend.aquery("Alpha?"))
    assert query_exc.value.code == "raganything_query_failed"
    assert "secret" not in query_exc.value.message
    assert query_exc.value.__cause__ is None
    assert "secret" not in "".join(traceback.format_exception(query_exc.value))


def test_public_cli_api_mcp_surfaces_remain_backend_neutral_after_adapter_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from md_to_rag import raganything_adapter  # noqa: F401

    project = tmp_path / "project"
    api.init(project)
    (project / "source" / "doc.md").write_text("# Alpha\n\nAlpha body.\n", encoding="utf-8")
    assert api.ingest(source=project / "source").status is CommandStatus.OK
    assert api.chunk(manifest=project / "documents" / "documents.jsonl").status is CommandStatus.OK
    assert api.embed(chunks=project / "chunks" / "chunks.jsonl").status is CommandStatus.OK
    assert api.index(embeddings=project / "embeddings" / "embeddings.jsonl").status is CommandStatus.OK
    monkeypatch.chdir(project)

    api_payloads = [
        api.query("Alpha").model_dump_json(),
        api.inspect(project).model_dump_json(),
    ]
    cli_result = runner.invoke(app, ["query", "Alpha", "--json"], prog_name="md-to-rag")
    tool_payloads = [tool.model_dump_json() for tool in mcp.list_tools()]
    artifact_payloads = [
        (project / "corpus_manifest.json").read_text(encoding="utf-8"),
        (project / "indexes" / "index_manifest.json").read_text(encoding="utf-8"),
    ]

    assert cli_result.exit_code == 0
    assert json.loads(cli_result.output)["command"] == CommandName.QUERY.value
    serialized = "\n".join([*api_payloads, cli_result.output, *tool_payloads, *artifact_payloads])
    assert "raganything" not in serialized.lower()
    assert "RAGAnythingConfig" not in serialized
