from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from md_to_rag import api
from md_to_rag.cli import app
from md_to_rag.schemas import CommandStatus


runner = CliRunner()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prepare_chunked_project(project: Path) -> Path:
    api.init(project)
    (project / "source" / "alpha.md").write_text(
        "# Alpha\n\nFirst paragraph.\nSecond line.\n\nSecond paragraph.\n",
        encoding="utf-8",
    )
    (project / "source" / "beta.md").write_text(
        "## Beta\n\nOnly block.\n",
        encoding="utf-8",
    )
    ingest_response = api.ingest(source=project / "source")
    assert ingest_response.status is CommandStatus.OK
    chunk_response = api.chunk(manifest=project / "documents" / "documents.jsonl")
    assert chunk_response.status is CommandStatus.OK
    return project / "chunks" / "chunks.jsonl"


def test_embed_defaults_to_current_project_chunks_and_writes_portable_embeddings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    chunks_path = _prepare_chunked_project(project)
    chunk_rows = _jsonl(chunks_path)
    monkeypatch.chdir(project / "source")

    response = api.embed()

    assert response.__class__.__name__ == "EmbedResponse"
    assert response.status is CommandStatus.OK
    assert response.message == "Embedding artifacts generated."
    assert response.artifact_path == str((project / "embeddings" / "embeddings.jsonl").resolve())
    assert response.data.project_root == str(project.resolve())
    assert response.data.chunks_path == "chunks/chunks.jsonl"
    assert response.data.embeddings_path == "embeddings/embeddings.jsonl"
    assert response.data.changed is True
    assert response.data.chunk_count == len(chunk_rows)
    assert response.data.embedding_count == len(chunk_rows)
    assert response.data.profile == {
        "provider": "md_to_rag.local_hash",
        "model": "deterministic-hash-v1",
        "dimensions": 8,
        "version": "1.0",
    }

    embedding_rows = _jsonl(project / "embeddings" / "embeddings.jsonl")
    assert len(embedding_rows) == len(chunk_rows)
    first_embedding = embedding_rows[0]
    first_chunk = chunk_rows[0]
    assert first_embedding["schema_name"] == "md_to_rag.embedding"
    assert first_embedding["schema_version"] == "1.0"
    assert first_embedding["embedding_id"].startswith("emb_")
    assert first_embedding["chunk_id"] == first_chunk["chunk_id"]
    assert first_embedding["doc_id"] == first_chunk["doc_id"]
    assert first_embedding["source_id"] == first_chunk["source_id"]
    assert first_embedding["source_path"] == first_chunk["source_path"]
    assert first_embedding["chunk_content_hash"] == first_chunk["content_hash"]
    assert first_embedding["embedding_hash"].startswith("sha256:")
    assert first_embedding["profile"] == response.data.profile
    assert first_embedding["metadata"] == first_chunk["metadata"]
    assert first_embedding["provenance"]["chunk_id"] == first_chunk["chunk_id"]
    assert first_embedding["provenance"]["chunk_content_hash"] == first_chunk["content_hash"]
    assert first_embedding["provenance"]["chunks_path"] == "chunks/chunks.jsonl"
    assert len(first_embedding["embedding"]) == response.data.profile["dimensions"]
    assert all(isinstance(value, float) for value in first_embedding["embedding"])

    manifest = json.loads((project / "corpus_manifest.json").read_text(encoding="utf-8"))
    embed_status = next(
        status for status in manifest["command_status"] if status["command"] == "embed"
    )
    assert embed_status["status"] == "ok"
    assert embed_status["artifact_path"] == "embeddings/embeddings.jsonl"
    assert embed_status["data"]["chunk_count"] == len(chunk_rows)
    assert embed_status["data"]["embedding_count"] == len(chunk_rows)
    assert embed_status["data"]["chunks_path"] == "chunks/chunks.jsonl"
    assert embed_status["data"]["embeddings_path"] == "embeddings/embeddings.jsonl"
    assert embed_status["data"]["chunks_hash"] == response.data.chunks_hash
    assert embed_status["data"]["embeddings_hash"] == response.data.embeddings_hash
    assert embed_status["data"]["profile"] == response.data.profile
    assert "raganything" not in json.dumps(manifest).lower()


def test_embed_accepts_explicit_chunks_relative_to_cwd(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    _prepare_chunked_project(project)
    monkeypatch.chdir(tmp_path)

    response = api.embed(chunks=Path("project") / "chunks" / "chunks.jsonl")

    assert response.status is CommandStatus.OK
    assert response.data.project_root == str(project.resolve())
    assert response.data.chunks_path == "chunks/chunks.jsonl"
    assert (project / "embeddings" / "embeddings.jsonl").is_file()


def test_embed_rerun_reuses_unchanged_artifacts_and_manifest(tmp_path: Path) -> None:
    from md_to_rag import embed as embed_module

    class CountingProvider(embed_module.DeterministicHashEmbeddingProvider):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def embed(self, content: str, *, chunk_id: str, content_hash: str) -> list[float]:
            self.calls += 1
            return super().embed(content, chunk_id=chunk_id, content_hash=content_hash)

    project = tmp_path / "project"
    chunks_path = _prepare_chunked_project(project)
    provider = CountingProvider()

    first = embed_module.embed_project(chunks_path, provider=provider)
    embeddings_path = project / "embeddings" / "embeddings.jsonl"
    first_rows = _jsonl(embeddings_path)
    first_bytes = embeddings_path.read_bytes()
    first_mtime = embeddings_path.stat().st_mtime_ns
    first_manifest_bytes = (project / "corpus_manifest.json").read_bytes()
    first_call_count = provider.calls

    second = embed_module.embed_project(chunks_path, provider=provider)

    assert first.status is CommandStatus.OK
    assert second.status is CommandStatus.OK
    assert second.message == "Embedding artifacts unchanged."
    assert second.data.changed is False
    assert provider.calls == first_call_count
    assert _jsonl(embeddings_path) == first_rows
    assert embeddings_path.read_bytes() == first_bytes
    assert embeddings_path.stat().st_mtime_ns == first_mtime
    assert (project / "corpus_manifest.json").read_bytes() == first_manifest_bytes


def test_embed_changed_profile_recomputes_without_persisting_secrets(tmp_path: Path) -> None:
    from md_to_rag import embed as embed_module

    project = tmp_path / "project"
    chunks_path = _prepare_chunked_project(project)
    first = api.embed(chunks=chunks_path)
    first_rows = _jsonl(project / "embeddings" / "embeddings.jsonl")

    changed_profile = embed_module.DeterministicHashEmbeddingProvider(
        model="deterministic-hash-v2",
        dimensions=4,
        options={
            "batch_size": 2,
            "max_tokens": 512,
            "tokenizer": "portable",
            "key": "standalone-key-secret",
            "keys": ["standalone-keys-secret"],
            "keyValue": "standalone-key-value-secret",
            "bearer": "bearer-secret",
            "tokens": ["standalone-tokens-secret"],
            "tokensValue": ["standalone-tokens-value-secret"],
            "sessionTokens": ["session-token-plural-secret"],
            "session_tokens": ["session-token-snake-plural-secret"],
            "idTokens": ["id-token-plural-secret"],
            "api_key": "sk-secret",
            "api_key_value": "api-key-value-secret",
            "api_keys": ["plural-secret"],
            "api key": "spaced-secret",
            "apiKeys": ["camel-plural-secret"],
            "openaiKey": "openai-key-secret",
            "OpenAIKey": "open-ai-key-secret",
            "openaiApiKeyValue": "openai-api-key-value-secret",
            "token": "token-secret",
            "access_token_value": "access-token-value-secret",
            "accessTokens": ["access-token-plural-secret"],
            "private_key": "private-secret",
            "accessToken": "access-secret",
            "clientSecrets": ["client-secret-plural-secret"],
            "secretKey": "secret-key-value",
            "privateKey": "private-key-value",
            "passwords": ["password-plural-secret"],
            "subscription_key": "subscription-secret",
            "subscriptionKey": "camel-subscription-secret",
            "auth_header": "Bearer header-secret",
            "authHeader": "Bearer camel-header-secret",
            "authHeaders": "Bearer auth-headers-secret",
            "auth_headers": "Bearer snake-auth-headers-secret",
            "authHeaderValue": "Bearer auth-header-value-secret",
            "apiKeyHeaderValue": "api-key-header-value-secret",
            "accessTokenHeaderValue": "access-token-header-value-secret",
            "subscriptionKeyHeaderValue": "subscription-key-header-value-secret",
            "tokenHeaderValue": "token-header-value-secret",
            "auth.header": "Bearer dotted-header-secret",
            "authKey": "auth-key-secret",
            "auth_key": "auth-key-snake-secret",
            "authentication": "authentication-secret",
            "oauth_key": "oauth-key-secret",
            "appKey": "app-key-secret",
            "consumer_key": "consumer-key-secret",
            "access_key_id": "access-key-id-secret",
            "jwt": "jwt-secret",
            "headers": {
                "Bearer": "header-bearer-secret",
                "Cookie": "session=cookie-secret",
                "Content-Type": "application/json",
            },
            "cookie": "top-cookie-secret",
            "cookies": "top-cookies-secret",
            "api": {
                "base_url": "https://example.invalid/v1",
                "key": "nested-api-key-secret",
            },
            "nested": {
                "credentialsJson": "credentials-secret",
                "password": "pw",
                "keep": "value",
            },
        },
    )
    second = embed_module.embed_project(chunks_path, provider=changed_profile)
    second_rows = _jsonl(project / "embeddings" / "embeddings.jsonl")

    assert first.data.profile["dimensions"] == 8
    assert second.status is CommandStatus.OK
    assert second.data.changed is True
    assert second.data.profile == {
        "provider": "md_to_rag.local_hash",
        "model": "deterministic-hash-v2",
        "dimensions": 4,
        "version": "1.0",
        "options": {
            "api": {"base_url": "https://example.invalid/v1"},
            "batch_size": 2,
            "headers": {"Content-Type": "application/json"},
            "max_tokens": 512,
            "nested": {"keep": "value"},
            "tokenizer": "portable",
        },
    }
    assert first_rows != second_rows
    assert all(len(row["embedding"]) == 4 for row in second_rows)
    persisted_text = (
        (project / "embeddings" / "embeddings.jsonl").read_text(encoding="utf-8")
        + (project / "corpus_manifest.json").read_text(encoding="utf-8")
    )
    assert "standalone-key-secret" not in persisted_text
    assert "standalone-keys-secret" not in persisted_text
    assert "standalone-key-value-secret" not in persisted_text
    assert "bearer-secret" not in persisted_text
    assert "header-bearer-secret" not in persisted_text
    assert "cookie-secret" not in persisted_text
    assert "top-cookie-secret" not in persisted_text
    assert "top-cookies-secret" not in persisted_text
    assert "standalone-tokens-secret" not in persisted_text
    assert "standalone-tokens-value-secret" not in persisted_text
    assert "session-token-plural-secret" not in persisted_text
    assert "session-token-snake-plural-secret" not in persisted_text
    assert "id-token-plural-secret" not in persisted_text
    assert "sk-secret" not in persisted_text
    assert "api-key-value-secret" not in persisted_text
    assert "plural-secret" not in persisted_text
    assert "spaced-secret" not in persisted_text
    assert "camel-plural-secret" not in persisted_text
    assert "openai-key-secret" not in persisted_text
    assert "open-ai-key-secret" not in persisted_text
    assert "openai-api-key-value-secret" not in persisted_text
    assert "token-secret" not in persisted_text
    assert "access-token-value-secret" not in persisted_text
    assert "access-token-plural-secret" not in persisted_text
    assert "private-secret" not in persisted_text
    assert "access-secret" not in persisted_text
    assert "client-secret-plural-secret" not in persisted_text
    assert "secret-key-value" not in persisted_text
    assert "private-key-value" not in persisted_text
    assert "password-plural-secret" not in persisted_text
    assert "subscription-secret" not in persisted_text
    assert "camel-subscription-secret" not in persisted_text
    assert "header-secret" not in persisted_text
    assert "camel-header-secret" not in persisted_text
    assert "auth-headers-secret" not in persisted_text
    assert "snake-auth-headers-secret" not in persisted_text
    assert "auth-header-value-secret" not in persisted_text
    assert "api-key-header-value-secret" not in persisted_text
    assert "access-token-header-value-secret" not in persisted_text
    assert "subscription-key-header-value-secret" not in persisted_text
    assert "token-header-value-secret" not in persisted_text
    assert "dotted-header-secret" not in persisted_text
    assert "auth-key-secret" not in persisted_text
    assert "auth-key-snake-secret" not in persisted_text
    assert "authentication-secret" not in persisted_text
    assert "oauth-key-secret" not in persisted_text
    assert "app-key-secret" not in persisted_text
    assert "consumer-key-secret" not in persisted_text
    assert "access-key-id-secret" not in persisted_text
    assert "jwt-secret" not in persisted_text
    assert "nested-api-key-secret" not in persisted_text
    assert "credentials-secret" not in persisted_text
    assert "password" not in persisted_text
    assert "raganything" not in persisted_text.lower()


def test_embed_redacts_list_shaped_header_profile_secrets(tmp_path: Path) -> None:
    from md_to_rag import embed as embed_module

    class HeaderListProvider(embed_module.DeterministicHashEmbeddingProvider):
        def __init__(self) -> None:
            super().__init__(dimensions=1)

        def profile(self) -> dict[str, Any]:
            return {
                "provider": "header-list-provider",
                "model": "local",
                "dimensions": 1,
                "headerList": [
                    ["Authorization", "Bearer header-list-alias-secret"],
                ],
                "headersScalar": "Authorization: Bearer scalar-header-secret",
                "apiKeyHeaderScalar": "X-Api-Key: sk_live_header_scalar_123",
                "subscriptionHeaderScalar": (
                    "Ocp-Apim-Subscription-Key: sub_header_scalar_123"
                ),
                "customHeaderScalar": "JWT: eyJhbGciOiJIUzI1NiJ9.scalar",
                "headers": [
                    ["Authorization", "Bearer header-list-token-secret"],
                    ["Proxy-Authorization", "Bearer header-list-proxy-secret", True],
                    ["Cookie", "session=header-list-cookie-secret"],
                    ["X-Api-Key", "header-list-api-key-secret"],
                    ["Content-Type", "application/json"],
                ],
            }

        def embed(self, content: str, *, chunk_id: str, content_hash: str) -> list[float]:
            return [1.0]

    project = tmp_path / "project"
    chunks_path = _prepare_chunked_project(project)

    response = embed_module.embed_project(chunks_path, provider=HeaderListProvider())

    assert response.status is CommandStatus.OK
    assert response.data.profile["headerList"] == [["Authorization", "<redacted>"]]
    assert response.data.profile["headersScalar"] == "<redacted>"
    assert response.data.profile["apiKeyHeaderScalar"] == "<redacted>"
    assert response.data.profile["subscriptionHeaderScalar"] == "<redacted>"
    assert response.data.profile["customHeaderScalar"] == "<redacted>"
    assert response.data.profile["headers"] == [
        ["Authorization", "<redacted>"],
        ["Proxy-Authorization", "<redacted>", True],
        ["Cookie", "<redacted>"],
        ["X-Api-Key", "<redacted>"],
        ["Content-Type", "application/json"],
    ]
    persisted_text = (
        (project / "embeddings" / "embeddings.jsonl").read_text(encoding="utf-8")
        + (project / "corpus_manifest.json").read_text(encoding="utf-8")
    )
    assert "header-list-alias-secret" not in persisted_text
    assert "scalar-header-secret" not in persisted_text
    assert "sk_live_header_scalar_123" not in persisted_text
    assert "sub_header_scalar_123" not in persisted_text
    assert "eyJhbGciOiJIUzI1NiJ9.scalar" not in persisted_text
    assert "header-list-token-secret" not in persisted_text
    assert "header-list-proxy-secret" not in persisted_text
    assert "header-list-cookie-secret" not in persisted_text
    assert "header-list-api-key-secret" not in persisted_text
    assert "raganything" not in persisted_text.lower()


def test_embed_redacts_object_shaped_header_profile_secrets(tmp_path: Path) -> None:
    from md_to_rag import embed as embed_module

    class HeaderObjectProvider(embed_module.DeterministicHashEmbeddingProvider):
        def __init__(self) -> None:
            super().__init__(dimensions=1)

        def profile(self) -> dict[str, Any]:
            return {
                "provider": "header-object-provider",
                "model": "local",
                "dimensions": 1,
                "header": {
                    "name": "Authorization",
                    "value": "Bearer singular-object-header-secret",
                    "enabled": True,
                },
                "headerValueOnly": {
                    "value": "Bearer value-only-header-secret",
                },
                "headersByName": {
                    "primary": {
                        "name": "Authorization",
                        "value": "Bearer nested-map-header-secret",
                    },
                    "scalar": "Authorization: Bearer nested-scalar-header-secret",
                    "content": {"name": "Content-Type", "value": "application/json"},
                    "contentScalar": "application/json",
                },
                "headers": [
                    {
                        "name": "Authorization",
                        "value": "Bearer object-header-token-secret",
                        "enabled": True,
                    },
                    {
                        "key": "Cookie",
                        "value": "session=object-header-cookie-secret",
                    },
                    {
                        "name": "X-Api-Key",
                        "value": 123456,
                        "enabled": True,
                    },
                    {
                        "name": "Primary",
                        "key": "Authorization",
                        "value": "Bearer mixed-alias-header-secret",
                    },
                    {
                        "name": "Primary",
                        "value": "Bearer primary-value-header-secret",
                    },
                    {"name": "Content-Type", "value": "application/json"},
                ],
            }

        def embed(self, content: str, *, chunk_id: str, content_hash: str) -> list[float]:
            return [1.0]

    project = tmp_path / "project"
    chunks_path = _prepare_chunked_project(project)

    response = embed_module.embed_project(chunks_path, provider=HeaderObjectProvider())

    assert response.status is CommandStatus.OK
    assert response.data.profile["header"] == {
        "name": "Authorization",
        "value": "<redacted>",
        "enabled": True,
    }
    assert response.data.profile["headerValueOnly"] == {"value": "<redacted>"}
    assert response.data.profile["headersByName"] == {
        "primary": {"name": "Authorization", "value": "<redacted>"},
        "scalar": "<redacted>",
        "content": {"name": "Content-Type", "value": "application/json"},
        "contentScalar": "application/json",
    }
    assert response.data.profile["headers"] == [
        {"name": "Authorization", "value": "<redacted>", "enabled": True},
        {"key": "Cookie", "value": "<redacted>"},
        {"name": "X-Api-Key", "value": "<redacted>", "enabled": True},
        {"name": "Primary", "key": "Authorization", "value": "<redacted>"},
        {"name": "Primary", "value": "<redacted>"},
        {"name": "Content-Type", "value": "application/json"},
    ]
    persisted_text = (
        (project / "embeddings" / "embeddings.jsonl").read_text(encoding="utf-8")
        + (project / "corpus_manifest.json").read_text(encoding="utf-8")
    )
    assert "singular-object-header-secret" not in persisted_text
    assert "value-only-header-secret" not in persisted_text
    assert "nested-map-header-secret" not in persisted_text
    assert "nested-scalar-header-secret" not in persisted_text
    assert "object-header-token-secret" not in persisted_text
    assert "object-header-cookie-secret" not in persisted_text
    assert "mixed-alias-header-secret" not in persisted_text
    assert "primary-value-header-secret" not in persisted_text
    assert "123456" not in persisted_text
    assert "raganything" not in persisted_text.lower()


def test_embed_default_provider_seed_ignores_redacted_profile_secrets(tmp_path: Path) -> None:
    from md_to_rag import embed as embed_module

    class SecretProfileProvider(embed_module.DeterministicHashEmbeddingProvider):
        def __init__(self, secret: str) -> None:
            super().__init__(dimensions=2)
            self.secret = secret

        def profile(self) -> dict[str, Any]:
            profile = super().profile()
            profile["api_key"] = self.secret
            return profile

    left_project = tmp_path / "left"
    right_project = tmp_path / "right"
    left_chunks = _prepare_chunked_project(left_project)
    right_chunks = _prepare_chunked_project(right_project)

    left = embed_module.embed_project(left_chunks, provider=SecretProfileProvider("left-secret"))
    right = embed_module.embed_project(right_chunks, provider=SecretProfileProvider("right-secret"))

    assert left.status is CommandStatus.OK
    assert right.status is CommandStatus.OK
    assert left.data.profile == right.data.profile
    assert _jsonl(left_project / "embeddings" / "embeddings.jsonl") == _jsonl(
        right_project / "embeddings" / "embeddings.jsonl"
    )


def test_embed_preserves_explicit_falsey_provider(tmp_path: Path) -> None:
    from md_to_rag import embed as embed_module

    class FalseyProvider(embed_module.DeterministicHashEmbeddingProvider):
        def __bool__(self) -> bool:
            return False

        def profile(self) -> dict[str, Any]:
            return {
                "provider": "falsey-provider",
                "model": "local",
                "dimensions": 1,
            }

        def embed(self, content: str, *, chunk_id: str, content_hash: str) -> list[float]:
            return [42.0]

    project = tmp_path / "project"
    chunks_path = _prepare_chunked_project(project)

    response = embed_module.embed_project(chunks_path, provider=FalseyProvider())

    assert response.status is CommandStatus.OK
    assert response.data.profile["provider"] == "falsey-provider"
    assert {tuple(row["embedding"]) for row in _jsonl(project / "embeddings" / "embeddings.jsonl")} == {
        (42.0,)
    }


def test_embed_profile_cache_preserves_nonsecret_token_options(tmp_path: Path) -> None:
    from md_to_rag import embed as embed_module

    class TokenOptionProvider(embed_module.DeterministicHashEmbeddingProvider):
        def __init__(self, max_tokens: int) -> None:
            super().__init__(dimensions=1)
            self.max_tokens = max_tokens

        def profile(self) -> dict[str, Any]:
            return {
                "provider": "token-option",
                "model": "local",
                "dimensions": 1,
                "version": "1.0",
                "max_tokens": self.max_tokens,
                "token_limit": self.max_tokens,
                "input_token_limit": self.max_tokens,
                "tokenizer": "portable",
            }

        def embed(self, content: str, *, chunk_id: str, content_hash: str) -> list[float]:
            return [float(self.max_tokens)]

    project = tmp_path / "project"
    chunks_path = _prepare_chunked_project(project)

    first = embed_module.embed_project(chunks_path, provider=TokenOptionProvider(1))
    first_rows = _jsonl(project / "embeddings" / "embeddings.jsonl")
    second = embed_module.embed_project(chunks_path, provider=TokenOptionProvider(2))
    second_rows = _jsonl(project / "embeddings" / "embeddings.jsonl")

    assert first.status is CommandStatus.OK
    assert second.status is CommandStatus.OK
    assert second.data.changed is True
    assert first.data.profile["max_tokens"] == 1
    assert second.data.profile["max_tokens"] == 2
    assert second.data.profile["token_limit"] == 2
    assert second.data.profile["input_token_limit"] == 2
    assert second.data.profile["tokenizer"] == "portable"
    assert first_rows != second_rows
    assert {tuple(row["embedding"]) for row in second_rows} == {(2.0,)}


def test_embed_profile_cache_distinguishes_explicit_null_options(tmp_path: Path) -> None:
    from md_to_rag import embed as embed_module

    class NullOptionProvider(embed_module.DeterministicHashEmbeddingProvider):
        def __init__(self, *, include_revision: bool, value: float) -> None:
            super().__init__(dimensions=1)
            self.include_revision = include_revision
            self.value = value

        def profile(self) -> dict[str, Any]:
            profile: dict[str, Any] = {
                "provider": "null-option",
                "model": "local",
                "dimensions": 1,
                "version": "1.0",
            }
            if self.include_revision:
                profile["revision"] = None
            return profile

        def embed(self, content: str, *, chunk_id: str, content_hash: str) -> list[float]:
            return [self.value]

    project = tmp_path / "project"
    chunks_path = _prepare_chunked_project(project)

    first = embed_module.embed_project(
        chunks_path,
        provider=NullOptionProvider(include_revision=True, value=1.0),
    )
    first_rows = _jsonl(project / "embeddings" / "embeddings.jsonl")
    second = embed_module.embed_project(
        chunks_path,
        provider=NullOptionProvider(include_revision=False, value=2.0),
    )
    second_rows = _jsonl(project / "embeddings" / "embeddings.jsonl")

    assert first.status is CommandStatus.OK
    assert second.status is CommandStatus.OK
    assert first.data.profile["revision"] is None
    assert "revision" not in second.data.profile
    assert second.data.changed is True
    assert first_rows != second_rows
    assert {tuple(row["embedding"]) for row in second_rows} == {(2.0,)}


def test_embed_profile_cache_distinguishes_json_distinct_values(tmp_path: Path) -> None:
    from md_to_rag import embed as embed_module

    class JsonDistinctProvider(embed_module.DeterministicHashEmbeddingProvider):
        def __init__(self, option: Any, value: float) -> None:
            super().__init__(dimensions=1)
            self.option = option
            self.value = value

        def profile(self) -> dict[str, Any]:
            return {
                "provider": "json-distinct",
                "model": "local",
                "dimensions": 1,
                "version": "1.0",
                "option": self.option,
            }

        def embed(self, content: str, *, chunk_id: str, content_hash: str) -> list[float]:
            return [self.value]

    project = tmp_path / "project"
    chunks_path = _prepare_chunked_project(project)

    first = embed_module.embed_project(chunks_path, provider=JsonDistinctProvider(1, 1.0))
    first_rows = _jsonl(project / "embeddings" / "embeddings.jsonl")
    second = embed_module.embed_project(chunks_path, provider=JsonDistinctProvider(1.0, 2.0))
    second_rows = _jsonl(project / "embeddings" / "embeddings.jsonl")

    assert first.status is CommandStatus.OK
    assert second.status is CommandStatus.OK
    assert first.data.profile["option"] == 1
    assert second.data.profile["option"] == 1.0
    assert second.data.changed is True
    assert first_rows != second_rows
    assert {tuple(row["embedding"]) for row in second_rows} == {(2.0,)}


def test_embed_profile_cache_distinguishes_explicit_empty_options(tmp_path: Path) -> None:
    from md_to_rag import embed as embed_module

    class EmptyOptionsProvider(embed_module.DeterministicHashEmbeddingProvider):
        def __init__(self, *, include_options: bool, value: float) -> None:
            super().__init__(dimensions=1)
            self.include_options = include_options
            self.value = value

        def profile(self) -> dict[str, Any]:
            profile: dict[str, Any] = {
                "provider": "empty-options",
                "model": "local",
                "dimensions": 1,
                "version": "1.0",
            }
            if self.include_options:
                profile["options"] = {}
            return profile

        def embed(self, content: str, *, chunk_id: str, content_hash: str) -> list[float]:
            return [self.value]

    project = tmp_path / "project"
    chunks_path = _prepare_chunked_project(project)

    first = embed_module.embed_project(
        chunks_path,
        provider=EmptyOptionsProvider(include_options=True, value=1.0),
    )
    first_rows = _jsonl(project / "embeddings" / "embeddings.jsonl")
    second = embed_module.embed_project(
        chunks_path,
        provider=EmptyOptionsProvider(include_options=False, value=2.0),
    )
    second_rows = _jsonl(project / "embeddings" / "embeddings.jsonl")

    assert first.status is CommandStatus.OK
    assert second.status is CommandStatus.OK
    assert first.data.profile["options"] == {}
    assert "options" not in second.data.profile
    assert second.data.changed is True
    assert first_rows != second_rows
    assert {tuple(row["embedding"]) for row in second_rows} == {(2.0,)}


def test_embed_profile_cache_preserves_nested_max_tokens(tmp_path: Path) -> None:
    from md_to_rag import embed as embed_module

    class NestedMaxTokensProvider(embed_module.DeterministicHashEmbeddingProvider):
        def __init__(self, *, max_tokens: int, value: float) -> None:
            super().__init__(dimensions=1)
            self.max_tokens = max_tokens
            self.value = value

        def profile(self) -> dict[str, Any]:
            return {
                "provider": "nested-max-tokens",
                "model": "local",
                "dimensions": 1,
                "version": "1.0",
                "api": {"max_tokens": self.max_tokens},
            }

        def embed(self, content: str, *, chunk_id: str, content_hash: str) -> list[float]:
            return [self.value]

    project = tmp_path / "project"
    chunks_path = _prepare_chunked_project(project)

    first = embed_module.embed_project(
        chunks_path,
        provider=NestedMaxTokensProvider(max_tokens=1024, value=1.0),
    )
    first_rows = _jsonl(project / "embeddings" / "embeddings.jsonl")
    second = embed_module.embed_project(
        chunks_path,
        provider=NestedMaxTokensProvider(max_tokens=2048, value=2.0),
    )
    second_rows = _jsonl(project / "embeddings" / "embeddings.jsonl")

    assert first.status is CommandStatus.OK
    assert second.status is CommandStatus.OK
    assert first.data.profile["api"]["max_tokens"] == 1024
    assert second.data.profile["api"]["max_tokens"] == 2048
    assert second.data.changed is True
    assert first_rows != second_rows
    assert {tuple(row["embedding"]) for row in second_rows} == {(2.0,)}


def test_embed_reports_invalid_provider_profile_as_typed_error(tmp_path: Path) -> None:
    from md_to_rag import embed as embed_module

    class BadProfileProvider(embed_module.DeterministicHashEmbeddingProvider):
        def profile(self) -> dict[str, Any]:
            return {
                "provider": "bad-profile",
                "model": "bad",
                "dimensions": "wide",
            }

    class MissingDimensionsProvider(embed_module.DeterministicHashEmbeddingProvider):
        def profile(self) -> dict[str, Any]:
            return {
                "provider": "missing-dimensions",
                "model": "bad",
                "version": "1.0",
            }

    class NonJsonProfileProvider(embed_module.DeterministicHashEmbeddingProvider):
        def profile(self) -> dict[str, Any]:
            return {
                "provider": "non-json-profile",
                "model": "bad",
                "dimensions": 1,
                "revision": float("nan"),
            }

    class NonStringKeyProfileProvider(embed_module.DeterministicHashEmbeddingProvider):
        def profile(self) -> dict[str, Any]:
            return {
                "provider": "non-string-key-profile",
                "model": "bad",
                "dimensions": 1,
                1: "not portable",
            }

    class NonUtf8ProfileValueProvider(embed_module.DeterministicHashEmbeddingProvider):
        def __init__(self) -> None:
            super().__init__(dimensions=1)

        def profile(self) -> dict[str, Any]:
            return {
                "provider": "non-utf8-profile-value",
                "model": "\ud800",
                "dimensions": 1,
            }

    class NonUtf8ProfileKeyProvider(embed_module.DeterministicHashEmbeddingProvider):
        def __init__(self) -> None:
            super().__init__(dimensions=1)

        def profile(self) -> dict[str, Any]:
            return {
                "provider": "non-utf8-profile-key",
                "model": "bad",
                "dimensions": 1,
                "\ud800": "not portable",
            }

    class RecursiveProfileProvider(embed_module.DeterministicHashEmbeddingProvider):
        def profile(self) -> dict[str, Any]:
            profile: dict[str, Any] = {
                "provider": "recursive-profile",
                "model": "bad",
                "dimensions": 1,
            }
            profile["self"] = profile
            return profile

    for case_name, provider in (
        ("bad-dimensions", BadProfileProvider()),
        ("missing-dimensions", MissingDimensionsProvider()),
        ("non-json-profile", NonJsonProfileProvider()),
        ("non-string-key-profile", NonStringKeyProfileProvider()),
        ("non-utf8-profile-value", NonUtf8ProfileValueProvider()),
        ("non-utf8-profile-key", NonUtf8ProfileKeyProvider()),
        ("recursive-profile", RecursiveProfileProvider()),
        (
            "builtin-non-json-option",
            embed_module.DeterministicHashEmbeddingProvider(
                options={"bad": float("nan")}
            ),
        ),
    ):
        project = tmp_path / case_name
        chunks_path = _prepare_chunked_project(project)

        response = embed_module.embed_project(chunks_path, provider=provider)

        assert response.status is CommandStatus.ERROR
        assert response.error is not None
        assert response.error.code == "embedding_profile_invalid"
        assert not (project / "embeddings" / "embeddings.jsonl").exists()


def test_embed_rejects_output_artifact_inside_nested_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    chunks_path = _prepare_chunked_project(project)
    nested_project = project / "embeddings"
    api.init(nested_project)
    nested_manifest_bytes = (nested_project / "corpus_manifest.json").read_bytes()

    response = api.embed(chunks=chunks_path)

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "artifact_path_nested_project"
    assert not (project / "embeddings" / "embeddings.jsonl").exists()
    assert (nested_project / "corpus_manifest.json").read_bytes() == nested_manifest_bytes


def test_embed_wraps_provider_callback_failures_as_typed_errors(tmp_path: Path) -> None:
    from md_to_rag import embed as embed_module

    class BadProfileProvider(embed_module.DeterministicHashEmbeddingProvider):
        def profile(self) -> dict[str, Any]:
            raise RuntimeError(
                "profile exploded api_key=profile-secret Authorization: Bearer profile-token"
            )

    class BadEmbedProvider(embed_module.DeterministicHashEmbeddingProvider):
        def embed(self, content: str, *, chunk_id: str, content_hash: str) -> list[float]:
            raise RuntimeError(
                "embed exploded api_key=embed-secret Authorization: Bearer embed-token"
            )

    class BadProfileInputErrorProvider(embed_module.DeterministicHashEmbeddingProvider):
        def profile(self) -> dict[str, Any]:
            raise embed_module.EmbedInputError(
                "embedding_provider_failed",
                "wrapped api_key=profile-input-secret Authorization: Bearer profile-input-token",
            )

    class BadEmbedInputErrorProvider(embed_module.DeterministicHashEmbeddingProvider):
        def embed(self, content: str, *, chunk_id: str, content_hash: str) -> list[float]:
            raise embed_module.EmbedInputError(
                "embedding_provider_failed",
                "wrapped api_key=embed-input-secret Authorization: Bearer embed-input-token",
            )

    for case_name, provider in (
        ("profile", BadProfileProvider()),
        ("embed", BadEmbedProvider()),
        ("profile-input-error", BadProfileInputErrorProvider()),
        ("embed-input-error", BadEmbedInputErrorProvider()),
    ):
        project = tmp_path / case_name
        chunks_path = _prepare_chunked_project(project)

        response = embed_module.embed_project(chunks_path, provider=provider)

        assert response.status is CommandStatus.ERROR
        assert response.error is not None
        assert response.error.code == "embedding_provider_failed"
        response_text = f"{response.message} {response.error.message}"
        assert "traceback" not in response_text.lower()
        assert "profile-secret" not in response_text
        assert "profile-token" not in response_text
        assert "embed-secret" not in response_text
        assert "embed-token" not in response_text
        assert "profile-input-secret" not in response_text
        assert "profile-input-token" not in response_text
        assert "embed-input-secret" not in response_text
        assert "embed-input-token" not in response_text
        assert not (project / "embeddings" / "embeddings.jsonl").exists()


def test_embed_rejects_invalid_provider_vectors(tmp_path: Path) -> None:
    from md_to_rag import embed as embed_module

    class BadVectorProvider(embed_module.DeterministicHashEmbeddingProvider):
        def embed(self, content: str, *, chunk_id: str, content_hash: str):
            return ["not-a-number"]

    class OversizedIntegerVectorProvider(embed_module.DeterministicHashEmbeddingProvider):
        def __init__(self) -> None:
            super().__init__(dimensions=1)

        def embed(self, content: str, *, chunk_id: str, content_hash: str):
            return [10**400]

    for case_name, provider in (
        ("not-a-number", BadVectorProvider()),
        ("oversized-integer", OversizedIntegerVectorProvider()),
    ):
        project = tmp_path / case_name
        chunks_path = _prepare_chunked_project(project)

        response = embed_module.embed_project(chunks_path, provider=provider)

        assert response.status is CommandStatus.ERROR
        assert response.error is not None
        assert response.error.code == "embedding_vector_invalid"
        assert not (project / "embeddings" / "embeddings.jsonl").exists()


def test_embed_reports_missing_chunks_artifact_without_traceback(tmp_path: Path) -> None:
    project = tmp_path / "project"
    api.init(project)

    result = runner.invoke(
        app,
        ["embed", "--chunks", str(project / "chunks" / "chunks.jsonl"), "--json"],
        prog_name="md-to-rag",
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["command"] == "embed"
    assert payload["status"] == "missing_artifact"
    assert payload["error"]["code"] == "chunks_not_found"
    assert "traceback" not in result.output.lower()


def test_embed_rejects_invalid_jsonl_and_chunk_schema(tmp_path: Path) -> None:
    for case_name, artifact_text, expected_code in (
        ("invalid-jsonl", "{not json}\n", "chunks_invalid_jsonl"),
        (
            "invalid-schema",
            json.dumps(
                {
                    "schema_name": "md_to_rag.document",
                    "schema_version": "1.0",
                    "chunk_id": "chk_bad",
                    "doc_id": "doc_bad",
                    "source_id": "src_bad",
                    "source_path": "source/doc.md",
                    "content_hash": "sha256:bad",
                    "content": "Bad",
                    "metadata": {},
                    "provenance": {},
                    "heading_path": [],
                },
                sort_keys=True,
            )
            + "\n",
            "chunk_schema_invalid",
        ),
    ):
        project = tmp_path / case_name
        api.init(project)
        chunks_path = project / "chunks" / "chunks.jsonl"
        chunks_path.write_text(artifact_text, encoding="utf-8")

        response = api.embed(chunks=chunks_path)

        assert response.status is CommandStatus.ERROR
        assert response.error is not None
        assert response.error.code == expected_code
        assert not (project / "embeddings" / "embeddings.jsonl").exists()


def test_embed_rejects_corrupt_chunk_row_integrity(tmp_path: Path) -> None:
    project = tmp_path / "project"
    chunks_path = _prepare_chunked_project(project)
    rows = _jsonl(chunks_path)
    cases: dict[str, dict[str, Any]] = {
        "content-hash": {"content_hash": "sha256:not-the-content-hash"},
        "empty-chunk-id": {"chunk_id": ""},
        "metadata-list": {"metadata": []},
        "provenance-list": {"provenance": []},
        "heading-path-string": {"heading_path": "Top"},
    }

    for case_name, patch in cases.items():
        case_project = tmp_path / case_name
        api.init(case_project)
        case_chunks_path = case_project / "chunks" / "chunks.jsonl"
        _write_jsonl(case_chunks_path, [rows[0] | patch])

        response = api.embed(chunks=case_chunks_path)

        assert response.status is CommandStatus.ERROR
        assert response.error is not None
        assert response.error.code == "chunk_schema_invalid"
        assert not (case_project / "embeddings" / "embeddings.jsonl").exists()


def test_embed_rejects_duplicate_chunk_ids_and_nonportable_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    chunks_path = _prepare_chunked_project(project)
    rows = _jsonl(chunks_path)

    duplicate_project = tmp_path / "duplicate"
    api.init(duplicate_project)
    duplicate_rows = [rows[0], rows[1] | {"chunk_id": rows[0]["chunk_id"]}]
    _write_jsonl(duplicate_project / "chunks" / "chunks.jsonl", duplicate_rows)
    duplicate_response = api.embed(chunks=duplicate_project / "chunks" / "chunks.jsonl")
    assert duplicate_response.status is CommandStatus.ERROR
    assert duplicate_response.error is not None
    assert duplicate_response.error.code == "duplicate_chunk_id"

    bad_source_project = tmp_path / "bad-source"
    api.init(bad_source_project)
    _write_jsonl(
        bad_source_project / "chunks" / "chunks.jsonl",
        [rows[0] | {"source_path": "../source/doc.md"}],
    )
    bad_source_response = api.embed(chunks=bad_source_project / "chunks" / "chunks.jsonl")
    assert bad_source_response.status is CommandStatus.ERROR
    assert bad_source_response.error is not None
    assert bad_source_response.error.code == "chunk_schema_invalid"


def test_embed_accepts_empty_chunks_artifact(tmp_path: Path) -> None:
    project = tmp_path / "project"
    api.init(project)
    chunks_path = project / "chunks" / "chunks.jsonl"
    chunks_path.write_text("", encoding="utf-8")

    response = api.embed(chunks=chunks_path)

    assert response.status is CommandStatus.OK
    assert response.data.chunk_count == 0
    assert response.data.embedding_count == 0
    assert (project / "embeddings" / "embeddings.jsonl").read_text(encoding="utf-8") == ""


def test_embed_rejects_nonportable_chunks_artifact_paths(tmp_path: Path, monkeypatch) -> None:
    from md_to_rag import embed as embed_module

    project = tmp_path / "project"
    chunks_path = _prepare_chunked_project(project)
    original_relative_to_project = embed_module._relative_to_project

    def fake_relative_to_project(path: Path, project_root: Path):
        if path == chunks_path.resolve():
            return "chunks/CON.jsonl"
        return original_relative_to_project(path, project_root)

    monkeypatch.setattr(embed_module, "_relative_to_project", fake_relative_to_project)

    response = api.embed(chunks=chunks_path)

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "chunks_path_not_portable"
    assert not (project / "embeddings" / "embeddings.jsonl").exists()


def test_embed_rejects_windows_drive_relative_chunks_path(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    _prepare_chunked_project(project)
    monkeypatch.chdir(project)

    response = api.embed(chunks="C:chunks/chunks.jsonl")

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "chunks_path_not_portable"
    assert not (project / "embeddings" / "embeddings.jsonl").exists()


def test_embed_rejects_generated_embeddings_artifact_as_input(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _prepare_chunked_project(project)
    embeddings_path = project / "embeddings" / "embeddings.jsonl"
    embeddings_path.write_text("do not overwrite\n", encoding="utf-8")

    response = api.embed(chunks=embeddings_path)

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "chunks_artifact_collision"
    assert embeddings_path.read_text(encoding="utf-8") == "do not overwrite\n"


def test_embed_rejects_case_only_embeddings_artifact_alias_as_input(tmp_path: Path) -> None:
    project = tmp_path / "project"
    chunks_path = _prepare_chunked_project(project)
    alias_path = project / "embeddings" / "EMBEDDINGS.JSONL"
    alias_path.write_bytes(chunks_path.read_bytes())
    alias_text = alias_path.read_text(encoding="utf-8")

    response = api.embed(chunks=alias_path)

    assert response.status is CommandStatus.ERROR
    assert response.error is not None
    assert response.error.code == "chunks_artifact_collision"
    assert alias_path.read_text(encoding="utf-8") == alias_text
    output_path = project / "embeddings" / "embeddings.jsonl"
    if str(alias_path).casefold() != str(output_path).casefold():
        assert not (project / "embeddings" / "embeddings.jsonl").exists()
