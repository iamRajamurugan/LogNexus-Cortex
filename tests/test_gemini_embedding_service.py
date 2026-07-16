from __future__ import annotations

import socket

import pytest

from src.services.exceptions import InvalidApiKeyError, QuotaExceededError, TransientEmbeddingError
from src.services.gemini import GeminiEmbeddingService


class _QuotaError(Exception):
    def __init__(self) -> None:
        super().__init__("429 RESOURCE_EXHAUSTED: quota exceeded")
        self.code = 429


class _InvalidKeyError(Exception):
    def __init__(self) -> None:
        super().__init__("403 PERMISSION_DENIED: API key not valid")
        self.code = 403


class _TransientError(Exception):
    def __init__(self, code: int = 503) -> None:
        super().__init__(f"{code} SERVICE_UNAVAILABLE: temporary issue")
        self.code = code


class _FakeClient:
    def __init__(self, behavior):
        self.behavior = behavior
        self.calls = 0

    def embed_query(self, text):
        self.calls += 1
        return self.behavior(self.calls, text)

    def embed_documents(self, texts):
        self.calls += 1
        return self.behavior(self.calls, texts)


def _service() -> GeminiEmbeddingService:
    from src.config import AppSettings

    settings = AppSettings(
        project_root=None,  # type: ignore[arg-type]
        app_title="AI Log Intelligence Platform",
        app_env="test",
        log_level="INFO",
        max_upload_size_mb=200,
        gemini_api_key="test-key",
        gemini_model="gemini-2.5-flash",
        embedding_model="gemini-embedding-2",
        pinecone_index_dimension=3072,
        pinecone_metric="cosine",
        pinecone_api_key="",
        pinecone_index_name="",
        pinecone_namespace="",
        pinecone_cloud="",
        pinecone_region="",
        reports_dir=None,  # type: ignore[arg-type]
        assets_dir=None,  # type: ignore[arg-type]
    )
    return GeminiEmbeddingService(settings=settings, max_attempts=3, initial_backoff_seconds=0.0)


def test_successful_embedding(monkeypatch):
    service = _service()
    service._embeddings_client = _FakeClient(
        lambda _calls, payload: [
            [0.1, 0.2, 0.3]
            for _ in payload
        ] if isinstance(payload, list) else [0.1, 0.2, 0.3]
    )  # type: ignore[attr-defined]

    assert service.embed_text("hello") == [0.1, 0.2, 0.3]
    assert service.embed_texts(["a", "b"]) == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]


def test_quota_exhausted_raises_quota_error(monkeypatch, caplog):
    service = _service()
    service._embeddings_client = _FakeClient(lambda _calls, _payload: (_ for _ in ()).throw(_QuotaError()))  # type: ignore[attr-defined]
    sleep_calls: list[float] = []
    monkeypatch.setattr("src.services.gemini.time.sleep", lambda seconds: sleep_calls.append(seconds))

    with pytest.raises(QuotaExceededError):
        service.embed_text("hello")

    assert sleep_calls == []


def test_temporary_server_error_retries_then_succeeds(monkeypatch):
    service = _service()

    def behavior(call_number, _payload):
        if call_number < 3:
            raise _TransientError(503)
        return [1.0, 2.0, 3.0]

    service._embeddings_client = _FakeClient(behavior)  # type: ignore[attr-defined]
    sleep_calls: list[float] = []
    monkeypatch.setattr("src.services.gemini.time.sleep", lambda seconds: sleep_calls.append(seconds))

    result = service.embed_text("hello")

    assert result == [1.0, 2.0, 3.0]
    assert sleep_calls == [0.0, 0.0]


def test_invalid_api_key_raises_invalid_key_error():
    service = _service()
    service._embeddings_client = _FakeClient(lambda _calls, _payload: (_ for _ in ()).throw(_InvalidKeyError()))  # type: ignore[attr-defined]

    with pytest.raises(InvalidApiKeyError):
        service.embed_text("hello")


def test_network_timeout_is_retried(monkeypatch):
    service = _service()

    def behavior(call_number, _payload):
        if call_number < 3:
            raise socket.timeout("timed out")
        return [9.0, 9.0, 9.0]

    service._embeddings_client = _FakeClient(behavior)  # type: ignore[attr-defined]
    sleep_calls: list[float] = []
    monkeypatch.setattr("src.services.gemini.time.sleep", lambda seconds: sleep_calls.append(seconds))

    result = service.embed_text("hello")

    assert result == [9.0, 9.0, 9.0]
    assert sleep_calls == [0.0, 0.0]
