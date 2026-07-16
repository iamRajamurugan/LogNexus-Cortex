from __future__ import annotations

import pytest

from src.core.indexing import LogIndexingPipeline
from src.models import LogChunk, NormalizedLogEntry
from src.services.chunk_store import InMemoryChunkStore
from src.services.exceptions import QuotaExceededError


class _FakeEmbeddingService:
    def __init__(self, mode: str = "ok") -> None:
        self.mode = mode
        self.batch_calls = 0
        self.single_calls = 0

    def embed_texts(self, texts):
        self.batch_calls += 1
        if self.mode == "quota":
            raise QuotaExceededError("Gemini API quota has been exhausted. Please try again later or use another API key.")
        return [[float(len(text))] * 3 for text in texts]

    def embed_chunk(self, chunk):
        self.single_calls += 1
        if self.mode == "quota":
            raise QuotaExceededError("Gemini API quota has been exhausted. Please try again later or use another API key.")
        return [float(len(chunk.embedding_text))] * 3


class _FakePineconeService:
    def __init__(self) -> None:
        self.vectors = []

    def upsert_vectors(self, vectors, namespace=None):
        self.vectors.extend(vectors)


def _chunk() -> LogChunk:
    event = NormalizedLogEntry(
        timestamp=None,
        level="INFO",
        component="app",
        thread="main",
        message="started",
        exception=None,
        raw_log="raw",
        source_file="app.log",
        line_number=1,
        metadata={},
    )
    return LogChunk(
        chunk_id="app.log:1",
        source_file="app.log",
        event_index=1,
        previous_event=None,
        current_event=event,
        next_event=None,
        embedding_text="Timestamp:\n2026-07-13",
    )


def test_indexing_successfully_stores_chunk_text():
    chunk_store = InMemoryChunkStore()
    pipeline = LogIndexingPipeline(
        embedding_service=_FakeEmbeddingService(),
        pinecone_service=_FakePineconeService(),
        chunk_store=chunk_store,
    )

    result = pipeline.index_chunks([_chunk()])

    assert result.indexed_chunks == 1
    assert chunk_store.get("app.log:1") == "Timestamp:\n2026-07-13"


def test_default_batch_size_is_smaller_for_tpm_safety():
    pipeline = LogIndexingPipeline(
        embedding_service=_FakeEmbeddingService(),
        pinecone_service=_FakePineconeService(),
    )

    assert pipeline.batch_size == 8


def test_quota_exhaustion_aborts_indexing_immediately():
    embedding_service = _FakeEmbeddingService(mode="quota")
    pinecone_service = _FakePineconeService()
    pipeline = LogIndexingPipeline(
        embedding_service=embedding_service,
        pinecone_service=pinecone_service,
        chunk_store=InMemoryChunkStore(),
    )

    with pytest.raises(QuotaExceededError):
        pipeline.index_chunks([_chunk(), _chunk()])

    assert embedding_service.batch_calls == 1
    assert embedding_service.single_calls == 0
    assert pinecone_service.vectors == []
