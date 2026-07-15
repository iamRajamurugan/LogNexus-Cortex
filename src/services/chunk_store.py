"""Chunk text storage abstractions for retrieval-time resolution.

Version 1 uses an in-memory store. The interface is intentionally small so it
can later be replaced with Redis or a database without changing callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ChunkStore(ABC):
    """Abstract storage for mapping chunk IDs to embedding text."""

    @abstractmethod
    def put(self, chunk_id: str, embedding_text: str) -> None:
        """Store the embedding text for a chunk ID."""

    @abstractmethod
    def put_many(self, items: dict[str, str]) -> None:
        """Store many chunk texts in one call."""

    @abstractmethod
    def get(self, chunk_id: str) -> str | None:
        """Return the stored embedding text for a chunk ID, if available."""


class InMemoryChunkStore(ChunkStore):
    """Simple in-memory chunk store for Version 1."""

    def __init__(self) -> None:
        self._items: dict[str, str] = {}

    def put(self, chunk_id: str, embedding_text: str) -> None:
        self._items[chunk_id] = embedding_text

    def put_many(self, items: dict[str, str]) -> None:
        self._items.update(items)

    def get(self, chunk_id: str) -> str | None:
        return self._items.get(chunk_id)
