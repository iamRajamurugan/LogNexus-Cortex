"""Service-level exceptions for embedding and vector storage."""

from __future__ import annotations


class EmbeddingServiceError(RuntimeError):
    """Raised when Gemini embedding generation fails."""


class PineconeServiceError(RuntimeError):
    """Raised when Pinecone initialization or indexing fails."""


class RetrievalServiceError(RuntimeError):
    """Raised when log retrieval cannot be completed."""
