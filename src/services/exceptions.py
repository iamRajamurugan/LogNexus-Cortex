"""Service-level exceptions for embedding and vector storage."""

from __future__ import annotations


class EmbeddingServiceError(RuntimeError):
    """Raised when Gemini embedding generation fails."""


class QuotaExceededError(EmbeddingServiceError):
    """Raised when Gemini returns a 429 RESOURCE_EXHAUSTED error."""


class InvalidApiKeyError(EmbeddingServiceError):
    """Raised when Gemini rejects the API key or access credentials."""


class TransientEmbeddingError(EmbeddingServiceError):
    """Raised when a retryable Gemini embedding error persists after retries."""


class PineconeServiceError(RuntimeError):
    """Raised when Pinecone initialization or indexing fails."""


class RetrievalServiceError(RuntimeError):
    """Raised when log retrieval cannot be completed."""
