"""Similarity-based retrieval for indexed log chunks."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.models import RetrievalResult
from src.services.chunk_store import ChunkStore
from src.services.exceptions import EmbeddingServiceError, PineconeServiceError, RetrievalServiceError
from src.services.gemini import GeminiEmbeddingService
from src.services.pinecone import PineconeService


logger = logging.getLogger(__name__)


class RetrievalService:
	"""Retrieve the most relevant indexed log chunks for a user query."""

	def __init__(
		self,
		embedding_service: GeminiEmbeddingService | None = None,
		pinecone_service: PineconeService | None = None,
		chunk_store: ChunkStore | None = None,
		default_top_k: int = 5,
	) -> None:
		self.embedding_service = embedding_service or GeminiEmbeddingService()
		self.pinecone_service = pinecone_service or PineconeService()
		self.chunk_store = chunk_store
		self.default_top_k = max(1, default_top_k)

	def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
		"""Return the top-K most relevant log chunks for a natural-language query."""

		normalized_query = query.strip()
		if not normalized_query:
			raise RetrievalServiceError("Query cannot be empty.")

		limit = self._normalize_top_k(top_k)

		try:
			query_embedding = self.embedding_service.embed_text(normalized_query)
		except EmbeddingServiceError as exc:
			raise RetrievalServiceError("Failed to generate an embedding for the query.") from exc

		try:
			matches = self.pinecone_service.query_vectors(query_embedding, top_k=limit)
		except PineconeServiceError as exc:
			raise RetrievalServiceError("Failed to retrieve similar log chunks from Pinecone.") from exc

		if not matches:
			logger.info("No retrieval matches found for query: %s", normalized_query)
			return []

		results: list[RetrievalResult] = []
		for match in matches:
			result = self._to_result(match)
			if result is None:
				continue
			results.append(result)

		return results

	def _normalize_top_k(self, top_k: int | None) -> int:
		if top_k is None:
			return self.default_top_k
		return max(1, top_k)

	def _to_result(self, match: dict[str, object]) -> RetrievalResult | None:
		metadata = dict(match.get("metadata") or {})
		chunk_id = self._as_string(metadata.get("chunk_id") or match.get("id"))
		retrieved_text = self._resolve_chunk_text(chunk_id)
		if retrieved_text is None:
			logger.warning("Skipping retrieved chunk %s because embedding text is unavailable in the local store.", chunk_id)
			return None

		return RetrievalResult(
			chunk_id=chunk_id,
			source_file=self._as_string(metadata.get("source_file")),
			timestamp=self._parse_timestamp(metadata.get("timestamp")),
			level=self._optional_string(metadata.get("level")),
			component=self._optional_string(metadata.get("component")),
			thread=self._optional_string(metadata.get("thread")),
			event_index=self._parse_int(metadata.get("event_index")),
			retrieved_text=retrieved_text,
			similarity_score=self._parse_float(match.get("score")),
		)

	def _resolve_chunk_text(self, chunk_id: str) -> str | None:
		if self.chunk_store is None:
			return None

		resolved_text = self.chunk_store.get(chunk_id)
		if resolved_text is None:
			return None

		text_value = resolved_text.strip()
		return text_value or None

	def _parse_timestamp(self, value: object | None) -> datetime | None:
		if value is None:
			return None
		if isinstance(value, datetime):
			return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)

		text_value = str(value).strip()
		if not text_value:
			return None

		try:
			parsed_value = datetime.fromisoformat(text_value.replace("Z", "+00:00"))
			return parsed_value if parsed_value.tzinfo is not None else parsed_value.replace(tzinfo=timezone.utc)
		except ValueError:
			return None

	def _parse_int(self, value: object | None) -> int | None:
		if value is None:
			return None
		try:
			return int(value)
		except (TypeError, ValueError):
			return None

	def _parse_float(self, value: object | None) -> float | None:
		if value is None:
			return None
		try:
			return float(value)
		except (TypeError, ValueError):
			return None

	def _as_string(self, value: object | None) -> str:
		if value is None:
			return ""
		return str(value).strip()

	def _optional_string(self, value: object | None) -> str | None:
		text_value = self._as_string(value)
		return text_value or None