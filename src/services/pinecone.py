"""Pinecone service for index initialization and vector storage."""

from __future__ import annotations

from functools import cached_property
import logging
from typing import Any, TypedDict

from src.config import AppSettings, get_settings
from src.services.exceptions import PineconeServiceError


logger = logging.getLogger(__name__)


class PineconeVectorRecord(TypedDict):
	"""Vector payload accepted by Pinecone upserts."""

	id: str
	values: list[float]
	metadata: dict[str, object]


class PineconeMatch(TypedDict, total=False):
	"""Simplified Pinecone match payload used by the retrieval layer."""

	id: str
	score: float
	metadata: dict[str, object]


class PineconeService:
	"""Pinecone vector storage and index management."""

	def __init__(self, settings: AppSettings | None = None) -> None:
		self.settings = settings or get_settings()

	@cached_property
	def _client(self):
		"""Create the Pinecone client once per instance."""

		if not self.settings.pinecone_api_key:
			raise PineconeServiceError("PINECONE_API_KEY is required to access Pinecone.")

		try:
			from pinecone import Pinecone

			return Pinecone(api_key=self.settings.pinecone_api_key)
		except Exception as exc:  # pragma: no cover - defensive wrapper around SDK init
			raise PineconeServiceError("Failed to initialize Pinecone client.") from exc

	@cached_property
	def _index(self) -> Any:
		"""Create or reuse the configured Pinecone index."""

		return self.ensure_index()

	def ensure_index(self) -> Any:
		"""Return the configured index, creating it if needed."""

		index_name = self.settings.pinecone_index_name
		if not index_name:
			raise PineconeServiceError("PINECONE_INDEX_NAME is required to create or reuse an index.")

		if self._client.has_index(index_name):
			return self._client.Index(index_name)

		if not self.settings.pinecone_cloud or not self.settings.pinecone_region:
			raise PineconeServiceError("PINECONE_CLOUD and PINECONE_REGION are required to create a Pinecone index.")

		try:
			from pinecone import ServerlessSpec

			self._client.create_index(
				name=index_name,
				dimension=self.settings.pinecone_index_dimension,
				metric=self.settings.pinecone_metric,
				spec=ServerlessSpec(
					cloud=self.settings.pinecone_cloud,
					region=self.settings.pinecone_region,
				),
			)
			return self._client.Index(index_name)
		except Exception as exc:
			logger.exception("Failed to create Pinecone index %s.", index_name)
			raise PineconeServiceError(f"Failed to create Pinecone index '{index_name}'.") from exc

	def upsert_vectors(self, vectors: list[PineconeVectorRecord], namespace: str | None = None) -> None:
		"""Upsert prepared vectors into the configured Pinecone index."""

		if not vectors:
			return

		try:
			self._index.upsert(vectors=vectors, namespace=namespace or self.settings.pinecone_namespace or None)
		except Exception as exc:
			logger.exception("Pinecone upsert failed for index %s.", self.settings.pinecone_index_name)
			raise PineconeServiceError("Failed to upsert vectors into Pinecone.") from exc

	def query_vectors(
		self,
		vector: list[float],
		top_k: int,
		namespace: str | None = None,
	) -> list[PineconeMatch]:
		"""Run a similarity query against the configured Pinecone index."""

		if top_k <= 0:
			return []

		try:
			query_response = self._index.query(
				vector=vector,
				top_k=top_k,
				namespace=namespace or self.settings.pinecone_namespace or None,
				include_metadata=True,
				include_values=False,
			)
		except Exception as exc:
			logger.exception("Pinecone query failed for index %s.", self.settings.pinecone_index_name)
			raise PineconeServiceError("Failed to query Pinecone for similar vectors.") from exc

		matches = getattr(query_response, "matches", None) or []
		return [
			PineconeMatch(
				id=str(match.id),
				score=float(match.score) if getattr(match, "score", None) is not None else 0.0,
				metadata=dict(getattr(match, "metadata", {}) or {}),
			)
			for match in matches
		]

	def get_index(self) -> Any:
		"""Expose the reused Pinecone index for downstream callers."""

		return self._index
