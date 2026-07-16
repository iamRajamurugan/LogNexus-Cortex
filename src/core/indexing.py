"""Embedding generation and Pinecone indexing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Sequence

from src.models import LogChunk
from src.services.chunk_store import ChunkStore
from src.services.exceptions import (
	EmbeddingServiceError,
	InvalidApiKeyError,
	PineconeServiceError,
	QuotaExceededError,
	TransientEmbeddingError,
)
from src.services.gemini import GeminiEmbeddingService
from src.services.pinecone import PineconeService, PineconeVectorRecord


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IndexingResult:
	"""Summary of a batch indexing run."""

	total_chunks: int
	indexed_chunks: int
	skipped_chunk_ids: list[str] = field(default_factory=list)


class LogIndexingPipeline:
	"""Generate embeddings and store log chunks in Pinecone."""

	def __init__(
		self,
		embedding_service: GeminiEmbeddingService | None = None,
		pinecone_service: PineconeService | None = None,
		chunk_store: ChunkStore | None = None,
		batch_size: int = 8,
	) -> None:
		self.embedding_service = embedding_service or GeminiEmbeddingService()
		self.pinecone_service = pinecone_service or PineconeService()
		self.chunk_store = chunk_store
		self.batch_size = max(1, batch_size)

	def index_chunks(self, chunks: Sequence[LogChunk]) -> IndexingResult:
		"""Index a sequence of chunks into Pinecone."""

		total_chunks = len(chunks)
		indexed_chunks = 0
		skipped_chunk_ids: list[str] = []

		for batch in self._iter_batches(chunks):
			batch_records, batch_skipped, chunk_texts = self._prepare_batch(batch)
			skipped_chunk_ids.extend(batch_skipped)
			if not batch_records:
				continue

			try:
				self.pinecone_service.upsert_vectors(batch_records)
			except PineconeServiceError:
				raise

			if self.chunk_store is not None:
				self.chunk_store.put_many(chunk_texts)

			indexed_chunks += len(batch_records)

		return IndexingResult(
			total_chunks=total_chunks,
			indexed_chunks=indexed_chunks,
			skipped_chunk_ids=skipped_chunk_ids,
		)

	def _prepare_batch(self, batch: Sequence[LogChunk]) -> tuple[list[PineconeVectorRecord], list[str], dict[str, str]]:
		texts: list[str] = []
		valid_chunks: list[LogChunk] = []
		skipped_chunk_ids: list[str] = []

		for chunk in batch:
			embedding_text = chunk.embedding_text.strip()
			if not embedding_text:
				logger.warning("Skipping chunk %s because embedding text is empty.", chunk.chunk_id)
				skipped_chunk_ids.append(chunk.chunk_id)
				continue

			texts.append(embedding_text)
			valid_chunks.append(chunk)

		if not valid_chunks:
			return [], skipped_chunk_ids, {}

		try:
			embeddings = self.embedding_service.embed_texts(texts)
		except (QuotaExceededError, InvalidApiKeyError, TransientEmbeddingError):
			raise
		except EmbeddingServiceError:
			logger.warning("Batch embedding failed; falling back to per-chunk embedding.")
			return self._embed_batch_individually(valid_chunks, skipped_chunk_ids)

		batch_records = [
			self._build_vector_record(chunk, embedding)
			for chunk, embedding in zip(valid_chunks, embeddings, strict=True)
		]
		return batch_records, skipped_chunk_ids, {chunk.chunk_id: chunk.embedding_text for chunk in valid_chunks}

	def _embed_batch_individually(
		self,
		chunks: Sequence[LogChunk],
		skipped_chunk_ids: list[str],
	) -> tuple[list[PineconeVectorRecord], list[str], dict[str, str]]:
		records: list[PineconeVectorRecord] = []
		chunk_texts: dict[str, str] = {}
		for chunk in chunks:
			try:
				embedding = self.embedding_service.embed_chunk(chunk)
			except (QuotaExceededError, InvalidApiKeyError, TransientEmbeddingError):
				raise
			except EmbeddingServiceError:
				logger.warning("Skipping chunk %s because embedding generation failed.", chunk.chunk_id)
				skipped_chunk_ids.append(chunk.chunk_id)
				continue

			records.append(self._build_vector_record(chunk, embedding))
			chunk_texts[chunk.chunk_id] = chunk.embedding_text

		return records, skipped_chunk_ids, chunk_texts

	def _build_vector_record(self, chunk: LogChunk, embedding: list[float]) -> PineconeVectorRecord:
		metadata = self._build_metadata(chunk)
		return PineconeVectorRecord(id=chunk.chunk_id, values=embedding, metadata=metadata)

	def _build_metadata(self, chunk: LogChunk) -> dict[str, object]:
		current_event = chunk.current_event
		return self._clean_metadata({
			"chunk_id": chunk.chunk_id,
			"source_file": chunk.source_file,
			"timestamp": current_event.timestamp.isoformat() if current_event.timestamp is not None else None,
			"level": current_event.level,
			"component": current_event.component,
			"thread": current_event.thread,
			"event_index": chunk.event_index,
		})

	def _clean_metadata(self, metadata: dict[str, object | None]) -> dict[str, object]:
		"""Remove None values before sending metadata to Pinecone."""

		return {key: value for key, value in metadata.items() if value is not None}

	def _iter_batches(self, chunks: Sequence[LogChunk]) -> list[Sequence[LogChunk]]:
		return [chunks[index : index + self.batch_size] for index in range(0, len(chunks), self.batch_size)]