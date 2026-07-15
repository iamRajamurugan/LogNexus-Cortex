"""Gemini service integration for chat and embeddings.

Chat behavior remains intentionally unimplemented. Embedding support is
provided here so the indexing pipeline can reuse a single initialized client.
"""

from __future__ import annotations

from functools import cached_property
import logging

from src.config import AppSettings, get_settings
from src.models import LogChunk
from src.services.exceptions import EmbeddingServiceError


logger = logging.getLogger(__name__)


class GeminiChatService:
	"""Placeholder for Gemini-powered chat and analysis generation."""

	def __init__(self, settings: AppSettings | None = None) -> None:
		self.settings = settings or get_settings()

	def generate_response(self, *args: object, **kwargs: object) -> str:
		raise NotImplementedError("Gemini chat behavior is not implemented yet.")


class GeminiEmbeddingService:
	"""Gemini text embedding generation via LangChain."""

	def __init__(self, settings: AppSettings | None = None) -> None:
		self.settings = settings or get_settings()

	@cached_property
	def _embeddings_client(self):
		"""Create the LangChain Gemini embeddings client once per instance."""

		if not self.settings.gemini_api_key:
			raise EmbeddingServiceError("GEMINI_API_KEY is required to generate embeddings.")

		try:
			from langchain_google_genai import GoogleGenerativeAIEmbeddings

			return GoogleGenerativeAIEmbeddings(
				model=self.settings.embedding_model,
				google_api_key=self.settings.gemini_api_key,
			)
		except Exception as exc:  # pragma: no cover - defensive wrapper around SDK init
			raise EmbeddingServiceError("Failed to initialize Gemini embeddings client.") from exc

	def embed_text(self, text: str) -> list[float]:
		"""Embed a single chunk of text."""

		try:
			return self._embeddings_client.embed_query(text)
		except Exception as exc:
			logger.exception("Gemini embedding failed for a single text input.")
			raise EmbeddingServiceError("Gemini embedding failed for a single text input.") from exc

	def embed_texts(self, texts: list[str]) -> list[list[float]]:
		"""Embed a batch of texts using the same initialized client."""

		try:
			return self._embeddings_client.embed_documents(texts)
		except Exception as exc:
			logger.exception("Gemini batch embedding failed.")
			raise EmbeddingServiceError("Gemini batch embedding failed.") from exc

	def embed_chunk(self, chunk: LogChunk) -> list[float]:
		"""Embed a chunk using the prepared embedding text."""

		return self.embed_text(chunk.embedding_text)
