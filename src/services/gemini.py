"""Gemini service integration for chat and embeddings.

Chat behavior remains intentionally unimplemented. Embedding support is
provided here so the indexing pipeline can reuse a single initialized client.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cached_property
import logging
import time
from typing import Any

from src.config import AppSettings, get_settings
from src.models import LogChunk
from src.services.exceptions import (
	EmbeddingServiceError,
	InvalidApiKeyError,
	QuotaExceededError,
	TransientEmbeddingError,
)


logger = logging.getLogger(__name__)


class GeminiChatService:
	"""Placeholder for Gemini-powered chat and analysis generation."""

	def __init__(self, settings: AppSettings | None = None) -> None:
		self.settings = settings or get_settings()

	def generate_response(self, *args: object, **kwargs: object) -> str:
		raise NotImplementedError("Gemini chat behavior is not implemented yet.")


class GeminiEmbeddingService:
	"""Gemini text embedding generation via LangChain."""

	def __init__(
		self,
		settings: AppSettings | None = None,
		max_attempts: int = 3,
		initial_backoff_seconds: float = 1.0,
		backoff_multiplier: float = 2.0,
	) -> None:
		self.settings = settings or get_settings()
		self.max_attempts = max(1, max_attempts)
		self.initial_backoff_seconds = max(0.0, initial_backoff_seconds)
		self.backoff_multiplier = max(1.0, backoff_multiplier)

	@cached_property
	def _embeddings_client(self):
		"""Create the LangChain Gemini embeddings client once per instance."""

		if not self.settings.gemini_api_key:
			raise InvalidApiKeyError("GEMINI_API_KEY is required to generate embeddings.")

		try:
			from langchain_google_genai import GoogleGenerativeAIEmbeddings

			return GoogleGenerativeAIEmbeddings(
				model=self.settings.embedding_model,
				google_api_key=self.settings.gemini_api_key,
			)
		except Exception as exc:  # pragma: no cover - defensive wrapper around SDK init
			self._log_failure("Failed to initialize Gemini embeddings client.", exc)
			raise EmbeddingServiceError("Failed to initialize Gemini embeddings client.") from None

	def embed_text(self, text: str) -> list[float]:
		"""Embed a single chunk of text."""

		if not text.strip():
			return []

		return self._execute_embedding_operation("single embedding", lambda client: client.embed_query(text))

	def embed_texts(self, texts: list[str]) -> list[list[float]]:
		"""Embed a batch of texts using the same initialized client."""

		if not texts:
			return []

		return self._execute_embedding_operation("batch embedding", lambda client: client.embed_documents(texts))

	def embed_chunk(self, chunk: LogChunk) -> list[float]:
		"""Embed a chunk using the prepared embedding text."""

		return self.embed_text(chunk.embedding_text)

	def _execute_embedding_operation(self, operation_name: str, operation: Callable[[Any], Any]) -> Any:
		delay = self.initial_backoff_seconds
		last_error: Exception | None = None

		for attempt in range(1, self.max_attempts + 1):
			try:
				return operation(self._embeddings_client)
			except Exception as exc:
				last_error = exc
				if self._is_quota_error(exc):
					self._log_failure(f"Gemini API quota exhausted during {operation_name}.", exc)
					raise QuotaExceededError(
						"Gemini API quota has been exhausted. Please try again later or use another API key."
					) from None

				if self._is_invalid_api_key_error(exc):
					self._log_failure(f"Gemini API key rejected during {operation_name}.", exc)
					raise InvalidApiKeyError(
						"Gemini API key is invalid or unauthorized. Please check your environment configuration."
					) from None

				if self._is_transient_error(exc):
					if attempt < self.max_attempts:
						logger.warning(
							"Transient Gemini embedding error during %s (attempt %s/%s); retrying in %.1fs.",
							operation_name,
							attempt,
							self.max_attempts,
							delay,
						)
						time.sleep(delay)
						delay *= self.backoff_multiplier
						continue

					self._log_failure(
						f"Transient Gemini embedding error persisted during {operation_name} after {self.max_attempts} attempts.",
						exc,
					)
					raise TransientEmbeddingError(
						"Gemini service is temporarily unavailable. Please try again later."
					) from None

				self._log_failure(f"Gemini embedding failed during {operation_name}.", exc)
				raise EmbeddingServiceError("Gemini embedding failed.") from None

		if last_error is not None:
			self._log_failure(f"Gemini embedding failed during {operation_name}.", last_error)
		raise EmbeddingServiceError("Gemini embedding failed.") from None

	def _is_quota_error(self, exc: Exception) -> bool:
		code = self._extract_status_code(exc)
		message = str(exc).lower()
		name = type(exc).__name__.lower()
		return code == 429 or "resource_exhausted" in message or name in {"resourceexhausted", "toomanyrequests"}

	def _is_invalid_api_key_error(self, exc: Exception) -> bool:
		code = self._extract_status_code(exc)
		message = str(exc).lower()
		name = type(exc).__name__.lower()
		return code in {401, 403} or name in {"unauthenticated", "permissiondenied", "forbidden"} or "api key" in message

	def _is_transient_error(self, exc: Exception) -> bool:
		code = self._extract_status_code(exc)
		message = str(exc).lower()
		name = type(exc).__name__.lower()
		return (
			code in {503, 504}
			or isinstance(exc, (TimeoutError, ConnectionResetError))
			or name in {"serviceunavailable", "deadlineexceeded", "timeouterror", "timeout", "readtimeout"}
			or any(fragment in message for fragment in ("timeout", "timed out", "connection reset", "connection aborted", "temporarily unavailable"))
		)

	def _extract_status_code(self, exc: Exception) -> int | None:
		for attribute_name in ("code", "status_code"):
			attribute_value = getattr(exc, attribute_name, None)
			if callable(attribute_value):
				try:
					attribute_value = attribute_value()
				except Exception:
					attribute_value = None
			if isinstance(attribute_value, int):
				return attribute_value

		response = getattr(exc, "response", None)
		response_status = getattr(response, "status_code", None)
		if isinstance(response_status, int):
			return response_status

		return None

	def _log_failure(self, message: str, exc: Exception | None = None) -> None:
		logger.error(message)
		if exc is not None and logger.isEnabledFor(logging.DEBUG):
			logger.debug(message, exc_info=exc)
