"""Retrieval result models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class RetrievalResult(BaseModel):
	"""Normalized similarity-search result returned by the retrieval layer."""

	chunk_id: str
	source_file: str
	timestamp: datetime | None = None
	level: str | None = None
	component: str | None = None
	thread: str | None = None
	event_index: int | None = None
	retrieved_text: str
	similarity_score: float | None = None