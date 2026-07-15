"""Normalized log domain models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class LogSeverity(str, Enum):
	"""Common log severity values."""

	TRACE = "TRACE"
	DEBUG = "DEBUG"
	INFO = "INFO"
	WARN = "WARN"
	ERROR = "ERROR"
	FATAL = "FATAL"


class NormalizedLogEntry(BaseModel):
	"""Normalized representation of a single log event."""

	timestamp: datetime | None = None
	level: str | None = None
	component: str | None = None
	thread: str | None = None
	message: str | None = None
	exception: str | None = None
	raw_log: str = ""
	source_file: str = ""
	line_number: int | None = None
	metadata: dict[str, str] = Field(default_factory=dict)


class LogChunk(BaseModel):
	"""Event-based chunk used for future retrieval and embedding preparation."""

	chunk_id: str = ""
	source_file: str = ""
	event_index: int = 0
	previous_event: NormalizedLogEntry | None = None
	current_event: NormalizedLogEntry
	next_event: NormalizedLogEntry | None = None
	embedding_text: str = ""
