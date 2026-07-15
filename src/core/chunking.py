"""Event-based log chunking and embedding text preparation."""

from __future__ import annotations

from collections.abc import Sequence

from src.models import LogChunk, NormalizedLogEntry


class LogChunker:
    """Build one chunk per logical log event with lightweight context."""

    def chunk_events(self, events: Sequence[NormalizedLogEntry]) -> list[LogChunk]:
        """Convert normalized events into event-based chunks."""

        grouped_events: dict[str, list[NormalizedLogEntry]] = {}
        file_order: list[str] = []

        for event in events:
            if event.source_file not in grouped_events:
                grouped_events[event.source_file] = []
                file_order.append(event.source_file)
            grouped_events[event.source_file].append(event)

        chunks: list[LogChunk] = []
        for source_file in file_order:
            source_events = grouped_events[source_file]
            for index, current_event in enumerate(source_events):
                previous_event = source_events[index - 1] if index > 0 else None
                next_event = source_events[index + 1] if index + 1 < len(source_events) else None
                chunk = LogChunk(
                    chunk_id=f"{source_file}:{index + 1}",
                    source_file=source_file,
                    event_index=index + 1,
                    previous_event=previous_event,
                    current_event=current_event,
                    next_event=next_event,
                    embedding_text=self.build_embedding_text(current_event, previous_event, next_event),
                )
                chunks.append(chunk)

        return chunks

    def build_embedding_text(
        self,
        current_event: NormalizedLogEntry,
        previous_event: NormalizedLogEntry | None = None,
        next_event: NormalizedLogEntry | None = None,
    ) -> str:
        """Create a readable text block for future embedding generation."""

        sections = [
            f"Timestamp:\n{self._format_value(current_event.timestamp)}",
            f"Level:\n{self._format_value(current_event.level)}",
            f"Component:\n{self._format_value(current_event.component)}",
            f"Thread:\n{self._format_value(current_event.thread)}",
            f"Message:\n{self._format_value(current_event.message)}",
            f"Exception:\n{self._format_value(current_event.exception)}",
            f"Raw Log:\n{self._format_value(current_event.raw_log)}",
        ]

        context_sections: list[str] = []
        if previous_event is not None:
            context_sections.append(f"Previous Event:\n{self._format_event(previous_event)}")
        if next_event is not None:
            context_sections.append(f"Next Event:\n{self._format_event(next_event)}")

        if context_sections:
            sections.append("Context:\n" + "\n\n".join(context_sections))

        return "\n\n".join(sections).strip()

    def _format_event(self, event: NormalizedLogEntry) -> str:
        return self.build_embedding_text(event).strip()

    def _format_value(self, value: object | None) -> str:
        if value is None:
            return "None"
        if isinstance(value, str):
            return value.strip() or "None"
        return str(value).strip() or "None"