"""Event-aware log chunking and embedding text preparation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re

from src.models import LogChunk, NormalizedLogEntry


@dataclass(frozen=True, slots=True)
class _EventWindow:
    """Internal window of events used to build one chunk."""

    source_file: str
    chunk_id: str
    event_index: int
    current_event: NormalizedLogEntry
    previous_event: NormalizedLogEntry | None = None
    next_event: NormalizedLogEntry | None = None
    heartbeat_summary: str | None = None


class LogChunker:
    """Build bounded, event-aware chunks for log retrieval."""

    MAX_EMBEDDING_CHARS = 2400
    HEARTBEAT_RUN_THRESHOLD = 5
    HEARTBEAT_PATTERNS = (
        re.compile(r"\bheartbeat\b", re.IGNORECASE),
        re.compile(r"\bkeep[- ]?alive\b", re.IGNORECASE),
        re.compile(r"\bhealth(?:\s+check)?\b", re.IGNORECASE),
        re.compile(r"\bping\b", re.IGNORECASE),
        re.compile(r"\bpoll(?:ing)?\b", re.IGNORECASE),
        re.compile(r"\bnoop\b", re.IGNORECASE),
    )
    CORRELATION_KEYS = ("requestid", "orderid", "traceid", "sessionid")

    def chunk_events(self, events: Sequence[NormalizedLogEntry]) -> list[LogChunk]:
        """Convert normalized events into bounded chunks."""

        grouped_events: dict[str, list[NormalizedLogEntry]] = {}
        file_order: list[str] = []

        for event in events:
            source_file = event.source_file or "unknown"
            if source_file not in grouped_events:
                grouped_events[source_file] = []
                file_order.append(source_file)
            grouped_events[source_file].append(event)

        chunks: list[LogChunk] = []
        for source_file in file_order:
            chunks.extend(self._chunk_source_events(source_file, grouped_events[source_file]))

        return chunks

    def build_embedding_text(
        self,
        current_event: NormalizedLogEntry,
        previous_event: NormalizedLogEntry | None = None,
        next_event: NormalizedLogEntry | None = None,
    ) -> str:
        """Create a readable text block for future embedding generation."""

        return self._render_embedding_text(current_event=current_event, previous_event=previous_event, next_event=next_event)

    def _chunk_source_events(self, source_file: str, source_events: Sequence[NormalizedLogEntry]) -> list[LogChunk]:
        chunks: list[LogChunk] = []
        index = 0

        while index < len(source_events):
            current_event = source_events[index]
            run_end = self._heartbeat_run_end(source_events, index)

            if self._is_heartbeat_event(current_event) and run_end - index + 1 > self.HEARTBEAT_RUN_THRESHOLD:
                previous_event = source_events[index - 1] if index > 0 else None
                next_event = source_events[run_end + 1] if run_end + 1 < len(source_events) else None
                window = self._build_heartbeat_window(
                    source_file=source_file,
                    start_index=index,
                    end_index=run_end,
                    current_event=current_event,
                    previous_event=previous_event,
                    next_event=next_event,
                    start_run_event=source_events[index],
                    end_run_event=source_events[run_end],
                )
                chunks.append(self._build_chunk(window))
                index = run_end + 1
                continue

            previous_event = source_events[index - 1] if index > 0 else None
            next_event = source_events[index + 1] if index + 1 < len(source_events) else None
            window = _EventWindow(
                source_file=source_file,
                chunk_id=f"{source_file}:{index + 1}",
                event_index=index + 1,
                current_event=current_event,
                previous_event=previous_event,
                next_event=next_event,
            )
            chunks.append(self._build_chunk(window))
            index += 1

        return chunks

    def _build_chunk(self, window: _EventWindow) -> LogChunk:
        return LogChunk(
            chunk_id=window.chunk_id,
            source_file=window.source_file,
            event_index=window.event_index,
            previous_event=window.previous_event,
            current_event=window.current_event,
            next_event=window.next_event,
            embedding_text=self._render_embedding_text(
                current_event=window.current_event,
                previous_event=window.previous_event,
                next_event=window.next_event,
                heartbeat_summary=window.heartbeat_summary,
            ),
        )

    def _render_embedding_text(
        self,
        current_event: NormalizedLogEntry,
        previous_event: NormalizedLogEntry | None = None,
        next_event: NormalizedLogEntry | None = None,
        heartbeat_summary: str | None = None,
    ) -> str:
        current_sections = [
            self._format_section("Timestamp", self._format_value(current_event.timestamp)),
            self._format_section("Component", self._format_value(current_event.component)),
            self._format_section("Level", self._format_value(current_event.level)),
            self._format_section("Thread", self._format_value(current_event.thread)),
            self._format_section("Message", self._format_value(current_event.message)),
        ]

        exception_value = self._format_value(current_event.exception)
        if exception_value != "None":
            current_sections.append(self._format_section("Exception", exception_value))

        metadata_text = self._format_metadata(current_event)
        context_text = self._build_context_text(previous_event, next_event, heartbeat_summary)

        candidate_sections = [section for section in current_sections if section]
        if metadata_text:
            candidate_sections.append(self._format_section("Metadata", metadata_text))
        if context_text:
            candidate_sections.append(self._format_section("Context", context_text))

        rendered_text = self._join_sections(candidate_sections)
        if len(rendered_text) <= self.MAX_EMBEDDING_CHARS:
            return rendered_text

        pruned_variants: list[list[str]] = []
        pruned_variants.append([section for section in candidate_sections if not section.startswith("Context:\n")])
        pruned_variants.append([section for section in candidate_sections if not section.startswith("Metadata:\n")])
        pruned_variants.append([section for section in candidate_sections if not section.startswith("Thread:\n")])
        pruned_variants.append([section for section in candidate_sections if not section.startswith("Previous Event:\n")])
        pruned_variants.append([section for section in candidate_sections if not section.startswith("Next Event:\n")])

        for variant in pruned_variants:
            variant_text = self._join_sections(variant)
            if len(variant_text) <= self.MAX_EMBEDDING_CHARS:
                return variant_text

        return self._truncate_to_limit(rendered_text, self.MAX_EMBEDDING_CHARS)

    def _build_heartbeat_window(
        self,
        source_file: str,
        start_index: int,
        end_index: int,
        current_event: NormalizedLogEntry,
        previous_event: NormalizedLogEntry | None,
        next_event: NormalizedLogEntry | None,
        start_run_event: NormalizedLogEntry,
        end_run_event: NormalizedLogEntry,
    ) -> _EventWindow:
        omitted_count = end_index - start_index + 1
        summary = self._heartbeat_summary(source_file, start_run_event, end_run_event, omitted_count)
        return _EventWindow(
            source_file=source_file,
            chunk_id=f"{source_file}:{start_index + 1}-hb",
            event_index=start_index + 1,
            current_event=current_event,
            previous_event=previous_event,
            next_event=next_event,
            heartbeat_summary=summary,
        )

    def _heartbeat_summary(
        self,
        source_file: str,
        start_event: NormalizedLogEntry,
        end_event: NormalizedLogEntry,
        omitted_count: int,
    ) -> str:
        start_timestamp = self._format_value(start_event.timestamp)
        end_timestamp = self._format_value(end_event.timestamp)
        return (
            f"{omitted_count} normal heartbeat events omitted\n"
            f"Source File: {source_file}\n"
            f"Between: {start_timestamp} and {end_timestamp}\n"
            f"Representative Message: {self._format_value(start_event.message)}"
        )

    def _build_context_text(
        self,
        previous_event: NormalizedLogEntry | None,
        next_event: NormalizedLogEntry | None,
        heartbeat_summary: str | None,
    ) -> str:
        context_parts: list[str] = []
        if previous_event is not None:
            context_parts.append(f"Previous Event: {self._summarize_event(previous_event)}")
        if next_event is not None:
            context_parts.append(f"Next Event: {self._summarize_event(next_event)}")
        if heartbeat_summary is not None:
            context_parts.append(f"Heartbeat Summary:\n{heartbeat_summary}")
        return "\n\n".join(context_parts).strip()

    def _summarize_event(self, event: NormalizedLogEntry) -> str:
        summary_lines = [
            f"Timestamp: {self._format_value(event.timestamp)}",
            f"Component: {self._format_value(event.component)}",
            f"Level: {self._format_value(event.level)}",
            f"Message: {self._truncate_to_limit(self._format_value(event.message), 400)}",
        ]
        if event.exception:
            summary_lines.append(f"Exception: {self._truncate_to_limit(event.exception, 400)}")
        return " | ".join(summary_lines)

    def _format_metadata(self, event: NormalizedLogEntry) -> str:
        correlation_values: list[str] = []
        for key in self.CORRELATION_KEYS:
            value = self._lookup_metadata_value(event.metadata, key)
            if value:
                correlation_values.append(f"{self._display_key(key)}: {value}")
        return "\n".join(correlation_values)

    def _lookup_metadata_value(self, metadata: dict[str, str], key: str) -> str | None:
        normalized_key = key.lower()
        for raw_key, raw_value in metadata.items():
            raw_key_normalized = re.sub(r"[^a-z0-9]", "", raw_key.lower())
            if raw_key_normalized == normalized_key:
                value = raw_value.strip()
                return value or None
        return None

    def _display_key(self, key: str) -> str:
        return key[0].upper() + key[1:]

    def _join_sections(self, sections: Sequence[str]) -> str:
        return "\n\n".join(section for section in sections if section).strip()

    def _format_section(self, title: str, value: str | None) -> str:
        if value is None or not value.strip():
            return ""
        return f"{title}:\n{value.strip()}"

    def _truncate_to_limit(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 40)].rstrip() + "\n[truncated to fit embedding limit]"

    def _heartbeat_run_end(self, events: Sequence[NormalizedLogEntry], start_index: int) -> int:
        end_index = start_index
        while end_index + 1 < len(events) and self._is_heartbeat_event(events[end_index + 1]):
            end_index += 1
        return end_index

    def _is_heartbeat_event(self, event: NormalizedLogEntry) -> bool:
        if not self._is_low_signal_level(event.level):
            return False

        message = f"{event.message or ''} {event.raw_log or ''}".strip()
        if not message:
            return False

        return any(pattern.search(message) for pattern in self.HEARTBEAT_PATTERNS)

    def _is_low_signal_level(self, level: str | None) -> bool:
        if level is None:
            return False
        normalized_level = level.strip().upper()
        return normalized_level in {"INFO", "DEBUG", "TRACE"}

    def _format_value(self, value: object | None) -> str:
        if value is None:
            return "None"
        if isinstance(value, str):
            return value.strip() or "None"
        return str(value).strip() or "None"