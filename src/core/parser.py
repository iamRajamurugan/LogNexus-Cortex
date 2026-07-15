"""Generic log parsing for Spring Boot, generic text, and JSON logs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import logging
import re
from datetime import datetime, timezone

from src.models import NormalizedLogEntry


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LogFileInput:
    """Input payload for a single uploaded log file."""

    source_file: str
    content: str


class LogParser:
    """Parse uploaded files into normalized log events."""

    _LEVEL_ALIASES = {
        "WARNING": "WARN",
        "WARN": "WARN",
        "ERROR": "ERROR",
        "ERR": "ERROR",
        "INFO": "INFO",
        "DEBUG": "DEBUG",
        "TRACE": "TRACE",
        "FATAL": "FATAL",
        "CRITICAL": "FATAL",
    }

    _SPRING_BOOT_PATTERN = re.compile(
        r"^(?P<timestamp>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[\.,]\d{1,6})?(?:Z|[+-]\d{2}:?\d{2})?)\s+"
        r"(?P<level>[A-Z]+)\s+"
        r"(?P<pid>\d+)\s+---\s+"
        r"\[(?P<thread>[^\]]+)\]\s+"
        r"(?P<component>[^:]+?)\s*:\s*"
        r"(?P<message>.*)$"
    )

    _TIMESTAMP_PREFIX_PATTERN = re.compile(
        r"^(?P<timestamp>(?:\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[\.,]\d{1,6})?(?:Z|[+-]\d{2}:?\d{2})?)|"
        r"(?:\d{4}/\d{2}/\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[\.,]\d{1,6})?)|"
        r"(?:\d{2}/\d{2}/\d{4}[ T]\d{2}:\d{2}:\d{2}(?:[\.,]\d{1,6})?)|"
        r"(?:\d{2}-[A-Za-z]{3}-\d{4}[ T]\d{2}:\d{2}:\d{2}(?:[\.,]\d{1,6})?))\b"
    )

    _GENERIC_LEVEL_PATTERN = re.compile(
        r"^(?P<level>TRACE|DEBUG|INFO|WARN|WARNING|ERROR|ERR|FATAL|CRITICAL)\b",
        re.IGNORECASE,
    )

    _JSON_CANDIDATES = ("{", "[")

    def parse_sources(self, sources: Sequence[LogFileInput]) -> list[NormalizedLogEntry]:
        """Parse multiple files independently and return one merged event list."""

        parsed_events: list[NormalizedLogEntry] = []
        for source in sources:
            if not source.content.strip():
                continue

            try:
                parsed_events.extend(self.parse_content(source.content, source.source_file))
            except Exception:  # pragma: no cover - defensive against malformed inputs
                logger.exception("Failed to parse source file %s", source.source_file)

        return parsed_events

    def parse_content(self, content: str, source_file: str) -> list[NormalizedLogEntry]:
        """Parse a single file's content into normalized events."""

        normalized_content = content.lstrip("\ufeff")
        if self._looks_like_json(normalized_content):
            parsed_json = self._parse_json_content(normalized_content, source_file)
            if parsed_json:
                return parsed_json

        return self._parse_text_content(normalized_content, source_file)

    def _looks_like_json(self, content: str) -> bool:
        stripped_content = content.lstrip()
        if not stripped_content:
            return False

        if stripped_content.startswith(self._JSON_CANDIDATES):
            return True

        first_line = stripped_content.splitlines()[0].strip()
        return first_line.startswith("{") or first_line.startswith("[")

    def _parse_json_content(self, content: str, source_file: str) -> list[NormalizedLogEntry]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return self._parse_json_lines(content, source_file)

        if isinstance(payload, list):
            return [entry for entry in (self._normalize_json_object(item, source_file) for item in payload) if entry is not None]

        if isinstance(payload, dict):
            return [self._normalize_json_object(payload, source_file)] if payload else []

        logger.warning("Skipping unsupported JSON payload in %s", source_file)
        return []

    def _parse_json_lines(self, content: str, source_file: str) -> list[NormalizedLogEntry]:
        parsed_events: list[NormalizedLogEntry] = []
        for line_number, raw_line in enumerate(content.splitlines(), start=1):
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue

            try:
                payload = json.loads(stripped_line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON line %s in %s", line_number, source_file)
                continue

            if isinstance(payload, dict):
                entry = self._normalize_json_object(payload, source_file, raw_log=stripped_line, line_number=line_number)
                parsed_events.append(entry)
            elif isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict):
                        parsed_events.append(
                            self._normalize_json_object(item, source_file, raw_log=stripped_line, line_number=line_number)
                        )
                    else:
                        parsed_events.append(
                            NormalizedLogEntry(
                                source_file=source_file,
                                raw_log=stripped_line,
                                line_number=line_number,
                                message=str(item),
                            )
                        )
            else:
                parsed_events.append(
                    NormalizedLogEntry(
                        source_file=source_file,
                        raw_log=stripped_line,
                        line_number=line_number,
                        message=str(payload),
                    )
                )

        return parsed_events

    def _normalize_json_object(
        self,
        payload: dict[str, object],
        source_file: str,
        raw_log: str | None = None,
        line_number: int | None = None,
    ) -> NormalizedLogEntry:
        flattened_payload = self._flatten_json(payload)
        normalized_metadata = {
            key: self._stringify_value(value)
            for key, value in flattened_payload.items()
            if key.lower() not in self._recognized_json_keys()
        }

        timestamp_value = self._first_present(flattened_payload, "timestamp", "time", "ts", "@timestamp", "datetime", "date_time")
        level_value = self._normalize_level(
            self._first_present(flattened_payload, "level", "severity", "log.level", "log_level")
        )
        component_value = self._first_present(
            flattened_payload,
            "component",
            "logger",
            "logger_name",
            "loggername",
            "class",
            "source",
            "source_class",
            "service",
            "service_name",
            "name",
        )
        thread_value = self._first_present(flattened_payload, "thread", "thread_name", "threadname", "thd")
        message_value = self._first_present(flattened_payload, "message", "msg", "log", "text", "description")
        exception_value = self._first_present(
            flattened_payload,
            "exception",
            "error",
            "stack_trace",
            "stacktrace",
            "trace",
            "throwable",
            "exception_message",
        )

        if message_value is None and exception_value is not None:
            message_value = exception_value

        return NormalizedLogEntry(
            timestamp=self._parse_datetime(timestamp_value),
            level=level_value,
            component=component_value,
            thread=thread_value,
            message=message_value,
            exception=exception_value,
            raw_log=raw_log if raw_log is not None else json.dumps(payload, ensure_ascii=False),
            source_file=source_file,
            line_number=line_number,
            metadata=normalized_metadata,
        )

    def _parse_text_content(self, content: str, source_file: str) -> list[NormalizedLogEntry]:
        event_blocks = self._split_text_into_event_blocks(content)
        parsed_events: list[NormalizedLogEntry] = []

        for event_index, event_block in enumerate(event_blocks, start=1):
            if not event_block.strip():
                continue

            parsed_event = self._parse_text_event(event_block, source_file, event_index)
            if parsed_event is not None:
                parsed_events.append(parsed_event)

        return parsed_events

    def _split_text_into_event_blocks(self, content: str) -> list[str]:
        blocks: list[str] = []
        current_lines: list[str] = []

        for raw_line in content.splitlines():
            line = raw_line.rstrip("\r")
            if self._is_event_start(line):
                if current_lines:
                    blocks.append("\n".join(current_lines).rstrip())
                current_lines = [line]
                continue

            if not current_lines and not line.strip():
                continue

            current_lines.append(line)

        if current_lines:
            blocks.append("\n".join(current_lines).rstrip())

        if not blocks and content.strip():
            blocks.append(content.strip())

        return blocks

    def _parse_text_event(self, event_block: str, source_file: str, event_index: int) -> NormalizedLogEntry | None:
        lines = event_block.splitlines()
        first_line = lines[0].strip() if lines else ""
        if not first_line:
            return None

        spring_boot_match = self._SPRING_BOOT_PATTERN.match(first_line)
        if spring_boot_match is not None:
            return self._build_spring_boot_event(event_block, source_file, spring_boot_match, event_index)

        return self._build_generic_text_event(event_block, source_file, event_index)

    def _build_spring_boot_event(
        self,
        event_block: str,
        source_file: str,
        match: re.Match[str],
        event_index: int,
    ) -> NormalizedLogEntry:
        message = match.group("message").strip()
        remaining_lines = event_block.splitlines()[1:]
        exception_text = self._extract_exception_text(message, remaining_lines)
        raw_message = message if not remaining_lines else "\n".join([message, *remaining_lines]).rstrip()

        return NormalizedLogEntry(
            timestamp=self._parse_datetime(match.group("timestamp")),
            level=self._normalize_level(match.group("level")),
            component=match.group("component").strip() or None,
            thread=match.group("thread").strip() or None,
            message=message or None,
            exception=exception_text,
            raw_log=event_block,
            source_file=source_file,
            line_number=event_index,
            metadata={"parser": "spring_boot", "raw_message": raw_message},
        )

    def _build_generic_text_event(self, event_block: str, source_file: str, event_index: int) -> NormalizedLogEntry:
        lines = event_block.splitlines()
        first_line = lines[0].strip() if lines else ""
        timestamp_match = self._TIMESTAMP_PREFIX_PATTERN.match(first_line)

        timestamp_value: str | None = None
        content_after_timestamp = first_line
        if timestamp_match is not None:
            timestamp_value = timestamp_match.group("timestamp")
            content_after_timestamp = first_line[timestamp_match.end():].strip()

        level_value = None
        level_match = self._GENERIC_LEVEL_PATTERN.match(content_after_timestamp)
        if level_match is not None:
            level_value = self._normalize_level(level_match.group("level"))
            content_after_timestamp = content_after_timestamp[level_match.end():].strip(" -:\t")

        message_lines = [content_after_timestamp] if content_after_timestamp else []
        message_lines.extend(line for line in lines[1:] if line is not None)
        message = "\n".join(line for line in message_lines if line is not None).strip() or None
        exception_text = self._extract_exception_text(message, lines[1:])

        return NormalizedLogEntry(
            timestamp=self._parse_datetime(timestamp_value),
            level=level_value,
            component=None,
            thread=None,
            message=message,
            exception=exception_text,
            raw_log=event_block,
            source_file=source_file,
            line_number=event_index,
            metadata={"parser": "generic_text"},
        )

    def _is_event_start(self, line: str) -> bool:
        stripped_line = line.lstrip()
        return bool(self._SPRING_BOOT_PATTERN.match(stripped_line) or self._TIMESTAMP_PREFIX_PATTERN.match(stripped_line))

    def _extract_exception_text(self, message: str | None, trailing_lines: Sequence[str]) -> str | None:
        exception_lines: list[str] = []
        if message and self._looks_like_exception(message):
            exception_lines.append(message)

        for line in trailing_lines:
            if self._looks_like_stack_trace_line(line):
                exception_lines.append(line.rstrip())

        if not exception_lines:
            return None

        return "\n".join(exception_lines).strip() or None

    def _looks_like_exception(self, text: str) -> bool:
        lowered_text = text.lower()
        return "exception" in lowered_text or "error" in lowered_text or "traceback" in lowered_text

    def _looks_like_stack_trace_line(self, line: str) -> bool:
        stripped_line = line.lstrip()
        return stripped_line.startswith(("at ", "Caused by:", "Suppressed:", "Traceback ")) or "Exception" in stripped_line

    def _parse_datetime(self, value: object | None) -> datetime | None:
        if value is None:
            return None

        if isinstance(value, datetime):
            return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)

        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)

        text_value = str(value).strip()
        if not text_value:
            return None

        normalized_value = text_value.replace("Z", "+00:00")
        candidates = (
            "%Y-%m-%d %H:%M:%S,%f%z",
            "%Y-%m-%d %H:%M:%S.%f%z",
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S.%f",
            "%Y/%m/%d %H:%M:%S",
            "%d/%m/%Y %H:%M:%S.%f",
            "%d/%m/%Y %H:%M:%S",
            "%d-%b-%Y %H:%M:%S.%f",
            "%d-%b-%Y %H:%M:%S",
        )

        for candidate in candidates:
            try:
                parsed_value = datetime.strptime(normalized_value, candidate)
                return parsed_value if parsed_value.tzinfo is not None else parsed_value.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        try:
            parsed_value = datetime.fromisoformat(normalized_value)
            return parsed_value if parsed_value.tzinfo is not None else parsed_value.replace(tzinfo=timezone.utc)
        except ValueError:
            logger.debug("Unable to parse timestamp value: %s", text_value)
            return None

    def _normalize_level(self, value: object | None) -> str | None:
        if value is None:
            return None

        normalized_value = str(value).strip().upper()
        return self._LEVEL_ALIASES.get(normalized_value, normalized_value or None)

    def _flatten_json(self, payload: dict[str, object], prefix: str = "") -> dict[str, object]:
        flattened: dict[str, object] = {}
        for key, value in payload.items():
            compound_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flattened.update(self._flatten_json(value, compound_key))
            else:
                flattened[compound_key] = value
        return flattened

    def _first_present(self, payload: dict[str, object], *keys: str) -> str | None:
        normalized_lookup = {key.lower(): value for key, value in payload.items()}
        for key in keys:
            if key.lower() in normalized_lookup:
                value = normalized_lookup[key.lower()]
                if value is None:
                    return None
                text_value = self._stringify_value(value)
                return text_value or None
        return None

    def _stringify_value(self, value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, list):
            return "\n".join(self._stringify_value(item) for item in value if item is not None).strip()
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return str(value).strip()

    def _recognized_json_keys(self) -> set[str]:
        return {
            "timestamp",
            "time",
            "ts",
            "@timestamp",
            "datetime",
            "date_time",
            "level",
            "severity",
            "log.level",
            "log_level",
            "component",
            "logger",
            "logger_name",
            "loggername",
            "class",
            "source",
            "source_class",
            "service",
            "service_name",
            "name",
            "thread",
            "thread_name",
            "threadname",
            "thd",
            "message",
            "msg",
            "log",
            "text",
            "description",
            "exception",
            "error",
            "stack_trace",
            "stacktrace",
            "trace",
            "throwable",
            "exception_message",
        }