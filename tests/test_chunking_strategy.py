from __future__ import annotations

from datetime import datetime, timezone

from src.core.chunking import LogChunker
from src.models import NormalizedLogEntry


def _event(
    message: str,
    *,
    level: str = "INFO",
    timestamp: str = "2026-07-13T10:15:00+00:00",
    component: str = "com.example.App",
    source_file: str = "app.log",
    exception: str | None = None,
    metadata: dict[str, str] | None = None,
) -> NormalizedLogEntry:
    return NormalizedLogEntry(
        timestamp=datetime.fromisoformat(timestamp),
        level=level,
        component=component,
        thread="main",
        message=message,
        exception=exception,
        raw_log=message,
        source_file=source_file,
        line_number=1,
        metadata=metadata or {},
    )


def test_chunk_embedding_text_is_bounded_and_human_readable():
    chunker = LogChunker()
    previous_event = _event(
        "previous event message " + ("x" * 900),
        timestamp="2026-07-13T10:14:58+00:00",
    )
    current_event = _event(
        "current event message " + ("y" * 900),
        timestamp="2026-07-13T10:15:00+00:00",
        exception="java.lang.RuntimeException: boom\n\tat com.example.App.run(App.java:10)",
        metadata={"requestId": "req-123", "traceId": "trace-456"},
    )
    next_event = _event(
        "next event message " + ("z" * 900),
        timestamp="2026-07-13T10:15:02+00:00",
        level="DEBUG",
    )

    embedding_text = chunker.build_embedding_text(current_event, previous_event, next_event)

    assert len(embedding_text) <= 2400
    assert "Timestamp:" in embedding_text
    assert "Component:" in embedding_text
    assert "Level:" in embedding_text
    assert "Exception:" in embedding_text
    assert "Metadata:" in embedding_text
    assert "Context:" in embedding_text
    assert "Raw Log:" not in embedding_text


def test_heartbeat_runs_are_compressed_into_a_single_chunk():
    chunker = LogChunker()
    events = [
        _event("heartbeat ping", timestamp=f"2026-07-13T10:15:0{i}+00:00")
        for i in range(6)
    ]
    events.append(_event("payment failed", level="ERROR", timestamp="2026-07-13T10:15:09+00:00"))

    chunks = chunker.chunk_events(events)

    assert len(chunks) == 2
    assert "heartbeat events omitted" in chunks[0].embedding_text.lower()
    assert len(chunks[0].embedding_text) <= 2400
