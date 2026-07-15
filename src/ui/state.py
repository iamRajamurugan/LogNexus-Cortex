"""Streamlit session state helpers for the AI Log Intelligence Platform."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core import LogChunker, LogIndexingPipeline, LogParser, RetrievalService
from src.prompts import PromptManager
from src.services.chunk_store import InMemoryChunkStore
from src.services.analysis_service import AnalysisService
from src.services.gemini import GeminiEmbeddingService
from src.services.pinecone import PineconeService


@dataclass(slots=True)
class UploadedLogFile:
    """In-memory representation of one uploaded log file."""

    name: str
    content: str


@dataclass(slots=True)
class ProcessingSummary:
    """Summary of the current processing run."""

    files_uploaded: int = 0
    events_parsed: int = 0
    chunks_created: int = 0
    vectors_stored: int = 0
    processing_time_seconds: float = 0.0
    last_run_at: datetime | None = None
    status_message: str = "No logs processed yet."


def initialize_session_state(session_state: Any) -> None:
    """Initialize all Streamlit session_state entries used by the app."""

    defaults: dict[str, object] = {
        "uploaded_log_files": [],
        "processed_chunks": [],
        "analysis_history": [],
        "processing_summary": ProcessingSummary(),
        "logs_indexed": False,
        "processing_error": "",
        "analysis_error": "",
        "retrieval_top_k": 5,
        "analysis_type": "General Chat",
        "app_services": None,
    }

    for key, value in defaults.items():
        if key not in session_state:
            session_state[key] = value


def get_app_services(session_state: Any) -> dict[str, object]:
    """Create or reuse backend service objects for the current session."""

    services = session_state.get("app_services")
    if services is not None:
        return services

    parser = LogParser()
    chunker = LogChunker()
    chunk_store = InMemoryChunkStore()
    embedding_service = GeminiEmbeddingService()
    pinecone_service = PineconeService()
    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        pinecone_service=pinecone_service,
        chunk_store=chunk_store,
    )
    indexing_pipeline = LogIndexingPipeline(
        embedding_service=embedding_service,
        pinecone_service=pinecone_service,
        chunk_store=chunk_store,
    )
    analysis_service = AnalysisService(
        retrieval_service=retrieval_service,
        prompt_manager=PromptManager(),
    )

    services = {
        "parser": parser,
        "chunker": chunker,
        "chunk_store": chunk_store,
        "embedding_service": embedding_service,
        "pinecone_service": pinecone_service,
        "retrieval_service": retrieval_service,
        "indexing_pipeline": indexing_pipeline,
        "analysis_service": analysis_service,
    }
    session_state["app_services"] = services
    return services
