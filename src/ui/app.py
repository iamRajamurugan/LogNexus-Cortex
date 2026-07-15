"""Streamlit application entrypoint for the AI Log Intelligence Platform."""

from __future__ import annotations

from time import perf_counter

import streamlit as st

from src.config import get_settings
from src.core import LogChunker, LogFileInput, LogParser
from src.models.analysis_response import AnalysisType
from src.services.analysis_service import AnalysisService, AnalysisServiceError
from src.services.exceptions import EmbeddingServiceError, PineconeServiceError
from src.ui.components import (
    render_analysis_response,
    render_chat_controls,
    render_placeholder,
    render_processing_summary,
    render_sidebar,
    render_status_banner,
)
from src.ui.state import ProcessingSummary, UploadedLogFile, get_app_services, initialize_session_state


def _set_page_config(app_title: str) -> None:
    st.set_page_config(page_title=app_title, page_icon="📄", layout="wide")


def _configure_styles() -> None:
    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.2rem; padding-bottom: 1.5rem; }
            .stMetric { background: rgba(255,255,255,0.03); border-radius: 12px; padding: 0.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _process_uploaded_logs(session_state: object, uploaded_files: list[UploadedLogFile]) -> None:
    services = get_app_services(session_state)
    parser: LogParser = services["parser"]  # type: ignore[assignment]
    chunker: LogChunker = services["chunker"]  # type: ignore[assignment]
    indexing_pipeline = services["indexing_pipeline"]

    if not uploaded_files:
        session_state["processing_error"] = "No files uploaded."
        session_state["logs_indexed"] = False
        summary = ProcessingSummary(status_message="No files uploaded.")
        session_state["processing_summary"] = summary
        return

    start_time = perf_counter()
    log_inputs = [LogFileInput(source_file=file.name, content=file.content) for file in uploaded_files]
    parsed_events = parser.parse_sources(log_inputs)
    chunks = chunker.chunk_events(parsed_events)

    try:
        indexing_result = indexing_pipeline.index_chunks(chunks)
    except (EmbeddingServiceError, PineconeServiceError) as exc:
        session_state["processing_error"] = str(exc)
        session_state["logs_indexed"] = False
        session_state["processed_chunks"] = chunks
        summary = ProcessingSummary(
            files_uploaded=len(uploaded_files),
            events_parsed=len(parsed_events),
            chunks_created=len(chunks),
            vectors_stored=0,
            processing_time_seconds=perf_counter() - start_time,
            last_run_at=None,
            status_message="Indexing failed.",
        )
        session_state["processing_summary"] = summary
        return

    summary = ProcessingSummary(
        files_uploaded=len(uploaded_files),
        events_parsed=len(parsed_events),
        chunks_created=len(chunks),
        vectors_stored=indexing_result.indexed_chunks,
        processing_time_seconds=perf_counter() - start_time,
        last_run_at=None,
        status_message="Logs indexed successfully.",
    )
    session_state["uploaded_log_files"] = uploaded_files
    session_state["processed_chunks"] = chunks
    session_state["processing_summary"] = summary
    session_state["logs_indexed"] = True
    session_state["processing_error"] = ""


def _analysis_type_from_label(label: str) -> AnalysisType:
    mapping = {
        AnalysisType.GENERAL_CHAT.value: AnalysisType.GENERAL_CHAT,
        AnalysisType.ROOT_CAUSE_ANALYSIS.value: AnalysisType.ROOT_CAUSE_ANALYSIS,
        AnalysisType.TIMELINE_EXPLANATION.value: AnalysisType.TIMELINE_EXPLANATION,
        AnalysisType.ERROR_EXPLANATION.value: AnalysisType.ERROR_EXPLANATION,
        AnalysisType.INCIDENT_SUMMARY.value: AnalysisType.INCIDENT_SUMMARY,
    }
    return mapping.get(label, AnalysisType.GENERAL_CHAT)


def main() -> None:
    settings = get_settings()
    _set_page_config(settings.app_title)
    _configure_styles()
    initialize_session_state(st.session_state)
    services = get_app_services(st.session_state)

    st.title(settings.app_title)
    st.caption("Upload log files, index them, and ask questions using the existing backend.")

    uploaded_files = render_sidebar(
        app_title=settings.app_title,
        uploaded_files=st.session_state["uploaded_log_files"],
        processing_summary=st.session_state["processing_summary"],
    )

    process_clicked = st.sidebar.button("Process Logs", type="primary", use_container_width=True)
    if process_clicked:
        _process_uploaded_logs(st.session_state, uploaded_files)
        if st.session_state["processing_error"]:
            st.sidebar.error(st.session_state["processing_error"])
        else:
            st.sidebar.success("Logs processed and indexed.")

    render_status_banner(st.session_state["logs_indexed"])
    render_processing_summary(st.session_state["processing_summary"])

    if not uploaded_files:
        render_placeholder()
        return

    if not st.session_state["logs_indexed"]:
        st.warning("Upload logs and click Process Logs to index them before starting chat.")
        return

    user_question = st.text_input("Ask a question about the logs")
    analysis_type_label, top_k, run_analysis = render_chat_controls(st.session_state["analysis_type"])
    st.session_state["analysis_type"] = analysis_type_label
    st.session_state["retrieval_top_k"] = top_k

    if run_analysis:
        if not user_question.strip():
            st.warning("Please enter a question before analyzing the logs.")
            return

        analysis_service: AnalysisService = services["analysis_service"]  # type: ignore[assignment]
        try:
            response = analysis_service.analyze(
                user_question=user_question,
                analysis_type=_analysis_type_from_label(analysis_type_label),
                top_k=top_k,
            )
        except AnalysisServiceError as exc:
            error_message = str(exc)
            if "GEMINI_API_KEY" in error_message:
                st.error("Gemini is unavailable. Add GEMINI_API_KEY to your environment and try again.")
            else:
                st.error(error_message)
            return

        st.session_state["analysis_history"].append({
            "question": user_question,
            "response": response,
        })
        if not response.evidence_used:
            st.warning("No relevant logs found for this question.")
        render_analysis_response(response)
