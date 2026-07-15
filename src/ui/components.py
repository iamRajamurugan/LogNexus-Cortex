"""Reusable Streamlit UI components for the log intelligence app."""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from src.models.analysis_response import AIAnalysisResponse
from src.models.analysis_response import AnalysisType
from src.ui.state import ProcessingSummary, UploadedLogFile


SUPPORTED_EXTENSIONS = ["log", "txt", "json"]


def render_sidebar(
    app_title: str,
    uploaded_files: Sequence[UploadedLogFile],
    processing_summary: ProcessingSummary,
) -> list[UploadedLogFile]:
    """Render the sidebar and return the current upload selection."""

    st.sidebar.title(app_title)
    st.sidebar.caption("Upload logs, process them, and analyze the indexed results.")

    uploaded = st.sidebar.file_uploader(
        "Upload one or more log files",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True,
        help="Supported formats: .log, .txt, .json",
    )

    st.sidebar.subheader("Processing Status")
    st.sidebar.metric("Files Uploaded", processing_summary.files_uploaded)
    st.sidebar.metric("Events Parsed", processing_summary.events_parsed)
    st.sidebar.metric("Chunks Created", processing_summary.chunks_created)
    st.sidebar.metric("Vectors Stored", processing_summary.vectors_stored)
    st.sidebar.metric("Processing Time", f"{processing_summary.processing_time_seconds:.2f}s")
    st.sidebar.caption(processing_summary.status_message)

    if not uploaded:
        return list(uploaded_files)

    return [UploadedLogFile(name=file.name, content=file.getvalue().decode("utf-8", errors="replace")) for file in uploaded]


def render_chat_controls(default_analysis_type: str) -> tuple[str, int, bool]:
    """Render analysis controls and return the chosen settings."""

    st.subheader("Analysis")
    analysis_type = st.selectbox(
        "Analysis Type",
        options=[
            AnalysisType.GENERAL_CHAT.value,
            AnalysisType.ROOT_CAUSE_ANALYSIS.value,
            AnalysisType.TIMELINE_EXPLANATION.value,
            AnalysisType.ERROR_EXPLANATION.value,
            AnalysisType.INCIDENT_SUMMARY.value,
        ],
        index=[
            AnalysisType.GENERAL_CHAT.value,
            AnalysisType.ROOT_CAUSE_ANALYSIS.value,
            AnalysisType.TIMELINE_EXPLANATION.value,
            AnalysisType.ERROR_EXPLANATION.value,
            AnalysisType.INCIDENT_SUMMARY.value,
        ].index(default_analysis_type)
        if default_analysis_type in {
            AnalysisType.GENERAL_CHAT.value,
            AnalysisType.ROOT_CAUSE_ANALYSIS.value,
            AnalysisType.TIMELINE_EXPLANATION.value,
            AnalysisType.ERROR_EXPLANATION.value,
            AnalysisType.INCIDENT_SUMMARY.value,
        }
        else 0,
    )
    top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=10, value=5)
    run_analysis = st.button("Analyze", type="primary", use_container_width=True)
    return analysis_type, top_k, run_analysis


def render_status_banner(logs_indexed: bool) -> None:
    """Display a status banner for indexed logs."""

    if logs_indexed:
        st.success("Logs indexed successfully.")


def render_analysis_response(response: AIAnalysisResponse) -> None:
    """Render the structured AI response returned by the analysis engine."""

    st.markdown("### AI Response")
    st.write(response.answer)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Analysis Type", response.analysis_type.value)
    col2.metric("Model Used", response.model_name or "Unknown")
    col3.metric("Retrieved Chunks", response.retrieved_chunks_count)
    col4.metric("Generation Time", response.generated_at.strftime("%Y-%m-%d %H:%M:%S"))

    if response.evidence_used:
        with st.expander("Evidence Used", expanded=False):
            for index, item in enumerate(response.evidence_used, start=1):
                st.markdown(
                    f"""
                    **Evidence {index}**

                    - Chunk ID: {item.chunk_id}
                    - Source File: {item.source_file or 'Unknown'}
                    - Timestamp: {item.timestamp.isoformat() if item.timestamp else 'Unknown'}
                    - Level: {item.level or 'Unknown'}
                    - Component: {item.component or 'Unknown'}
                    - Thread: {item.thread or 'Unknown'}
                    - Event Index: {item.event_index if item.event_index is not None else 'Unknown'}

                    ```text
                    {item.retrieved_text}
                    ```
                    """
                )


def render_processing_summary(processing_summary: ProcessingSummary) -> None:
    """Render the current processing summary in the main content area."""

    st.markdown("### Processing Summary")
    col1, col2 = st.columns(2)
    col1.metric("Files Uploaded", processing_summary.files_uploaded)
    col2.metric("Events Parsed", processing_summary.events_parsed)
    col3, col4 = st.columns(2)
    col3.metric("Chunks Created", processing_summary.chunks_created)
    col4.metric("Vectors Stored", processing_summary.vectors_stored)
    st.caption(f"Processing time: {processing_summary.processing_time_seconds:.2f}s")


def render_placeholder() -> None:
    """Render the idle-state helper text."""

    st.info("Upload log files in the sidebar, process them, and then ask a question about the logs.")
