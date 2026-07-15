"""Reusable prompt builders for log analysis workflows.

This module only constructs prompt strings. It does not call Gemini or parse
responses.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from src.models import RetrievalResult


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """Optional conversation history entry reserved for future support."""

    role: str
    content: str


class PromptManager:
    """Build prompt strings for the supported log analysis tasks."""

    def build_general_chat_prompt(
        self,
        user_question: str,
        retrieved_chunks: Sequence[RetrievalResult],
        conversation_history: Sequence[ConversationTurn] | None = None,
    ) -> str:
        """Build a prompt for general chat over retrieved log evidence."""

        return self._build_prompt(
            task_name="General Chat over Logs",
            user_question=user_question,
            retrieved_chunks=retrieved_chunks,
            conversation_history=conversation_history,
            task_instructions=(
                "Answer the user's question using the retrieved log evidence. "
                "If the evidence is insufficient, say so clearly and state what is missing."
            ),
        )

    def build_root_cause_prompt(
        self,
        user_question: str,
        retrieved_chunks: Sequence[RetrievalResult],
        conversation_history: Sequence[ConversationTurn] | None = None,
    ) -> str:
        """Build a prompt for root cause analysis over log evidence."""

        return self._build_prompt(
            task_name="Root Cause Analysis",
            user_question=user_question,
            retrieved_chunks=retrieved_chunks,
            conversation_history=conversation_history,
            task_instructions=(
                "Analyze the retrieved evidence to explain the most likely cause of the incident. "
                "Make clear what is observed versus what is inferred, and do not present unsupported guesses as facts."
            ),
        )

    def build_timeline_prompt(
        self,
        user_question: str,
        retrieved_chunks: Sequence[RetrievalResult],
        conversation_history: Sequence[ConversationTurn] | None = None,
    ) -> str:
        """Build a prompt for explaining the incident timeline."""

        return self._build_prompt(
            task_name="Timeline Explanation",
            user_question=user_question,
            retrieved_chunks=retrieved_chunks,
            conversation_history=conversation_history,
            task_instructions=(
                "Reconstruct the event sequence from the retrieved logs in chronological order when possible. "
                "Highlight the most relevant timestamps, transitions, and dependencies between events."
            ),
        )

    def build_error_explanation_prompt(
        self,
        user_question: str,
        retrieved_chunks: Sequence[RetrievalResult],
        conversation_history: Sequence[ConversationTurn] | None = None,
    ) -> str:
        """Build a prompt for explaining an error or exception."""

        return self._build_prompt(
            task_name="Error Explanation",
            user_question=user_question,
            retrieved_chunks=retrieved_chunks,
            conversation_history=conversation_history,
            task_instructions=(
                "Explain the error in simple technical language. "
                "Describe what the error means, what evidence supports it, and what the logs do not show."
            ),
        )

    def build_incident_summary_prompt(
        self,
        user_question: str,
        retrieved_chunks: Sequence[RetrievalResult],
        conversation_history: Sequence[ConversationTurn] | None = None,
    ) -> str:
        """Build a prompt for summarizing an incident from logs."""

        return self._build_prompt(
            task_name="Incident Summary",
            user_question=user_question,
            retrieved_chunks=retrieved_chunks,
            conversation_history=conversation_history,
            task_instructions=(
                "Summarize the incident concisely, focusing on the most important events, failures, and outcomes. "
                "Prefer a short technical summary that remains grounded in the retrieved logs."
            ),
        )

    def _build_prompt(
        self,
        task_name: str,
        user_question: str,
        retrieved_chunks: Sequence[RetrievalResult],
        conversation_history: Sequence[ConversationTurn] | None,
        task_instructions: str,
    ) -> str:
        sanitized_question = user_question.strip()
        evidence_block = self._format_evidence(retrieved_chunks)
        history_block = self._format_history(conversation_history)

        sections = [
            f"Task: {task_name}",
            "Instructions:",
            self._shared_instructions(),
            task_instructions,
            "User Question:",
            sanitized_question if sanitized_question else "No user question provided.",
        ]

        if history_block:
            sections.extend(["Conversation History:", history_block])

        sections.extend(["Retrieved Log Evidence:", evidence_block])
        sections.extend(
            [
                "Response Guidance:",
                "Use only the retrieved log evidence.",
                "If the evidence is insufficient, say that clearly.",
                "Mention timestamps, source files, and components whenever they are available.",
                "Keep the response concise but informative.",
            ]
        )

        return "\n\n".join(section for section in sections if section).strip()

    def _shared_instructions(self) -> str:
        return (
            "Answer only using the retrieved log evidence. "
            "Never invent information that is not supported by the logs. "
            "Explain reasoning in simple technical language. "
            "If there is insufficient evidence, state that clearly instead of guessing."
        )

    def _format_history(self, conversation_history: Sequence[ConversationTurn] | None) -> str:
        if not conversation_history:
            return ""

        formatted_turns: list[str] = []
        for index, turn in enumerate(conversation_history, start=1):
            role = turn.role.strip() or "unknown"
            content = turn.content.strip() or "(empty)"
            formatted_turns.append(f"{index}. {role}: {content}")

        return "\n".join(formatted_turns)

    def _format_evidence(self, retrieved_chunks: Sequence[RetrievalResult]) -> str:
        if not retrieved_chunks:
            return "No retrieved log evidence was available."

        formatted_chunks: list[str] = []
        for index, chunk in enumerate(retrieved_chunks, start=1):
            formatted_chunks.append(self._format_chunk(index, chunk))

        return "\n\n".join(formatted_chunks)

    def _format_chunk(self, index: int, chunk: RetrievalResult) -> str:
        timestamp_text = self._format_timestamp(chunk.timestamp)
        similarity_text = self._format_similarity(chunk.similarity_score)
        details = [
            f"Evidence {index}:",
            f"Chunk ID: {chunk.chunk_id}",
            f"Source File: {chunk.source_file or 'Unknown'}",
            f"Timestamp: {timestamp_text}",
            f"Level: {chunk.level or 'Unknown'}",
            f"Component: {chunk.component or 'Unknown'}",
            f"Thread: {chunk.thread or 'Unknown'}",
            f"Event Index: {chunk.event_index if chunk.event_index is not None else 'Unknown'}",
            f"Similarity Score: {similarity_text}",
            "Retrieved Text:",
            chunk.retrieved_text.strip() or "(empty)",
        ]
        return "\n".join(details)

    def _format_timestamp(self, timestamp: datetime | None) -> str:
        if timestamp is None:
            return "Unknown"
        if timestamp.tzinfo is None:
            return timestamp.isoformat()
        return timestamp.astimezone().isoformat()

    def _format_similarity(self, score: float | None) -> str:
        if score is None:
            return "Unknown"
        return f"{score:.4f}"
