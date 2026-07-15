"""AI analysis orchestration for log questions.

This module connects the retrieval layer, prompt management layer, and Gemini
chat model. It does not contain parser, chunking, indexing, or retrieval logic.
"""

from __future__ import annotations

import logging
from typing import Callable, Sequence

from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import AppSettings, get_settings
from src.core import RetrievalService
from src.models import RetrievalResult
from src.models.analysis_response import AIAnalysisResponse, AnalysisType
from src.prompts import ConversationTurn, PromptManager
from src.services.exceptions import RetrievalServiceError


logger = logging.getLogger(__name__)


class AnalysisServiceError(RuntimeError):
    """Raised when the analysis engine cannot complete a request."""


class AnalysisService:
    """Orchestrate retrieval, prompt construction, and Gemini response generation."""

    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        prompt_manager: PromptManager | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retrieval_service = retrieval_service or RetrievalService()
        self.prompt_manager = prompt_manager or PromptManager()

    def analyze(
        self,
        user_question: str,
        analysis_type: AnalysisType = AnalysisType.GENERAL_CHAT,
        top_k: int | None = None,
        conversation_history: Sequence[ConversationTurn] | None = None,
    ) -> AIAnalysisResponse:
        """Run the end-to-end analysis workflow for a user question."""

        normalized_question = user_question.strip()
        if not normalized_question:
            raise AnalysisServiceError("User question cannot be empty.")

        retrieved_chunks = self._retrieve_chunks(normalized_question, top_k)
        if not retrieved_chunks:
            return self._build_no_evidence_response(analysis_type, normalized_question)

        prompt = self._build_prompt(
            analysis_type=analysis_type,
            user_question=normalized_question,
            retrieved_chunks=retrieved_chunks,
            conversation_history=conversation_history,
        )
        answer = self._generate_answer(prompt)
        return AIAnalysisResponse(
            answer=answer,
            analysis_type=analysis_type,
            evidence_used=list(retrieved_chunks),
            retrieved_chunks_count=len(retrieved_chunks),
            model_name=self.settings.gemini_model,
        )

    def _retrieve_chunks(self, user_question: str, top_k: int | None) -> list[RetrievalResult]:
        try:
            return self.retrieval_service.retrieve(user_question, top_k=top_k)
        except RetrievalServiceError as exc:
            raise AnalysisServiceError(str(exc)) from exc

    def _build_prompt(
        self,
        analysis_type: AnalysisType,
        user_question: str,
        retrieved_chunks: Sequence[RetrievalResult],
        conversation_history: Sequence[ConversationTurn] | None,
    ) -> str:
        builder = self._resolve_prompt_builder(analysis_type)
        try:
            return builder(user_question, retrieved_chunks, conversation_history)
        except Exception as exc:
            logger.exception("Prompt generation failed for analysis type %s.", analysis_type.value)
            raise AnalysisServiceError("Prompt generation failed.") from exc

    def _resolve_prompt_builder(
        self,
        analysis_type: AnalysisType,
    ) -> Callable[[str, Sequence[RetrievalResult], Sequence[ConversationTurn] | None], str]:
        prompt_builders: dict[AnalysisType, Callable[[str, Sequence[RetrievalResult], Sequence[ConversationTurn] | None], str]] = {
            AnalysisType.GENERAL_CHAT: self.prompt_manager.build_general_chat_prompt,
            AnalysisType.ROOT_CAUSE_ANALYSIS: self.prompt_manager.build_root_cause_prompt,
            AnalysisType.TIMELINE_EXPLANATION: self.prompt_manager.build_timeline_prompt,
            AnalysisType.ERROR_EXPLANATION: self.prompt_manager.build_error_explanation_prompt,
            AnalysisType.INCIDENT_SUMMARY: self.prompt_manager.build_incident_summary_prompt,
        }

        return prompt_builders[analysis_type]

    def _generate_answer(self, prompt: str) -> str:
        if not self.settings.gemini_api_key:
            raise AnalysisServiceError("GEMINI_API_KEY is required to generate analysis responses.")

        try:
            model = ChatGoogleGenerativeAI(
                model=self.settings.gemini_model,
                google_api_key=self.settings.gemini_api_key,
                temperature=0.2,
            )
            response = model.invoke(prompt)
            content = getattr(response, "content", "")
            if isinstance(content, list):
                return "".join(str(item) for item in content).strip()
            return str(content).strip()
        except Exception as exc:
            logger.exception("Gemini analysis generation failed.")
            raise AnalysisServiceError("Gemini API failed while generating the analysis response.") from exc

    def _build_no_evidence_response(
        self,
        analysis_type: AnalysisType,
        user_question: str,
    ) -> AIAnalysisResponse:
        return AIAnalysisResponse(
            answer=(
                "There is insufficient evidence in the uploaded logs to answer this question confidently. "
                "Please upload additional relevant log data or refine the question."
            ),
            analysis_type=analysis_type,
            evidence_used=[],
            retrieved_chunks_count=0,
            model_name=self.settings.gemini_model,
        )
