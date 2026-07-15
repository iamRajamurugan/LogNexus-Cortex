"""Structured AI analysis response models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from src.models import RetrievalResult


class AnalysisType(str, Enum):
    """Supported analysis modes for the AI analysis engine."""

    GENERAL_CHAT = "General Chat"
    ROOT_CAUSE_ANALYSIS = "Root Cause Analysis"
    TIMELINE_EXPLANATION = "Timeline Explanation"
    ERROR_EXPLANATION = "Error Explanation"
    INCIDENT_SUMMARY = "Incident Summary"


class AIAnalysisResponse(BaseModel):
    """Structured response returned by the analysis engine."""

    answer: str
    analysis_type: AnalysisType
    evidence_used: list[RetrievalResult] = Field(default_factory=list)
    retrieved_chunks_count: int = 0
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    model_name: str = ""
