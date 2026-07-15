"""AI analysis output models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    """Single evidence item cited by an AI analysis response."""

    source_file: str = ""
    chunk_id: str = ""
    excerpt: str = ""
    timestamp: datetime | None = None
    relevance_score: float | None = None


class AIAnalysisOutput(BaseModel):
    """Structured response produced by future AI analysis workflows."""

    analysis_type: str = ""
    question: str = ""
    answer: str = ""
    summary: str = ""
    confidence: float | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)