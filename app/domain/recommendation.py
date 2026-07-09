"""Domain model for an AI-authored recommendation, awaiting clinician review
(Phase 4 adds the approval graph that moves it out of `pending_review`).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

RecommendationStatus = Literal["pending_review", "approved", "edited", "rejected"]


class Recommendation(BaseModel):
    lab_result_id: str
    patient_id: str
    recommendation_text: str
    citations: list[str]
    raw_llm_response: str
    status: RecommendationStatus = "pending_review"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
