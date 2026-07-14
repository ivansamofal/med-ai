"""Domain model for an OCR'd business document and its extracted entities —
the OCR pipeline's equivalent of `app.domain.recommendation.Recommendation`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

DocumentStatus = Literal["clean", "needs_review"]


class ScannedDocument(BaseModel):
    filename: str
    raw_text: str
    extracted_entities: dict
    validation_issues: list[str]
    status: DocumentStatus
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
