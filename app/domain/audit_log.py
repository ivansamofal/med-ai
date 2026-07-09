"""Domain model for an audit-log entry: who did what to a recommendation and
why. Every AI-authored write in this system should end up traceable back to
an actor and a reason — this is a medical system.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class AuditLogEntry(BaseModel):
    recommendation_id: str
    action: str
    actor: str
    reason: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
