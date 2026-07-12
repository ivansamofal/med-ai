"""Domain model for a scheduled clinician appointment, booked/managed by the
chat agent's scheduling tools (Phase 5).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

AppointmentStatus = Literal["scheduled", "cancelled"]


class Appointment(BaseModel):
    patient_id: str
    doctor: str
    scheduled_at: datetime
    duration_minutes: int = 30
    status: AppointmentStatus = "scheduled"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
