"""Read-only response models for the dashboard's list endpoints
(GET /recommendations, /appointments, /audit-log — GET /lab-results reuses
`LabResultResponse`)."""

from pydantic import BaseModel


class RecommendationResponse(BaseModel):
    id: str
    lab_result_id: str
    patient_id: str
    recommendation_text: str
    citations: list[str]
    status: str
    created_at: str


class AppointmentResponse(BaseModel):
    id: str
    patient_id: str
    doctor: str
    scheduled_at: str
    duration_minutes: int
    status: str
    created_at: str


class AuditLogResponse(BaseModel):
    id: str
    recommendation_id: str
    action: str
    actor: str
    reason: str | None
    created_at: str
