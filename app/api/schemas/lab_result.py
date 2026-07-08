"""Request/response models for the /lab-results endpoint.

`LabResultIngestRequest` mirrors the shape the external lab API sends us (loosely
typed strings for numeric/date fields, as vendors typically do) — see
`app.domain.lab_result.normalize_lab_result` for the conversion into our internal
canonical schema.
"""

from pydantic import BaseModel, ConfigDict


class LabResultIngestRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    patient_id: str
    order_id: str
    test_code: str
    test_name: str
    result_value: str
    unit: str
    reference_low: str | None = None
    reference_high: str | None = None
    abnormal_flag: str | None = None
    collected_at: str
    resulted_at: str
    lab_name: str | None = None


class LabResultResponse(BaseModel):
    id: str
    patient_id: str
    test_code: str
    test_name: str
    value: float
    unit: str
    is_abnormal: bool
    resulted_at: str
