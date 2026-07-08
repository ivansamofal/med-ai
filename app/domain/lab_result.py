"""Canonical lab-result domain model and normalization from the external lab API shape.

This is the direct rewrite of the old Symfony normalization step: the external API
returns loosely-typed, lab-vendor-specific fields; we convert them into one strict
internal schema before anything else in the system (storage, retrieval, LLM prompts)
touches a lab result.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class ReferenceRange(BaseModel):
    low: float | None = None
    high: float | None = None


class LabResult(BaseModel):
    """Normalized, internal representation of a single lab result."""

    patient_id: str
    external_order_id: str
    test_code: str
    test_name: str
    value: float
    unit: str
    reference_range: ReferenceRange
    is_abnormal: bool
    collected_at: datetime
    resulted_at: datetime
    source_lab: str
    raw_payload: dict
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def _parse_datetime(raw: str) -> datetime:
    # External API emits ISO-8601 with a trailing "Z"; fromisoformat wants "+00:00".
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _compute_is_abnormal(value: float, ref_range: ReferenceRange, flag: str | None) -> bool:
    if flag:
        return flag.strip().upper() not in ("", "N", "NORMAL")
    if ref_range.low is not None and value < ref_range.low:
        return True
    if ref_range.high is not None and value > ref_range.high:
        return True
    return False


def normalize_lab_result(raw: dict) -> LabResult:
    """Map a raw external lab API payload into the canonical `LabResult` schema.

    Raises `ValueError` (via pydantic) on missing/malformed required fields — the
    caller (API route) is responsible for turning that into an HTTP 422.
    """
    reference_range = ReferenceRange(
        low=float(raw["reference_low"]) if raw.get("reference_low") not in (None, "") else None,
        high=float(raw["reference_high"]) if raw.get("reference_high") not in (None, "") else None,
    )
    value = float(raw["result_value"])

    return LabResult(
        patient_id=str(raw["patient_id"]),
        external_order_id=str(raw["order_id"]),
        test_code=str(raw["test_code"]),
        test_name=str(raw["test_name"]),
        value=value,
        unit=str(raw["unit"]),
        reference_range=reference_range,
        is_abnormal=_compute_is_abnormal(value, reference_range, raw.get("abnormal_flag")),
        collected_at=_parse_datetime(raw["collected_at"]),
        resulted_at=_parse_datetime(raw["resulted_at"]),
        source_lab=str(raw.get("lab_name", "unknown")),
        raw_payload=raw,
    )
