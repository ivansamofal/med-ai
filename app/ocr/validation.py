"""Validates extracted document entities against known reference data — the
"verify attributes / ensure data consistency" half of the OCR pipeline.
Returns a list of issue strings (empty list = clean) rather than raising,
since a document with issues is still worth persisting for review, not
rejecting outright.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.knowledge.reference_ranges import load_critical_thresholds
from app.ocr.entities import ExtractedDocumentEntities


def validate_entities(entities: ExtractedDocumentEntities) -> list[str]:
    issues: list[str] = []

    if not entities.patient_name.strip():
        issues.append("missing patient name")
    if not entities.ordering_physician.strip():
        issues.append("missing ordering physician")
    if not entities.test_codes:
        issues.append("no test codes found")

    known_codes = load_critical_thresholds()
    for code in entities.test_codes:
        if code not in known_codes:
            issues.append(f"unknown test code: {code}")

    parsed_date = _try_parse_date(entities.document_date)
    if parsed_date is None:
        issues.append(f"unparseable document date: {entities.document_date!r}")
    elif parsed_date > datetime.now(timezone.utc):
        issues.append(f"document date is in the future: {entities.document_date}")

    return issues


def _try_parse_date(raw: str) -> datetime | None:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None
