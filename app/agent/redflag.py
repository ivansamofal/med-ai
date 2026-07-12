"""TODO (Phase 5, yours to implement): the red-flag escalation rule.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.domain.lab_result import LabResult
from app.knowledge.reference_ranges import CriticalThreshold, load_critical_thresholds


class RedFlag(BaseModel):
    test_code: str
    test_name: str
    value: float
    unit: str
    reason: str


def detect_red_flag(
    lab_result: LabResult,
    thresholds: dict[str, CriticalThreshold] | None = None,
) -> RedFlag | None:
    """Decide whether a single lab result is dangerous enough to escalate to
    an on-call clinician instead of letting the chat agent answer directly.

    WHAT: Compare `lab_result.value` against the `critical_low`/`critical_high`
    thresholds for its `test_code` (loaded from the reference-range CSV via
    `load_critical_thresholds`, injectable via `thresholds` for tests) and
    return a `RedFlag` if it's beyond either bound, `None` otherwise.

    WHY: `lab_result.is_abnormal` (Phase 1) already flags anything outside
    the *normal* range — that's most of what the recommendation chain (Phase
    3) drafts guidance for. A red flag is a narrower, more severe subset:
    values dangerous enough that a chatbot (or a clinician not yet paying
    attention) answering normally would be a safety failure. Conflating the
    two would escalate constantly (every mildly-high glucose reading) and
    dilute the signal for genuinely urgent values; only critical thresholds
    should trigger escalation here.

    Hints:
    - Not every test code has both bounds (see the CSV — `EGFR` has no
      `critical_high`, `HBA1C`/`TSH`/`LDL`/`HDL` have no critical thresholds
      at all yet). Missing data means "can't tell it's critical", not "treat
      it as one" — return `None` rather than guessing.
    - `<=`/`>=` at the boundary, not `<`/`>` — a value exactly at
      `critical_low`/`critical_high` is still critical.
    """
    resolved_thresholds = thresholds if thresholds is not None else load_critical_thresholds()
    threshold = resolved_thresholds.get(lab_result.test_code)
    if threshold is None:
        return None

    if threshold.critical_low is not None and lab_result.value <= threshold.critical_low:
        reason = (
            f"{lab_result.test_name} of {lab_result.value} {lab_result.unit} is at or "
            f"below the critical low threshold of {threshold.critical_low} {lab_result.unit}."
        )
    elif threshold.critical_high is not None and lab_result.value >= threshold.critical_high:
        reason = (
            f"{lab_result.test_name} of {lab_result.value} {lab_result.unit} is at or "
            f"above the critical high threshold of {threshold.critical_high} {lab_result.unit}."
        )
    else:
        return None

    return RedFlag(
        test_code=lab_result.test_code,
        test_name=lab_result.test_name,
        value=lab_result.value,
        unit=lab_result.unit,
        reason=reason,
    )
