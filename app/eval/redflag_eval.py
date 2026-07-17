"""Golden red-flag suite: checks `detect_red_flag` — the same guardrail the
chat graph (`app.agent.graph`) reads to force escalation instead of letting
the agent answer directly — against real critical thresholds from the
reference-range CSV.

This complements (doesn't replace) `tests/test_redflag.py`'s unit tests: the
golden set is a single, reviewable table of "this value must/must not
escalate" that doubles as a pass/fail report in `make eval`, alongside the
boundary and no-threshold-data guard cases the escalation rule has to get
right (critical thresholds are asymmetric per test code, and a missing bound
must never be treated as "automatically critical").
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app.agent.redflag import detect_red_flag
from app.domain.lab_result import LabResult, ReferenceRange
from app.eval.models import EvalResult, GoldenRedFlagCase
from app.knowledge.reference_ranges import CriticalThreshold

DEFAULT_GOLDEN_RED_FLAGS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "eval" / "golden_red_flags.json"
)


def load_golden_red_flags(path: Path = DEFAULT_GOLDEN_RED_FLAGS_PATH) -> list[GoldenRedFlagCase]:
    return [GoldenRedFlagCase(**row) for row in json.loads(path.read_text())]


def _lab_result_for(case: GoldenRedFlagCase) -> LabResult:
    return LabResult(
        patient_id="eval-patient",
        external_order_id="eval-order",
        test_code=case.test_code,
        test_name=case.test_name,
        value=case.value,
        unit=case.unit,
        reference_range=ReferenceRange(),
        is_abnormal=True,
        collected_at=datetime.now(timezone.utc),
        resulted_at=datetime.now(timezone.utc),
        source_lab="eval",
        raw_payload={},
    )


def evaluate_red_flags(
    cases: list[GoldenRedFlagCase], thresholds: dict[str, CriticalThreshold] | None = None
) -> list[EvalResult]:
    """Run every golden case through the real escalation rule and report
    pass/fail. `thresholds=None` loads the real reference-range CSV (the
    production config); tests can inject a fixture map instead."""
    results = []
    for case in cases:
        red_flag = detect_red_flag(_lab_result_for(case), thresholds)
        got_flag = red_flag is not None
        if got_flag == case.expect_red_flag:
            results.append(EvalResult(id=case.id, passed=True, detail=f"red_flag={got_flag}"))
        else:
            results.append(
                EvalResult(
                    id=case.id,
                    passed=False,
                    detail=f"expected red_flag={case.expect_red_flag}, got {got_flag}",
                )
            )
    return results
