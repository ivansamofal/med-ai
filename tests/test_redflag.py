from datetime import datetime, timezone

from app.agent.redflag import detect_red_flag
from app.domain.lab_result import LabResult, ReferenceRange
from app.knowledge.reference_ranges import CriticalThreshold

THRESHOLDS = {
    "GLU": CriticalThreshold(test_code="GLU", critical_low=50, critical_high=400),
    "EGFR": CriticalThreshold(test_code="EGFR", critical_low=15, critical_high=None),
    "HBA1C": CriticalThreshold(test_code="HBA1C", critical_low=None, critical_high=None),
}


def _lab_result(test_code: str, test_name: str, value: float, unit: str = "mg/dL") -> LabResult:
    return LabResult(
        patient_id="p-123",
        external_order_id="ord-1",
        test_code=test_code,
        test_name=test_name,
        value=value,
        unit=unit,
        reference_range=ReferenceRange(low=70, high=99),
        is_abnormal=True,
        collected_at=datetime.now(timezone.utc),
        resulted_at=datetime.now(timezone.utc),
        source_lab="Quest Diagnostics",
        raw_payload={},
    )


def test_value_within_critical_bounds_is_not_a_red_flag():
    lab_result = _lab_result("GLU", "Glucose", 250)

    assert detect_red_flag(lab_result, THRESHOLDS) is None


def test_value_above_critical_high_is_a_red_flag():
    lab_result = _lab_result("GLU", "Glucose", 420)

    red_flag = detect_red_flag(lab_result, THRESHOLDS)

    assert red_flag is not None
    assert red_flag.test_code == "GLU"
    assert "420" in red_flag.reason


def test_value_below_critical_low_is_a_red_flag():
    lab_result = _lab_result("GLU", "Glucose", 45)

    red_flag = detect_red_flag(lab_result, THRESHOLDS)

    assert red_flag is not None
    assert "45" in red_flag.reason


def test_value_exactly_at_critical_boundary_is_a_red_flag():
    lab_result = _lab_result("GLU", "Glucose", 400)

    assert detect_red_flag(lab_result, THRESHOLDS) is not None


def test_test_code_with_only_one_critical_bound_ignores_the_missing_side():
    # EGFR has a critical_low but no critical_high — an extremely high value
    # should not be treated as a red flag just because it's far from normal.
    lab_result = _lab_result("EGFR", "Estimated GFR", 500, unit="mL/min/1.73m2")

    assert detect_red_flag(lab_result, THRESHOLDS) is None


def test_test_code_with_no_critical_thresholds_never_flags():
    lab_result = _lab_result("HBA1C", "Hemoglobin A1c", 20, unit="%")

    assert detect_red_flag(lab_result, THRESHOLDS) is None


def test_unknown_test_code_never_flags():
    lab_result = _lab_result("UNKNOWN", "Mystery Test", 9999)

    assert detect_red_flag(lab_result, THRESHOLDS) is None


def test_loads_real_thresholds_from_csv_by_default():
    # No injected thresholds: falls back to the real reference-range CSV.
    lab_result = _lab_result("K", "Potassium", 7.0, unit="mmol/L")

    red_flag = detect_red_flag(lab_result)

    assert red_flag is not None
    assert red_flag.test_code == "K"
