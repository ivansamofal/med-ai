import pytest

from app.domain.lab_result import normalize_lab_result


def test_normalize_maps_fields_and_types(sample_raw_lab_result):
    lab_result = normalize_lab_result(sample_raw_lab_result)

    assert lab_result.patient_id == "p-123"
    assert lab_result.external_order_id == "ord-456"
    assert lab_result.value == 105.0
    assert lab_result.reference_range.low == 70.0
    assert lab_result.reference_range.high == 99.0
    assert lab_result.source_lab == "Quest Diagnostics"
    assert lab_result.raw_payload == sample_raw_lab_result


def test_normalize_uses_abnormal_flag_when_present(sample_raw_lab_result):
    lab_result = normalize_lab_result(sample_raw_lab_result)
    assert lab_result.is_abnormal is True


def test_normalize_falls_back_to_reference_range_when_no_flag(sample_raw_lab_result):
    sample_raw_lab_result["abnormal_flag"] = None
    sample_raw_lab_result["result_value"] = "50"  # below reference_low=70

    lab_result = normalize_lab_result(sample_raw_lab_result)
    assert lab_result.is_abnormal is True


def test_normalize_is_not_abnormal_within_range(sample_raw_lab_result):
    sample_raw_lab_result["abnormal_flag"] = None
    sample_raw_lab_result["result_value"] = "85"  # within [70, 99]

    lab_result = normalize_lab_result(sample_raw_lab_result)
    assert lab_result.is_abnormal is False


def test_normalize_raises_on_missing_required_field(sample_raw_lab_result):
    del sample_raw_lab_result["result_value"]

    with pytest.raises(KeyError):
        normalize_lab_result(sample_raw_lab_result)
