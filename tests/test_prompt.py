from datetime import datetime, timezone

import pytest

from app.domain.lab_result import LabResult, ReferenceRange
from app.knowledge.query_engine import RetrievedPassage
from app.recommendations.prompt import build_recommendation_prompt


@pytest.fixture
def abnormal_glucose_result() -> LabResult:
    return LabResult(
        patient_id="p-123",
        external_order_id="ord-456",
        test_code="GLU",
        test_name="Glucose",
        value=250.0,
        unit="mg/dL",
        reference_range=ReferenceRange(low=70, high=99),
        is_abnormal=True,
        collected_at=datetime.now(timezone.utc),
        resulted_at=datetime.now(timezone.utc),
        source_lab="Quest Diagnostics",
        raw_payload={},
    )


@pytest.fixture
def passages() -> list[RetrievedPassage]:
    return [
        RetrievedPassage(
            text="Severe hyperglycemia, generally above 400 mg/dL, warrants urgent evaluation.",
            source_title="Diabetes Guideline",
            source_type="guideline",
            score=0.9,
        ),
        RetrievedPassage(
            text="Glucose reference range is 70-99 mg/dL.",
            source_title="Glucose reference range",
            source_type="reference_range",
            score=0.8,
        ),
    ]


def test_prompt_includes_lab_result_details(abnormal_glucose_result, passages):
    prompt = build_recommendation_prompt(abnormal_glucose_result, passages)

    assert isinstance(prompt, str)
    assert "Glucose" in prompt
    assert "250" in prompt
    assert "mg/dL" in prompt


def test_prompt_includes_passage_citations_and_text(abnormal_glucose_result, passages):
    prompt = build_recommendation_prompt(abnormal_glucose_result, passages)

    for passage in passages:
        assert passage.source_title in prompt
        assert passage.text in prompt


def test_prompt_handles_no_passages(abnormal_glucose_result):
    prompt = build_recommendation_prompt(abnormal_glucose_result, [])

    assert isinstance(prompt, str)
    assert prompt.strip() != ""
