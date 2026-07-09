import pytest
from langchain_core.exceptions import OutputParserException

from app.recommendations.parser import ParsedRecommendation, get_output_parser


def test_format_instructions_mention_both_fields():
    instructions = get_output_parser().get_format_instructions()

    assert "text" in instructions
    assert "citations" in instructions


def test_parses_well_formed_json():
    raw = '{"text": "Glucose is within the normal range; no action needed.", "citations": ["Diabetes Guideline", "Glucose reference range"]}'

    parsed = get_output_parser().invoke(raw)

    assert isinstance(parsed, ParsedRecommendation)
    assert parsed.text == "Glucose is within the normal range; no action needed."
    assert parsed.citations == ["Diabetes Guideline", "Glucose reference range"]


def test_parses_json_with_no_citations():
    raw = '{"text": "No concerns at this time.", "citations": []}'

    parsed = get_output_parser().invoke(raw)

    assert parsed.text == "No concerns at this time."
    assert parsed.citations == []


def test_tolerates_markdown_code_fence():
    raw = '```json\n{"text": "Check again in three months.", "citations": ["Some Source"]}\n```'

    parsed = get_output_parser().invoke(raw)

    assert parsed.text == "Check again in three months."
    assert parsed.citations == ["Some Source"]


def test_raises_on_unparseable_response():
    raw = "The model just replied with free text, no JSON at all."

    with pytest.raises(OutputParserException):
        get_output_parser().invoke(raw)
