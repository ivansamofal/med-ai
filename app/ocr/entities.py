"""Structured entity extraction from OCR'd document text: schema, parser, and
prompt — the OCR-pipeline equivalent of
`app.recommendations.parser`/`app.recommendations.prompt`. Reuses
`get_chat_model()` (Phase 3's LLM interface) instead of a second model
integration; `FAKE_ENTITY_RESPONSE` gives the offline fake backend valid
JSON for this schema specifically, since the default `FAKE_RESPONSE` in
`app.llm.interface` is shaped for the recommendation chain instead.
"""

from __future__ import annotations

import json

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.llm.interface import get_chat_model

_PROMPT = ChatPromptTemplate.from_messages([("human", "{prompt_body}\n\n{format_instructions}")])


class ExtractedDocumentEntities(BaseModel):
    patient_name: str = Field(description="Patient name as written on the document.")
    document_date: str = Field(description="Date on the document, as written (not reformatted).")
    ordering_physician: str = Field(description="Ordering physician's name as written.")
    test_codes: list[str] = Field(description="Lab test codes mentioned on the document (e.g. GLU, HBA1C).")


FAKE_ENTITY_RESPONSE = json.dumps(
    {
        "patient_name": "Jane Doe",
        "document_date": "2026-06-01",
        "ordering_physician": "Dr. Smith",
        "test_codes": ["GLU", "HBA1C"],
    }
)


def get_entity_parser() -> PydanticOutputParser:
    return PydanticOutputParser(pydantic_object=ExtractedDocumentEntities)


def build_entity_extraction_prompt(raw_text: str) -> str:
    return (
        "The following text was OCR'd from a scanned business document (a lab "
        "requisition form). Extract the patient name, the document date, the "
        "ordering physician, and the list of lab test codes mentioned — exactly "
        "as written, with no inference or outside knowledge. If a field isn't "
        "present, use an empty string (or empty list for test codes).\n\n"
        f"OCR'd text:\n{raw_text}"
    )


async def extract_entities(raw_text: str) -> ExtractedDocumentEntities:
    parser = get_entity_parser()
    chain = _PROMPT | get_chat_model(fake_response=FAKE_ENTITY_RESPONSE) | parser
    return await chain.ainvoke(
        {
            "prompt_body": build_entity_extraction_prompt(raw_text),
            "format_instructions": parser.get_format_instructions(),
        }
    )
