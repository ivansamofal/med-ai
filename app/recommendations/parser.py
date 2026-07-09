"""Structured-output schema and parser for the LLM's response.

Uses LangChain's `PydanticOutputParser` bound into the chain (see
`chain.py`) rather than hand-rolled regex: the parser's `get_format_instructions()`
is embedded in the prompt, so every `LLM` implementation — real or fake — is
told the exact JSON schema to produce, and parsing/validation is one call.
"""

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class ParsedRecommendation(BaseModel):
    text: str = Field(description="Plain-language recommendation text for a clinician to review.")
    citations: list[str] = Field(description="Source titles the recommendation draws on.")


def get_output_parser() -> PydanticOutputParser:
    return PydanticOutputParser(pydantic_object=ParsedRecommendation)
