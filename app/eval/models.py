"""Shared result/golden-case shapes for the evaluation harness (Phase 6).

Kept separate from `app.domain` models: these describe an eval *case* and its
*outcome*, not a persisted system entity.
"""

from __future__ import annotations

from pydantic import BaseModel


class GoldenQuestion(BaseModel):
    """One golden knowledge-base Q&A case: a question plus the citation and
    grounding-text checks a correct retrieval must satisfy.

    `reference_context` (optional) is the actual source sentence/document text
    the question is drawn from — used as Ragas's ground-truth `reference_contexts`
    (app.eval.ragas_eval), which needs real comparable-length passage text rather
    than the short `required_keywords`: whole-string similarity measures don't
    separate "contains the right fact" from "unrelated" when comparing a few
    words against a full sentence. Falls back to `required_keywords` joined
    when absent, for older/hand-added questions."""

    id: str
    question: str
    source_types: list[str] | None = None
    expected_source_titles: list[str]
    required_keywords: list[str]
    reference_context: str | None = None


class GoldenRedFlagCase(BaseModel):
    """One golden red-flag case: a lab value plus whether it must escalate."""

    id: str
    test_code: str
    test_name: str
    value: float
    unit: str
    expect_red_flag: bool


class EvalResult(BaseModel):
    """Outcome of checking one golden case, suite-agnostic."""

    id: str
    passed: bool
    detail: str
