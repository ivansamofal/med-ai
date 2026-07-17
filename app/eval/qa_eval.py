"""Golden Q&A suite: checks that `query_knowledge_base` — the same retrieval
entry point the recommendation chain and chat agent call — actually surfaces
the *right* passage for each question, not just some passage.

Two checks per question, both against the top-k retrieved passages:

- **citation-presence**: at least one retrieved passage's `source_title` is
  one this question's answer should be allowed to cite. Catches retrieval
  pulling in the wrong document entirely.
- **faithfulness (proxy)**: the retrieved text actually contains the fact the
  question is asking about (e.g. the "5.0" INR threshold). A generated answer
  can only be faithful to what got retrieved — if the right number never made
  it into context, no amount of prompting fixes that downstream. Checking
  retrieval directly (rather than an LLM's paraphrase of it) makes this
  suite deterministic and runnable offline.
"""

from __future__ import annotations

import json
from pathlib import Path

from llama_index.core import VectorStoreIndex

from app.eval.models import EvalResult, GoldenQuestion
from app.knowledge.query_engine import RetrievedPassage, query_knowledge_base

DEFAULT_GOLDEN_QA_PATH = Path(__file__).resolve().parents[2] / "data" / "eval" / "golden_qa.json"


def load_golden_questions(path: Path = DEFAULT_GOLDEN_QA_PATH) -> list[GoldenQuestion]:
    return [GoldenQuestion(**row) for row in json.loads(path.read_text())]


def score_passages(question: GoldenQuestion, passages: list[RetrievedPassage]) -> EvalResult:
    """The citation-presence + faithfulness-proxy check, decoupled from *how*
    the passages were retrieved — shared by the plain golden-QA suite and the
    A/B experiment harness (`app.eval.experiment`), which each retrieve
    differently but score identically."""
    retrieved_titles = [p.source_title for p in passages]
    combined_text = " ".join(p.text for p in passages).lower()

    cited = any(title in retrieved_titles for title in question.expected_source_titles)
    missing_keywords = [kw for kw in question.required_keywords if kw.lower() not in combined_text]

    if cited and not missing_keywords:
        return EvalResult(id=question.id, passed=True, detail=f"retrieved: {retrieved_titles}")

    problems = []
    if not cited:
        problems.append(f"expected one of {question.expected_source_titles}, got {retrieved_titles}")
    if missing_keywords:
        problems.append(f"missing keywords {missing_keywords} in retrieved text")
    return EvalResult(id=question.id, passed=False, detail="; ".join(problems))


def _evaluate_one(question: GoldenQuestion, index: VectorStoreIndex | None, top_k: int) -> EvalResult:
    passages = query_knowledge_base(
        question.question, top_k=top_k, source_types=question.source_types, index=index
    )
    return score_passages(question, passages)


def evaluate_qa(
    questions: list[GoldenQuestion], index: VectorStoreIndex | None = None, top_k: int = 3
) -> list[EvalResult]:
    """Run every golden question through real retrieval and report pass/fail.

    `index` defaults to the real ingested knowledge base (`get_index()`,
    imported lazily so tests can inject a small fixture index instead).
    """
    if index is None:
        from app.knowledge.ingest import get_index

        index = get_index()

    return [_evaluate_one(question, index, top_k) for question in questions]
