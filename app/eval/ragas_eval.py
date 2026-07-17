"""Ragas-based retrieval evaluation, run over the same golden Q&A set as
`app.eval.qa_eval` — using Ragas's real metrics instead of this project's
hand-rolled citation+keyword proxy.

Deliberately uses Ragas's **non-LLM** context metrics
(`NonLLMContextPrecisionWithReference`, `NonLLMContextRecall`): they score
retrieved passages against reference text via string-distance, not an LLM
judge, so this runs against the real vector store with zero LLM/cloud
credentials — consistent with `make eval`'s "real embeddings, no required
key" split. Ragas's modern `ragas.metrics.collections` API replaced these
with LLM-only versions (`ContextPrecisionWithReference`/`ContextRecall` now
require an `llm` constructor arg), so this intentionally uses the
deprecated-but-still-functional `ragas.metrics` re-export instead — the
non-LLM behavior has no modern equivalent yet. `Faithfulness`/
`AnswerRelevancy` (LLM-based, judging *generated* text) aren't wired in here:
there's no free-form generation step to judge in this project yet (recommendation
generation is scoped to one lab result, not general Q&A) — see the README for
how to point these at `LLM_BACKEND=bedrock` if that changes.

`reference_contexts` for each question is `GoldenQuestion.reference_context`
— the actual source sentence(s) the question is drawn from (see
`app.eval.models.GoldenQuestion`) — falling back to `required_keywords`
joined for any question that doesn't have one. The fallback is deliberately
weak: whole-string similarity measures (Levenshtein/Jaro) don't separate
"contains the right fact" from "unrelated" when comparing a few keywords
against a full passage, so questions without a real `reference_context`
will score close to 0 on both metrics regardless of retrieval quality.
"""

from __future__ import annotations

import asyncio
import warnings
from dataclasses import dataclass

from llama_index.core import VectorStoreIndex

from app.eval.models import GoldenQuestion
from app.eval.qa_eval import load_golden_questions
from app.knowledge.query_engine import query_knowledge_base

with warnings.catch_warnings():
    # NonLLMContextPrecisionWithReference/NonLLMContextRecall are re-exported
    # from ragas.metrics via a deprecation shim (see module docstring) — the
    # warning is expected, not a bug to fix.
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas.metrics import NonLLMContextPrecisionWithReference, NonLLMContextRecall

from ragas.dataset_schema import SingleTurnSample


@dataclass
class RagasQuestionResult:
    id: str
    context_precision: float
    context_recall: float


@dataclass
class RagasSummary:
    results: list[RagasQuestionResult]

    @property
    def mean_context_precision(self) -> float:
        return sum(r.context_precision for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def mean_context_recall(self) -> float:
        return sum(r.context_recall for r in self.results) / len(self.results) if self.results else 0.0


async def _score_one(question: GoldenQuestion, index: VectorStoreIndex | None, top_k: int) -> RagasQuestionResult:
    passages = query_knowledge_base(
        question.question, top_k=top_k, source_types=question.source_types, index=index
    )
    reference = question.reference_context or " ".join(question.required_keywords)
    sample = SingleTurnSample(
        retrieved_contexts=[p.text for p in passages],
        reference_contexts=[reference],
    )

    precision_metric = NonLLMContextPrecisionWithReference()
    recall_metric = NonLLMContextRecall()
    precision = await precision_metric.single_turn_ascore(sample)
    recall = await recall_metric.single_turn_ascore(sample)

    return RagasQuestionResult(id=question.id, context_precision=precision, context_recall=recall)


async def _evaluate_with_ragas_async(
    questions: list[GoldenQuestion], index: VectorStoreIndex | None, top_k: int
) -> RagasSummary:
    results = [await _score_one(q, index, top_k) for q in questions]
    return RagasSummary(results=results)


def evaluate_with_ragas(
    questions: list[GoldenQuestion] | None = None, index: VectorStoreIndex | None = None, top_k: int = 3
) -> RagasSummary:
    """Sync entry point (Ragas' scorers are async) — `index` defaults to the
    real ingested knowledge base, same as `evaluate_qa`."""
    if questions is None:
        questions = load_golden_questions()
    if index is None:
        from app.knowledge.ingest import get_index

        index = get_index()
    return asyncio.run(_evaluate_with_ragas_async(questions, index, top_k))
