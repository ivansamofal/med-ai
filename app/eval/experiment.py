"""A/B testing harness: run several named variants of *how retrieval (or,
with a real LLM, generation) is done* over the same golden set and compare
pass rates — extends the eval harness rather than building a parallel
system, since the golden Q&A set already is the ground truth every variant
should be judged against. Scoring is `app.eval.qa_eval.score_passages`, the
exact same citation+keyword check the plain golden-QA suite uses, so a
variant's score is directly comparable to `make eval`'s baseline number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.eval.models import EvalResult, GoldenQuestion
from app.eval.qa_eval import score_passages
from app.knowledge.query_engine import RetrievedPassage


@dataclass
class ExperimentVariant:
    name: str
    retrieve_fn: Callable[[GoldenQuestion], list[RetrievedPassage]]


@dataclass
class VariantReport:
    name: str
    results: list[EvalResult]

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)


def run_experiment(
    variants: list[ExperimentVariant], questions: list[GoldenQuestion]
) -> list[VariantReport]:
    """Run every variant over every question and return one report per
    variant, in the order `variants` was given (the first variant is treated
    as the baseline by callers that print a delta)."""
    return [
        VariantReport(
            name=variant.name,
            results=[score_passages(q, variant.retrieve_fn(q)) for q in questions],
        )
        for variant in variants
    ]
