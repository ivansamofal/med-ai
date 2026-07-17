"""`make ab-eval` entry point: compares retrieval `top_k=3` vs `top_k=5`
against the real, already-ingested knowledge base (`make ingest` first) —
one concrete, always-meaningful A/B comparison, since it only needs real
embeddings, not a real LLM call (unlike a prompt-variant comparison, where
every variant returns identical text under the default fake LLM backend —
see the README for why that comparison needs `LLM_BACKEND=bedrock` to be
worth running).

Informational, not a gate: prints a report and always exits 0 — an A/B
result is a data point for a decision, not a pass/fail check like `make eval`.
"""

from __future__ import annotations

from app.eval.experiment import ExperimentVariant, VariantReport, run_experiment
from app.eval.qa_eval import load_golden_questions
from app.knowledge.query_engine import query_knowledge_base


def _print_report(reports: list[VariantReport]) -> None:
    baseline = reports[0]
    for report in reports:
        print(f"\n{report.name} ({len(report.results)} cases)")
        print("-" * 60)
        for result in report.results:
            status = "PASS" if result.passed else "FAIL"
            print(f"[{status}] {result.id}: {result.detail}")
        passed = sum(1 for r in report.results if r.passed)
        delta = "" if report is baseline else f" ({report.pass_rate - baseline.pass_rate:+.0%} vs {baseline.name})"
        print(f"{passed}/{len(report.results)} passed — pass rate {report.pass_rate:.0%}{delta}")


def main() -> int:
    questions = load_golden_questions()
    variants = [
        ExperimentVariant(
            name="top_k=3",
            retrieve_fn=lambda q: query_knowledge_base(q.question, top_k=3, source_types=q.source_types),
        ),
        ExperimentVariant(
            name="top_k=5",
            retrieve_fn=lambda q: query_knowledge_base(q.question, top_k=5, source_types=q.source_types),
        ),
    ]

    reports = run_experiment(variants, questions)
    _print_report(reports)

    print("\nmake ab-eval: informational only — see pass-rate deltas above")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
