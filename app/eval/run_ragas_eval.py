"""`make ragas-eval` entry point: runs Ragas's non-LLM context-precision/
context-recall metrics against the real, already-ingested knowledge base
(`make ingest` first). Same print-report-and-exit shape as `run_eval.py`,
but informational (like `make ab-eval`) rather than a pass/fail gate — Ragas
scores are continuous, not a golden pass/fail check.
"""

from __future__ import annotations

import sys

from app.eval.ragas_eval import evaluate_with_ragas


def main() -> int:
    summary = evaluate_with_ragas()

    print(f"\nRagas non-LLM context metrics ({len(summary.results)} cases)")
    print("-" * 60)
    for result in summary.results:
        print(f"{result.id}: context_precision={result.context_precision:.2f} context_recall={result.context_recall:.2f}")

    print(f"\nmean context_precision: {summary.mean_context_precision:.2f}")
    print(f"mean context_recall:    {summary.mean_context_recall:.2f}")
    print("\nmake ragas-eval: informational only — see app/eval/ragas_eval.py for metric scope")
    return 0


if __name__ == "__main__":
    sys.exit(main())
