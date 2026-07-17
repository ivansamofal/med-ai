"""`make eval` entry point: runs both golden suites against the real,
already-ingested knowledge base and the real critical-threshold CSV, prints a
pass/fail report, and exits non-zero if anything failed — so it can gate a
release the same way `make test` gates a merge.

Requires `make ingest` to have been run at least once (real retrieval needs
the real vector store populated); unlike `make test`, this deliberately does
not use fakes, since the whole point is to catch real regressions in
retrieval quality or the escalation rule's thresholds.
"""

from __future__ import annotations

import sys

from app.eval.models import EvalResult
from app.eval.qa_eval import load_golden_questions, evaluate_qa
from app.eval.redflag_eval import load_golden_red_flags, evaluate_red_flags


def _print_suite(name: str, results: list[EvalResult]) -> bool:
    print(f"\n{name} ({len(results)} cases)")
    print("-" * 60)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.id}: {result.detail}")

    passed = sum(1 for r in results if r.passed)
    print(f"{passed}/{len(results)} passed")
    return passed == len(results)


def main() -> int:
    qa_results = evaluate_qa(load_golden_questions())
    qa_ok = _print_suite("Golden knowledge-base Q&A (citation-presence + faithfulness)", qa_results)

    red_flag_results = evaluate_red_flags(load_golden_red_flags())
    red_flag_ok = _print_suite("Golden red-flag escalation cases", red_flag_results)

    print()
    if qa_ok and red_flag_ok:
        print("make eval: all suites passed")
        return 0

    print("make eval: FAILURES — see above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
