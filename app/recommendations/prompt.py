"""TODO (Phase 3, yours to implement): the recommendation prompt template."""

from app.domain.lab_result import LabResult
from app.knowledge.query_engine import RetrievedPassage


def build_recommendation_prompt(lab_result: LabResult, passages: list[RetrievedPassage]) -> str:
    """Build the prompt body the LLM will reason over.

    WHAT: Combine the lab result (test, value, unit, reference range, whether
    it's flagged abnormal) with the retrieved guideline/reference-range/
    drug-interaction passages into a single prompt that asks for a grounded,
    cited recommendation. The chain wraps whatever you return here in a
    `ChatPromptTemplate` alongside the output parser's format instructions
    (see `chain.py`) — you're writing the *content*, not the output format.

    WHY: This is the one place prompt-engineering trade-offs live in the whole
    system — how much of the lab result to state explicitly vs. let the model
    infer, how to present retrieved passages (verbatim quotes vs. a source
    list), and what safety framing to include (never diagnose, defer to a
    clinician, flag abnormal/critical values explicitly) directly shape
    whether the draft that reaches a clinician for review is trustworthy or
    generic filler.

    Hints:
    - State the lab result plainly: test name, value + unit, reference range,
      whether it's flagged abnormal.
    - Include each passage's text *and* its `source_title` in the prompt, and
      instruct the model to only use information from the passages provided —
      no outside knowledge, no diagnosis.
    - Every AI-authored recommendation is held behind human review before it
      reaches a patient (Phase 4) — the prompt should read like it's producing
      a *draft* for a clinician, not a final answer.
    """
    abnormal_note = "This value IS flagged abnormal." if lab_result.is_abnormal else "This value is within range."

    lines = [
        "You are drafting a recommendation for a clinician to review before it "
        "reaches a patient. This is a draft, not a final answer — the clinician "
        "may approve, edit, or reject it.",
        "",
        "Lab result:",
        f"- Test: {lab_result.test_name} ({lab_result.test_code})",
        f"- Value: {lab_result.value} {lab_result.unit}",
        f"- Reference range: {lab_result.reference_range.low}-{lab_result.reference_range.high} {lab_result.unit}",
        f"- {abnormal_note}",
        "",
    ]

    if passages:
        lines.append("Reference passages (use only these, cite each one you rely on by its title):")
        for passage in passages:
            lines.append(f"- [{passage.source_title}] {passage.text}")
        lines.append("")
    else:
        lines.append("No reference passages were retrieved for this result.")
        lines.append("")

    lines.append(
        "Write a short recommendation for the clinician based only on the "
        "information above. Do not provide a diagnosis. Do not use any "
        "outside medical knowledge beyond what's stated in the reference "
        "passages. Cite every passage you rely on by its title."
    )

    return "\n".join(lines)
