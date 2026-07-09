"""Entry points the rest of the app calls into the approval graph through:
start a review right after a recommendation is drafted, resume it when a
clinician acts. Both are sync (the graph is sync-first) — callers on the
async side (FastAPI routes, the SQS worker) wrap these in `asyncio.to_thread`.
"""

from langgraph.types import Command

from app.approval.graph import ApprovalState, build_approval_graph

_compiled_graph = None


def _graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_approval_graph()
    return _compiled_graph


def _config(recommendation_id: str) -> dict:
    return {"configurable": {"thread_id": recommendation_id}}


def start_review(recommendation_id: str, patient_id: str, recommendation_text: str, citations: list[str]) -> None:
    initial_state: ApprovalState = {
        "recommendation_id": recommendation_id,
        "patient_id": patient_id,
        "recommendation_text": recommendation_text,
        "citations": citations,
        "decision": None,
        "reviewer": None,
        "edited_text": None,
        "reason": None,
    }
    _graph().invoke(initial_state, config=_config(recommendation_id))


def resume_review(
    recommendation_id: str,
    decision: str,
    reviewer: str,
    edited_text: str | None = None,
    reason: str | None = None,
) -> dict:
    return _graph().invoke(
        Command(resume={"decision": decision, "reviewer": reviewer, "edited_text": edited_text, "reason": reason}),
        config=_config(recommendation_id),
    )
