import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.approval import graph as graph_module
from app.approval.graph import build_approval_graph


@pytest.fixture
def calls(monkeypatch):
    recorded = {"finalize": [], "audit": [], "notify": []}
    monkeypatch.setattr(
        graph_module, "finalize_recommendation", lambda *a: recorded["finalize"].append(a)
    )
    monkeypatch.setattr(graph_module, "write_audit_log", lambda *a: recorded["audit"].append(a))
    monkeypatch.setattr(
        graph_module, "notify_patient_of_recommendation", lambda *a: recorded["notify"].append(a)
    )
    return recorded


def _initial_state(recommendation_id: str) -> dict:
    return {
        "recommendation_id": recommendation_id,
        "patient_id": "p-123",
        "recommendation_text": "Original draft text.",
        "citations": ["Diabetes Guideline"],
        "decision": None,
        "reviewer": None,
        "edited_text": None,
        "reason": None,
    }


def _run(recommendation_id: str, resume: dict):
    compiled = build_approval_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": recommendation_id}}

    first = compiled.invoke(_initial_state(recommendation_id), config=config)
    assert "__interrupt__" in first, "graph should pause at await_clinician"

    return compiled.invoke(Command(resume=resume), config=config)


def test_approve_flow_finalizes_and_notifies_patient(calls):
    _run("rec-1", {"decision": "approved", "reviewer": "dr.jones"})

    assert calls["finalize"] == [("rec-1", "approved", "Original draft text.")]
    assert calls["audit"] == [("rec-1", "approved", "dr.jones", None)]
    assert len(calls["notify"]) == 1


def test_edit_flow_finalizes_with_edited_text(calls):
    _run(
        "rec-2",
        {"decision": "edited", "reviewer": "dr.jones", "edited_text": "Edited text."},
    )

    assert calls["finalize"] == [("rec-2", "edited", "Edited text.")]
    assert len(calls["notify"]) == 1


def test_reject_flow_does_not_notify_patient(calls):
    _run(
        "rec-3",
        {"decision": "rejected", "reviewer": "dr.jones", "reason": "Needs more context"},
    )

    assert calls["finalize"] == [("rec-3", "rejected", "Original draft text.")]
    assert calls["audit"] == [("rec-3", "rejected", "dr.jones", "Needs more context")]
    assert calls["notify"] == []
