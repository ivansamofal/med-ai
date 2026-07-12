"""The human-in-the-loop approval graph: every AI-authored recommendation is
held behind clinician review before it can reach a patient.

LangGraph owns this because it's stateful and branching — `interrupt()`
pauses execution while waiting on a human, and `MongoDBSaver` checkpoints that
paused state so a pending approval survives a process restart. LangChain
(Phase 3) only ever goes one direction (retrieve -> prompt -> generate); this
graph is what makes the system able to wait, and to do different things
depending on what the clinician decides.
"""

from typing import Literal, TypedDict

from bson import ObjectId
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from app.config import settings
from app.db.sync_mongo import get_sync_client, get_sync_database
from app.domain.audit_log import AuditLogEntry
from app.logging import get_logger

logger = get_logger(__name__)


class ApprovalState(TypedDict):
    recommendation_id: str
    patient_id: str
    recommendation_text: str
    citations: list[str]
    decision: Literal["approved", "edited", "rejected"] | None
    reviewer: str | None
    edited_text: str | None
    reason: str | None


def draft_ready(state: ApprovalState) -> ApprovalState:
    logger.info("approval_draft_ready", recommendation_id=state["recommendation_id"])
    return state


def await_clinician(state: ApprovalState) -> ApprovalState:
    """Pauses the graph (checkpointed) until resumed with a clinician's decision.

    Resume via `Command(resume={"decision": ..., "reviewer": ..., "edited_text": ..., "reason": ...})`.
    """
    response = interrupt(
        {
            "recommendation_id": state["recommendation_id"],
            "recommendation_text": state["recommendation_text"],
            "citations": state["citations"],
        }
    )
    return {
        **state,
        "decision": response["decision"],
        "reviewer": response.get("reviewer"),
        "edited_text": response.get("edited_text"),
        "reason": response.get("reason"),
    }


def finalize_recommendation(recommendation_id: str, status: str, final_text: str) -> None:
    """Persist the clinician's outcome onto the recommendation document."""
    get_sync_database()["recommendations"].update_one(
        {"_id": ObjectId(recommendation_id)},
        {"$set": {"status": status, "recommendation_text": final_text}},
    )


def write_audit_log(recommendation_id: str, action: str, actor: str, reason: str | None) -> None:
    """Record who did what and why — every approval action is audit-logged."""
    entry = AuditLogEntry(recommendation_id=recommendation_id, action=action, actor=actor, reason=reason)
    get_sync_database()["audit_log"].insert_one(entry.model_dump())


def notify_patient_of_recommendation(patient_id: str, recommendation_text: str) -> None:
    """Stub for the actual patient notification channel (email/SMS/portal
    message) — wiring a real channel is out of scope for this phase. Only
    call this for outcomes that should actually reach the patient.
    """
    logger.info("patient_notified", patient_id=patient_id)


def get_checkpointer() -> MongoDBSaver:
    return MongoDBSaver(get_sync_client(), db_name=settings.mongo_db_name)


def _final_text(state: ApprovalState) -> str:
    if state["decision"] == "edited":
        return state["edited_text"]
    return state["recommendation_text"]


def notify_patient(state: ApprovalState) -> ApprovalState:
    """Terminal node for approved/edited drafts: finalize, audit-log, and
    actually notify the patient.
    """
    final_text = _final_text(state)
    finalize_recommendation(state["recommendation_id"], state["decision"], final_text)
    write_audit_log(state["recommendation_id"], state["decision"], state["reviewer"], state["reason"])
    notify_patient_of_recommendation(state["patient_id"], final_text)
    return state


def record_rejection(state: ApprovalState) -> ApprovalState:
    """Terminal node for rejected drafts: finalize and audit-log, but never
    notify the patient — a rejected AI draft must not reach them.
    """
    finalize_recommendation(state["recommendation_id"], "rejected", state["recommendation_text"])
    write_audit_log(state["recommendation_id"], "rejected", state["reviewer"], state["reason"])
    return state


def _route_on_decision(state: ApprovalState) -> str:
    if state["decision"] == "rejected":
        return "record_rejection"
    return "notify_patient"


def build_approval_graph(checkpointer=None) -> CompiledStateGraph:
    """Wires `draft_ready -> await_clinician -> [notify_patient | record_rejection]`.

    "approved" and "edited" both finalize + audit-log + notify the patient
    (only the persisted text differs); "rejected" finalizes + audit-logs but
    deliberately never calls `notify_patient_of_recommendation` — nothing
    should reach a patient for a draft a clinician rejected.
    """
    graph = StateGraph(ApprovalState)
    graph.add_node("draft_ready", draft_ready)
    graph.add_node("await_clinician", await_clinician)
    graph.add_node("notify_patient", notify_patient)
    graph.add_node("record_rejection", record_rejection)

    graph.set_entry_point("draft_ready")
    graph.add_edge("draft_ready", "await_clinician")
    graph.add_conditional_edges(
        "await_clinician",
        _route_on_decision,
        {"notify_patient": "notify_patient", "record_rejection": "record_rejection"},
    )
    graph.add_edge("notify_patient", END)
    graph.add_edge("record_rejection", END)

    return graph.compile(checkpointer=checkpointer or get_checkpointer())
