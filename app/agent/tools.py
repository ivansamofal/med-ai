"""LangChain tool-calling surface for the chat agent: scheduling actions
(backed by `app.agent.scheduling` and the `appointments` collection) and
patient-lab-history / escalation actions (backed by Mongo and the
knowledge-base query engine). Every tool returns a small JSON-serializable
dict — the LLM only ever sees `tool_calls`/`ToolMessage`s, never touches
Mongo or the vector store directly.

Sync throughout (see `app.agent.scheduling`'s docstring for why): these run
inside the chat graph's `ToolNode`, itself invoked from the sync-first
LangGraph checkpointing pattern shared with the approval graph (Phase 4).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from bson import ObjectId
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from llama_index.core import VectorStoreIndex

from app.agent.redflag import detect_red_flag
from app.agent.scheduling import is_slot_available, list_available_slots
from app.db.sync_mongo import get_sync_database
from app.domain.appointment import Appointment
from app.domain.lab_result import LabResult
from app.knowledge.query_engine import query_knowledge_base
from app.logging import get_logger

logger = get_logger(__name__)


def _fetch_patient_lab_results(patient_id: str) -> list[LabResult]:
    docs = get_sync_database()["lab_results"].find({"patient_id": patient_id}).sort("resulted_at", -1)
    results = []
    for doc in docs:
        doc.pop("_id")
        results.append(LabResult(**doc))
    return results


def build_lab_history_context(patient_id: str, index: VectorStoreIndex | None = None) -> dict:
    """Ground a patient's lab history for the chat agent: raw values, a
    red-flag check, and cited context for interpretation.

    WHAT: For each of the patient's lab results (`_fetch_patient_lab_results`,
    provided above), report the raw value and run `detect_red_flag` on it
    (also provided). For results that are abnormal or red-flagged, attach one
    or two cited passages from the knowledge base via `query_knowledge_base`
    so the agent can explain *why* a value matters instead of reciting a bare
    number. Assemble a dict with a `results` list and a top-level `red_flag`
    bool — the graph's routing (your other TODO,
    `app.agent.graph.build_chat_graph`) reads that top-level flag to force an
    escalation instead of letting the agent free-answer.

    WHY: Same retrieval-strategy judgment call as Phase 2/3's TODOs
    (`query_knowledge_base`, `build_context_passages`), but now feeding a
    tool-calling agent instead of a one-shot chain: how much grounding is
    enough for the agent to answer safely, and — new to this phase — how the
    red-flag signal gets communicated back to the graph so a small (or fake)
    model can't simply fail to call `escalate_to_oncall` when it should have.
    Citing every normal result would bury the abnormal ones in noise; citing
    nothing leaves the agent free to editorialize on lab values with no
    grounding at all — the project's core "no hallucinated guidance" premise.

    Hints:
    - `query_knowledge_base(question, source_types=["reference_range", "guideline"], top_k=2, index=index)`
      returns `RetrievedPassage`s with `.source_title` for citations; thread
      `index` through so tests can inject a fixture index instead of the real
      one, same convention as `build_context_passages`.
    - `detect_red_flag(lab_result)` returns a `RedFlag | None`.
    - Keep the return value JSON-serializable (it becomes a `ToolMessage`
      via LangChain's tool-calling machinery).
    """
    lab_results = _fetch_patient_lab_results(patient_id)

    results = []
    any_red_flag = False
    for lab_result in lab_results:
        red_flag = detect_red_flag(lab_result)
        entry = {
            "test_code": lab_result.test_code,
            "test_name": lab_result.test_name,
            "value": lab_result.value,
            "unit": lab_result.unit,
            "is_abnormal": lab_result.is_abnormal,
            "resulted_at": lab_result.resulted_at.isoformat(),
            "red_flag": red_flag.reason if red_flag else None,
            "citations": [],
        }
        if red_flag is not None:
            any_red_flag = True

        if lab_result.is_abnormal or red_flag is not None:
            query = (
                f"{lab_result.test_name} ({lab_result.test_code}) result of "
                f"{lab_result.value} {lab_result.unit}"
            )
            passages = query_knowledge_base(
                query,
                top_k=2,
                source_types=["reference_range", "guideline"],
                index=index,
            )
            entry["citations"] = [p.source_title for p in passages]

        results.append(entry)

    return {"patient_id": patient_id, "results": results, "red_flag": any_red_flag}


@tool
def check_availability(doctor: str, date: str) -> dict:
    """List open 30-minute appointment slots for `doctor` on `date`
    (YYYY-MM-DD), business hours 09:00-17:00."""
    day = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    slots = list_available_slots(doctor, day)
    return {"doctor": doctor, "date": date, "available_slots": [s.isoformat() for s in slots]}


@tool
def book_appointment(
    doctor: str,
    start_time: str,
    duration_minutes: int = 30,
    *,
    patient_id: Annotated[str, InjectedState("patient_id")],
) -> dict:
    """Book an appointment for the current patient with `doctor` at
    `start_time` (ISO 8601). Fails if the slot is already booked."""
    start = datetime.fromisoformat(start_time)
    if not is_slot_available(doctor, start):
        return {"status": "conflict", "message": f"{doctor} is not available at {start_time}."}

    appointment = Appointment(
        patient_id=patient_id, doctor=doctor, scheduled_at=start, duration_minutes=duration_minutes
    )
    result = get_sync_database()["appointments"].insert_one(appointment.model_dump())
    appointment_id = str(result.inserted_id)
    logger.info("appointment_booked", doctor=doctor, appointment_id=appointment_id)
    return {"status": "booked", "appointment_id": appointment_id}


@tool
def reschedule_appointment(
    appointment_id: str,
    new_start_time: str,
    *,
    patient_id: Annotated[str, InjectedState("patient_id")],
) -> dict:
    """Move an existing appointment to `new_start_time` (ISO 8601). Fails if
    the new slot is already booked."""
    db = get_sync_database()
    doc = db["appointments"].find_one({"_id": ObjectId(appointment_id)})
    if doc is None or doc["patient_id"] != patient_id:
        return {"status": "not_found", "message": f"No appointment {appointment_id}."}

    new_start = datetime.fromisoformat(new_start_time)
    if not is_slot_available(doc["doctor"], new_start, exclude_appointment_id=appointment_id):
        return {"status": "conflict", "message": f"{doc['doctor']} is not available at {new_start_time}."}

    db["appointments"].update_one({"_id": ObjectId(appointment_id)}, {"$set": {"scheduled_at": new_start}})
    logger.info("appointment_rescheduled", appointment_id=appointment_id)
    return {"status": "rescheduled", "appointment_id": appointment_id}


@tool
def cancel_appointment(
    appointment_id: str,
    reason: str | None = None,
    *,
    patient_id: Annotated[str, InjectedState("patient_id")],
) -> dict:
    """Cancel an existing appointment."""
    db = get_sync_database()
    doc = db["appointments"].find_one({"_id": ObjectId(appointment_id)})
    if doc is None or doc["patient_id"] != patient_id:
        return {"status": "not_found", "message": f"No appointment {appointment_id}."}

    result = db["appointments"].update_one(
        {"_id": ObjectId(appointment_id)}, {"$set": {"status": "cancelled"}}
    )
    if result.matched_count == 0:
        return {"status": "not_found", "message": f"No appointment {appointment_id}."}

    logger.info("appointment_cancelled", appointment_id=appointment_id, reason=reason)
    return {"status": "cancelled", "appointment_id": appointment_id}


@tool
def get_patient_lab_history(
    *, patient_id: Annotated[str, InjectedState("patient_id")]
) -> dict:
    """Fetch the current patient's lab results, each grounded with cited
    reference-range/guideline context, flagged if any value needs
    escalation."""
    return build_lab_history_context(patient_id)


@tool
def escalate_to_oncall(patient_id: str, reason: str) -> dict:
    """Escalate to the on-call clinician instead of answering directly — use
    this immediately for any red-flag lab value."""
    logger.warning("escalated_to_oncall", patient_id=patient_id, reason=reason)
    return {"status": "escalated", "patient_id": patient_id}


CHAT_TOOLS = [
    check_availability,
    book_appointment,
    reschedule_appointment,
    cancel_appointment,
    get_patient_lab_history,
    escalate_to_oncall,
]
