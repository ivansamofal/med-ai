"""Deterministic scheduling backend for the chat agent's appointment tools: a
fixed business-hours calendar per doctor, checked against booked appointments
in Mongo for conflicts. Plain calendar math, not AI logic — same
"you already know this" category as the FastAPI/Mongo/SQS wiring in earlier
phases.

Runs on the sync Mongo client (`app.db.sync_mongo`), not the async Motor
client: these functions are called from inside the chat graph's tool node,
which — like the approval graph (Phase 4) — is a sync unit wrapped in
`asyncio.to_thread` at the FastAPI boundary, since LangGraph's `MongoDBSaver`
checkpointer has no async equivalent yet.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from bson import ObjectId

from app.db.sync_mongo import get_sync_database

BUSINESS_HOURS_START = 9
BUSINESS_HOURS_END = 17
SLOT_MINUTES = 30


def _day_slots(date: datetime) -> list[datetime]:
    start = date.replace(hour=BUSINESS_HOURS_START, minute=0, second=0, microsecond=0)
    end = date.replace(hour=BUSINESS_HOURS_END, minute=0, second=0, microsecond=0)
    slots = []
    current = start
    while current < end:
        slots.append(current)
        current += timedelta(minutes=SLOT_MINUTES)
    return slots


def _booked_slots(doctor: str, date: datetime) -> set[datetime]:
    day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)
    docs = get_sync_database()["appointments"].find(
        {
            "doctor": doctor,
            "status": "scheduled",
            "scheduled_at": {"$gte": day_start, "$lt": day_end},
        }
    )
    return {doc["scheduled_at"] for doc in docs}


def list_available_slots(doctor: str, date: datetime) -> list[datetime]:
    """Open 30-minute slots for `doctor` on `date`'s calendar day (09:00-17:00)."""
    booked = _booked_slots(doctor, date)
    return [slot for slot in _day_slots(date) if slot not in booked]


def is_slot_available(
    doctor: str, start_time: datetime, exclude_appointment_id: str | None = None
) -> bool:
    """Whether `doctor` has no other scheduled appointment at `start_time`.

    `exclude_appointment_id` lets a reschedule check availability without
    conflicting with the very appointment being moved.
    """
    query: dict = {"doctor": doctor, "status": "scheduled", "scheduled_at": start_time}
    if exclude_appointment_id is not None:
        query["_id"] = {"$ne": ObjectId(exclude_appointment_id)}
    return get_sync_database()["appointments"].find_one(query) is None
