from datetime import datetime, timezone

from app.agent.scheduling import is_slot_available, list_available_slots
from app.db.sync_mongo import get_sync_database
from app.domain.appointment import Appointment

DAY = datetime(2026, 8, 3, tzinfo=timezone.utc)  # a Monday, arbitrary


def _book(doctor: str, start: datetime, status: str = "scheduled") -> str:
    appointment = Appointment(patient_id="p-1", doctor=doctor, scheduled_at=start, status=status)
    result = get_sync_database()["appointments"].insert_one(appointment.model_dump())
    return str(result.inserted_id)


def test_list_available_slots_covers_business_hours_when_nothing_booked():
    slots = list_available_slots("dr.jones", DAY)

    assert slots[0] == DAY.replace(hour=9, minute=0)
    assert slots[-1] == DAY.replace(hour=16, minute=30)
    assert len(slots) == 16  # 09:00-17:00 in 30-minute increments


def test_list_available_slots_excludes_booked_times():
    booked_time = DAY.replace(hour=10, minute=0)
    _book("dr.jones", booked_time)

    slots = list_available_slots("dr.jones", DAY)

    assert booked_time not in slots


def test_list_available_slots_ignores_cancelled_appointments():
    booked_time = DAY.replace(hour=10, minute=0)
    _book("dr.jones", booked_time, status="cancelled")

    slots = list_available_slots("dr.jones", DAY)

    assert booked_time in slots


def test_list_available_slots_is_scoped_per_doctor():
    booked_time = DAY.replace(hour=10, minute=0)
    _book("dr.jones", booked_time)

    slots = list_available_slots("dr.smith", DAY)

    assert booked_time in slots


def test_is_slot_available_false_when_booked():
    booked_time = DAY.replace(hour=11, minute=0)
    _book("dr.jones", booked_time)

    assert is_slot_available("dr.jones", booked_time) is False


def test_is_slot_available_excludes_the_appointment_being_rescheduled():
    booked_time = DAY.replace(hour=11, minute=0)
    appointment_id = _book("dr.jones", booked_time)

    assert is_slot_available("dr.jones", booked_time, exclude_appointment_id=appointment_id) is True
