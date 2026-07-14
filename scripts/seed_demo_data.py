"""One-off seed script: populates lab results, draft recommendations (with a
mix of pending/approved/edited/rejected review outcomes), and appointments,
so the read-only dashboard (`/dashboard`) has something to show.

Not part of the real ingestion path — calls the same functions the API and
worker call (`normalize_lab_result`, `generate_recommendation`,
`start_review`/`resume_review`), directly and in-process, so this doesn't
need the FastAPI server, the SQS worker, or LocalStack running. Requires
`make up` (Mongo) and `make ingest` (the knowledge-base index, for the two
red-flag lab results' recommendation drafts to have something to cite).

Safe to re-run; every run adds a fresh batch rather than upserting.
"""

import asyncio
from datetime import datetime, timedelta, timezone

from app.approval.service import resume_review, start_review
from app.db.mongo import LabResultRepository, RecommendationRepository, get_database
from app.db.sync_mongo import get_sync_database
from app.domain.appointment import Appointment
from app.domain.lab_result import normalize_lab_result
from app.recommendations.chain import generate_recommendation

# (patient_id, test_code, test_name, value, unit, ref_low, ref_high, abnormal_flag, lab_name)
LAB_RESULTS = [
    ("p-2001", "GLU", "Glucose", "250", "mg/dL", "70", "99", "H", "Quest Diagnostics"),
    ("p-2001", "HBA1C", "Hemoglobin A1c", "8.9", "%", "4.0", "5.6", "H", "Quest Diagnostics"),
    ("p-2001", "LDL", "LDL Cholesterol", "165", "mg/dL", "0", "99", "H", "Quest Diagnostics"),
    ("p-2002", "K", "Potassium", "7.0", "mmol/L", "3.5", "5.1", "H", "LabCorp"),  # red flag
    ("p-2002", "NA", "Sodium", "140", "mmol/L", "135", "145", "", "LabCorp"),
    ("p-2003", "CR", "Creatinine", "1.1", "mg/dL", "0.6", "1.3", "", "Quest Diagnostics"),
    ("p-2003", "PLT", "Platelet Count", "18", "10^3/uL", "150", "450", "L", "Quest Diagnostics"),  # red flag
    ("p-2003", "HGB", "Hemoglobin", "13.2", "g/dL", "12.0", "17.5", "", "Quest Diagnostics"),
]


async def _seed_lab_results() -> list[dict]:
    repository = LabResultRepository(get_database())
    now = datetime.now(timezone.utc)
    seeded = []
    for i, (patient_id, code, name, value, unit, low, high, flag, lab) in enumerate(LAB_RESULTS):
        raw = {
            "patient_id": patient_id,
            "order_id": f"seed-ord-{i}",
            "test_code": code,
            "test_name": name,
            "result_value": value,
            "unit": unit,
            "reference_low": low,
            "reference_high": high,
            "abnormal_flag": flag,
            "collected_at": (now - timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
            "resulted_at": now.isoformat().replace("+00:00", "Z"),
            "lab_name": lab,
        }
        lab_result = normalize_lab_result(raw)
        lab_result_id = await repository.insert(lab_result)
        seeded.append({"id": lab_result_id, "is_abnormal": lab_result.is_abnormal})
    return seeded


async def _seed_recommendations(seeded_lab_results: list[dict]) -> list[str]:
    db = get_database()
    recommendation_ids = []
    for entry in seeded_lab_results:
        if not entry["is_abnormal"]:
            continue
        recommendation_id = await generate_recommendation(entry["id"])
        recommendation = await RecommendationRepository(db).get_by_id(recommendation_id)
        start_review(
            recommendation_id,
            recommendation.patient_id,
            recommendation.recommendation_text,
            recommendation.citations,
        )
        recommendation_ids.append(recommendation_id)
    return recommendation_ids


def _resolve_reviews(recommendation_ids: list[str]) -> None:
    # Cycle through outcomes so the dashboard shows every status; leave the
    # last one pending_review so there's always something left to act on.
    outcomes = ["approved", "edited", "rejected"]
    for i, recommendation_id in enumerate(recommendation_ids[:-1]):
        outcome = outcomes[i % len(outcomes)]
        if outcome == "approved":
            resume_review(recommendation_id, "approved", "dr.jones")
        elif outcome == "edited":
            resume_review(
                recommendation_id, "edited", "dr.jones", edited_text="Edited: recheck in 2 weeks."
            )
        else:
            resume_review(recommendation_id, "rejected", "dr.jones", reason="Needs more clinical context.")


def _seed_appointments() -> None:
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).replace(
        minute=0, second=0, microsecond=0
    )
    appointments = [
        Appointment(patient_id="p-2001", doctor="dr.jones", scheduled_at=tomorrow.replace(hour=10)),
        Appointment(patient_id="p-2002", doctor="dr.smith", scheduled_at=tomorrow.replace(hour=14)),
        Appointment(
            patient_id="p-2003",
            doctor="dr.jones",
            scheduled_at=tomorrow.replace(hour=11),
            status="cancelled",
        ),
    ]
    get_sync_database()["appointments"].insert_many([a.model_dump() for a in appointments])


async def main() -> None:
    lab_results = await _seed_lab_results()
    print(f"Seeded {len(lab_results)} lab results.")

    recommendation_ids = await _seed_recommendations(lab_results)
    print(f"Generated {len(recommendation_ids)} draft recommendations.")

    _resolve_reviews(recommendation_ids)
    print("Resolved reviews: approved/edited/rejected, one left pending_review.")

    _seed_appointments()
    print("Seeded 3 appointments.")


if __name__ == "__main__":
    asyncio.run(main())
