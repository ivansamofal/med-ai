from datetime import datetime, timezone

from bson import ObjectId

from app.config import settings
from app.db.mongo import LabResultRepository, get_client, get_database
from app.domain.lab_result import LabResult, ReferenceRange
from app.knowledge.query_engine import RetrievedPassage
from app.recommendations import chain as chain_module
from app.recommendations.chain import generate_recommendation


async def _insert_lab_result() -> str:
    db = get_database()
    lab_result = LabResult(
        patient_id="p-123",
        external_order_id="ord-456",
        test_code="GLU",
        test_name="Glucose",
        value=250.0,
        unit="mg/dL",
        reference_range=ReferenceRange(low=70, high=99),
        is_abnormal=True,
        collected_at=datetime.now(timezone.utc),
        resulted_at=datetime.now(timezone.utc),
        source_lab="Quest Diagnostics",
        raw_payload={},
    )
    return await LabResultRepository(db).insert(lab_result)


async def test_generate_recommendation_persists_pending_review(monkeypatch):
    lab_result_id = await _insert_lab_result()

    monkeypatch.setattr(
        chain_module,
        "build_context_passages",
        lambda lab_result, index=None: [
            RetrievedPassage(
                text="Severe hyperglycemia warrants urgent evaluation.",
                source_title="Diabetes Guideline",
                source_type="guideline",
                score=0.9,
            )
        ],
    )
    monkeypatch.setattr(
        chain_module,
        "build_recommendation_prompt",
        lambda lab_result, passages: "fake prompt body",
    )

    recommendation_id = await generate_recommendation(lab_result_id)

    db = get_client()[settings.mongo_db_name]
    document = await db["recommendations"].find_one({"_id": ObjectId(recommendation_id)})

    assert document is not None
    assert document["status"] == "pending_review"
    assert document["lab_result_id"] == lab_result_id
    assert document["patient_id"] == "p-123"
    assert document["citations"] == ["Fake Source A", "Fake Source B"]
    assert document["recommendation_text"]
