import json

from bson import ObjectId

from app.config import settings
from app.db.mongo import get_client
from app.events.sqs import resolve_queue_url, get_sqs_client


async def test_ingest_lab_result_returns_normalized_response(client, sample_raw_lab_result):
    response = await client.post("/lab-results", json=sample_raw_lab_result)

    assert response.status_code == 201
    body = response.json()
    assert body["patient_id"] == "p-123"
    assert body["test_code"] == "GLU"
    assert body["value"] == 105.0
    assert body["is_abnormal"] is True
    assert ObjectId.is_valid(body["id"])


async def test_ingest_lab_result_persists_to_mongo(client, sample_raw_lab_result):
    response = await client.post("/lab-results", json=sample_raw_lab_result)
    lab_result_id = response.json()["id"]

    db = get_client()[settings.mongo_db_name]
    document = await db["lab_results"].find_one({"_id": ObjectId(lab_result_id)})

    assert document is not None
    assert document["patient_id"] == "p-123"
    assert document["is_abnormal"] is True


async def test_ingest_lab_result_publishes_event_to_sqs(client, sample_raw_lab_result):
    response = await client.post("/lab-results", json=sample_raw_lab_result)
    lab_result_id = response.json()["id"]

    sqs = get_sqs_client()
    queue_url = resolve_queue_url()
    received = sqs.receive_message(QueueUrl=queue_url, WaitTimeSeconds=2, MaxNumberOfMessages=1)

    messages = received.get("Messages", [])
    assert len(messages) == 1
    event = json.loads(messages[0]["Body"])
    assert event["event_type"] == "lab_result_created"
    assert event["lab_result_id"] == lab_result_id
    assert event["patient_id"] == "p-123"
    assert event["is_abnormal"] is True

    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=messages[0]["ReceiptHandle"])


async def test_ingest_lab_result_rejects_malformed_payload(client, sample_raw_lab_result):
    del sample_raw_lab_result["result_value"]

    response = await client.post("/lab-results", json=sample_raw_lab_result)

    assert response.status_code == 422
