import json

from app.api.routes import demo as demo_module
from app.domain.lab_result import normalize_lab_result


def _reset_queue():
    demo_module._queue.clear()


async def test_next_lab_result_returns_an_ingestable_payload(client):
    _reset_queue()

    response = await client.get("/demo/next-lab-result")

    assert response.status_code == 200
    body = response.json()
    payload = body["payload"]
    # Shaped exactly like the real vendor payload `/lab-results` expects —
    # this must not raise.
    normalize_lab_result(payload)


async def test_next_lab_result_can_be_posted_to_the_real_ingestion_endpoint(client):
    _reset_queue()

    fetched = await client.get("/demo/next-lab-result")
    payload = fetched.json()["payload"]

    ingested = await client.post("/lab-results", json=payload)

    assert ingested.status_code == 201
    assert ingested.json()["patient_id"] == payload["patient_id"]
    assert ingested.json()["test_code"] == payload["test_code"]


async def test_next_lab_result_cycles_through_the_whole_fixture_without_error():
    _reset_queue()
    fixture = json.loads(demo_module.FIXTURE_PATH.read_text())

    seen_order_ids = set()
    for _ in range(len(fixture) * 2 + 3):
        sample = await demo_module.next_lab_result()
        seen_order_ids.add(sample["payload"]["order_id"])

    # Each draw mints a fresh order id even when the underlying fixture
    # sample repeats across a reshuffle.
    assert len(seen_order_ids) == len(fixture) * 2 + 3


async def test_next_lab_result_mints_a_fresh_order_id_each_time():
    _reset_queue()

    first = await demo_module.next_lab_result()
    second = await demo_module.next_lab_result()

    assert first["payload"]["order_id"] != second["payload"]["order_id"]
