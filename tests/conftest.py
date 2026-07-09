import os

# Must run before `app.config` is imported anywhere, so tests never touch the
# dev database. Requires `make up` (mongodb + localstack) running locally.
os.environ.setdefault("MONGO_DB_NAME", "medai_test")
os.environ.setdefault("SQS_LAB_RESULT_QUEUE_NAME", "lab-result-created-test")

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.db.mongo import close_client, get_client
from app.events.sqs import get_sqs_client, reset_sqs_state, resolve_queue_url
from app.main import app


@pytest.fixture(autouse=True)
async def clean_mongo():
    yield
    # Motor's client is bound to the event loop it was created on; pytest-asyncio
    # gives each test function its own loop, so the cached client must be closed
    # (not just cleaned up) before that loop closes, or the next test's use of
    # the stale client raises "Event loop is closed".
    db = get_client()[settings.mongo_db_name]
    await db["lab_results"].delete_many({})
    await db["recommendations"].delete_many({})
    close_client()


def _drain_queue() -> None:
    # Standard SQS queues don't guarantee delivery order, and LocalStack
    # persists messages across test runs — a leftover message from a prior
    # test (or a prior crashed session) would otherwise be the one a later
    # test's receive_message picks up. Drain before *and* after each test.
    client = get_sqs_client()
    queue_url = resolve_queue_url()
    while True:
        response = client.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=10, WaitTimeSeconds=0)
        messages = response.get("Messages", [])
        if not messages:
            break
        for message in messages:
            client.delete_message(QueueUrl=queue_url, ReceiptHandle=message["ReceiptHandle"])


@pytest.fixture(autouse=True)
def clean_sqs():
    reset_sqs_state()
    _drain_queue()
    yield
    _drain_queue()
    reset_sqs_state()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_raw_lab_result() -> dict:
    return {
        "patient_id": "p-123",
        "order_id": "ord-456",
        "test_code": "GLU",
        "test_name": "Glucose",
        "result_value": "105",
        "unit": "mg/dL",
        "reference_low": "70",
        "reference_high": "99",
        "abnormal_flag": "H",
        "collected_at": "2026-07-01T10:00:00Z",
        "resulted_at": "2026-07-01T14:32:00Z",
        "lab_name": "Quest Diagnostics",
    }
