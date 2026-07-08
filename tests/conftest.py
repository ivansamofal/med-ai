import os

# Must run before `app.config` is imported anywhere, so tests never touch the
# dev database. Requires `make up` (mongodb + localstack) running locally.
os.environ.setdefault("MONGO_DB_NAME", "medai_test")
os.environ.setdefault("SQS_LAB_RESULT_QUEUE_NAME", "lab-result-created-test")

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.db.mongo import get_client
from app.events.sqs import reset_sqs_state
from app.main import app


@pytest.fixture(autouse=True)
async def clean_mongo():
    yield
    db = get_client()[settings.mongo_db_name]
    await db["lab_results"].delete_many({})


@pytest.fixture(autouse=True)
def clean_sqs():
    reset_sqs_state()
    yield
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
