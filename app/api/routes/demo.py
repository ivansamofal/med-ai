"""Demo-only route simulating an external lab vendor API: instead of a real
polling/webhook integration, `GET /demo/next-lab-result` hands back one
canned payload, shaped exactly like `LabResultIngestRequest`, from a small
local fixture (`data/demo/fake_lab_api_responses.json`).

Not part of the 8-phase plan — added so the real pipeline (`POST
/lab-results` -> normalize -> Mongo -> `lab_result_created` on SQS -> the
recommendation worker's LlamaIndex retrieval + LangChain generation) can be
exercised end-to-end from the dashboard by clicking a button, instead of
needing a real lab API to integrate against or hand-writing curl commands.
This endpoint only ever returns fixture data — the caller is still
responsible for actually POSTing it to `/lab-results` to run the real
ingestion path.
"""

from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter

router = APIRouter()

FIXTURE_PATH = Path(__file__).resolve().parents[3] / "data" / "demo" / "fake_lab_api_responses.json"

_queue: list[dict] = []


def _refill_queue() -> None:
    samples = json.loads(FIXTURE_PATH.read_text())
    random.shuffle(samples)
    _queue.extend(samples)


@router.get("/demo/next-lab-result")
async def next_lab_result() -> dict:
    """Pop one simulated vendor payload in random order, one at a time;
    reshuffles a fresh batch once the current one is exhausted — mirroring
    "poll the lab API, get the next result" without ever blocking on a real
    external call.
    """
    if not _queue:
        _refill_queue()

    sample = _queue.pop()
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        **sample,
        # A real vendor poll never redelivers the same order id — fixture
        # samples get reused across reshuffles, so mint a fresh one each time.
        "order_id": f"{sample['order_id']}-{uuid.uuid4().hex[:6]}",
        "collected_at": now,
        "resulted_at": now,
    }
    return {"payload": payload, "remaining_in_batch": len(_queue)}
