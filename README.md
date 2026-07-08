# MedAI — Clinical Lab Intelligence & Care Coordination Platform

A Python rewrite of a production Symfony lab-results pipeline, rebuilt with real AI
grounding (LlamaIndex retrieval), a human-in-the-loop safety gate (LangGraph
`interrupt()`), and an agentic chat/scheduling assistant (LangGraph tool-calling) —
built phase by phase as a learning project.

## Phase 1 — Skeleton + ingestion

Direct rewrite of the old Symfony ingestion step. No AI yet.

**What was built:**

- FastAPI app (`app/main.py`) with `POST /lab-results`.
- `app/domain/lab_result.py` — `normalize_lab_result()` maps the loosely-typed
  external lab API payload (string numerics, `Z`-suffixed timestamps, vendor
  `abnormal_flag`) into a strict internal `LabResult` schema, including a
  same-behavior fallback: if the vendor doesn't send an abnormal flag, abnormality
  is derived from the reference range.
- `app/db/mongo.py` — Motor (async) client + `LabResultRepository`, storing
  normalized lab results in MongoDB.
- `app/events/sqs.py` — publishes a `lab_result_created` event to SQS after every
  successful ingest (boto3, run off the event loop via `asyncio.to_thread`).
  Auto-creates the queue when pointed at LocalStack; expects the queue to already
  exist against real AWS.
- `app/config.py` / `app/logging.py` — pydantic-settings config and structlog
  setup with a PHI-redaction processor (`patient_id`, `value`, `raw_payload`,
  `notes` are redacted from every log line).
- `docker-compose.yml` — `mongodb` + `localstack` (SQS) for local dev.
- Tests (`tests/`) run against those same local services — no mocks, no AWS keys.

**Key decisions:**

- Motor (async) for Mongo since routes are async; boto3 (sync) for SQS, wrapped in
  `asyncio.to_thread` — no async SQS client needed for this volume.
- `raw_payload` is kept on the normalized document for traceability/audit, per the
  "audit-log everything" requirement that gets built out fully in Phase 7.
- No TODOs for you in this phase — there's no AI logic yet, and normalization/
  FastAPI/Mongo/SQS wiring is exactly what you already know from the Symfony
  version. TODOs start at Phase 2 (retrieval logic).

**How to run:**

```bash
cp .env.example .env
make up          # starts mongodb + localstack
make test         # pytest, against the services above
make run          # uvicorn on :8000
```

```bash
curl -X POST localhost:8000/lab-results -H 'content-type: application/json' -d '{
  "patient_id": "p-123", "order_id": "ord-456",
  "test_code": "GLU", "test_name": "Glucose",
  "result_value": "105", "unit": "mg/dL",
  "reference_low": "70", "reference_high": "99",
  "abnormal_flag": "H",
  "collected_at": "2026-07-01T10:00:00Z", "resulted_at": "2026-07-01T14:32:00Z",
  "lab_name": "Quest Diagnostics"
}'
```
