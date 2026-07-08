# Project: MedAI — Clinical Lab Intelligence & Care Coordination Platform

I want to build a portfolio-grade project, phase by phase, to get hands-on experience
with LlamaIndex, LangChain, and LangGraph — grounded in a system I actually operated in
production (PHP/Symfony), rewritten in Python with real AI features, not tech-stacking.
Treat this as a real production system, not a demo script.

## Who I am

Senior backend engineer (10+ years PHP → Python). Already strong: FastAPI, async Python,
AWS (Bedrock, S3, SQS via boto3), PostgreSQL, MongoDB, event-driven architecture, Docker.
In my last role I built exactly the "before" version of this system: Symfony ingested lab
results from an external API, stored them in MongoDB, and generated AI recommendations via
Symfony → SQS → Bedrock → Symfony → MongoDB. That flow had no grounding (raw prompt, no
citations), no safety gate before a recommendation reached a patient, and no agency (it only
ever went one direction). This project fixes those three gaps, on purpose, as the reason to
use each new tool. Learning: LangChain, LlamaIndex, LangGraph, MongoDB Atlas Vector Search,
prompt engineering, LLM observability.
Don't explain FastAPI/SQL/Docker/Mongo/SQS basics to me; DO explain the AI-stack concepts
and trade-offs (retrieval strategies, agent state, checkpointing, guardrails).

## What the system does

1. Ingests lab results from an external API, normalizes them, stores in MongoDB, publishes
   a `lab_result_created` event to SQS (this part is the direct rewrite of the old system).
2. Indexes a clinical knowledge base (guideline PDFs, drug-interaction dicts, lab reference
   ranges) with LlamaIndex, so recommendations are retrieved-and-cited, not hallucinated.
3. Generates a draft recommendation via a LangChain chain (retrieve → prompt → Bedrock →
   parse), storing citations alongside the text.
4. Holds every AI-generated recommendation behind a **human-in-the-loop approval graph**
   (LangGraph, `interrupt()`) before it can reach a patient — a clinician approves, edits,
   or rejects.
5. Exposes a patient/clinician chatbot (LangGraph agent, tool-calling) that can actually
   act: check availability, book/reschedule/cancel appointments, pull a patient's lab
   history via the knowledge base, and escalate red-flag values to an on-call clinician.
6. Evaluates itself: a golden Q&A set checked for faithfulness/citation-presence, red-flag
   detection tests, and audit logging of every AI decision.

## Architecture requirements

- **Python 3.12+, FastAPI** for the API layer; async where it matters.
- **Vector store behind an interface**: one `VectorStore` protocol — `ChromaStore` (default,
  local, zero external deps) and `AtlasVectorStore` (MongoDB Atlas Vector Search, behind an
  env var, since production already runs MongoDB — no new database to justify).
- **LLM behind an interface too**: default AWS Bedrock (Claude); a deterministic `FakeLLM`
  so the whole test suite runs offline with no keys.
- **LlamaIndex** owns ingestion + indexing of the knowledge base only (guidelines, drug
  dicts, reference ranges). It is not used for orchestration.
- **LangChain** owns the single-shot retrieve → prompt → generate chain for the draft
  recommendation. It is not used for anything stateful or branching.
- **LangGraph** owns everything stateful/branching: the approval graph (with `interrupt()`
  and checkpointed state so it survives a restart while waiting on a clinician) and the
  chat/appointment agent (tool-calling, conversation memory, red-flag escalation routing).
- **MongoDB** for lab results, recommendations, chat sessions, audit log.
- **AWS SQS (or LocalStack locally)** for the ingestion → recommendation event flow.
- **docker-compose** for local infra: mongodb, chroma (or a Mongo container doubling as
  Atlas-like target), localstack.
- Config via pydantic-settings; structured logging (structlog) with PHI-safe redaction;
  typed code; pytest.
- A `Makefile` with common commands (`make up`, `make test`, `make eval`, `make ingest`).

## Build phases (one at a time — do NOT scaffold everything upfront)

Work strictly phase by phase. Finish a phase (tests green, short README section written)
before starting the next. At the start of each phase, tell me the plan for that phase in
a few sentences, then build it.

1. **Skeleton + ingestion**: repo layout, docker-compose (mongodb, localstack), FastAPI app,
   `POST /lab-results` → normalize → store in Mongo → publish `lab_result_created` to SQS.
   This is the direct Python rewrite of the old Symfony ingestion step — no AI yet.
2. **Knowledge base (LlamaIndex)**: loaders for guideline PDFs, drug-interaction dict (JSON),
   lab reference ranges (CSV); build/query engine behind the `VectorStore` protocol
   (`ChromaStore` default, `AtlasVectorStore` behind env var); `query_engine.py` exposed as
   a plain callable so it can later become a LangChain Tool.
3. **Recommendation chain (LangChain)**: SQS consumer worker picks up `lab_result_created`,
   calls the query engine for relevant guideline passages, builds a prompt with the lab
   result + retrieved context, calls the LLM interface, parses out recommendation text +
   citations, writes a `recommendation` document (status=`pending_review`) to Mongo.
4. **Approval graph (LangGraph)**: graph nodes `draft_ready → interrupt (await clinician) →
   approved/edited/rejected → notify_patient`. Checkpointed so a pending approval survives
   a process restart. `POST /reviews/{id}/approve|edit|reject` resumes the graph.
5. **Chat/appointment agent (LangGraph, tool-calling)**: tools — `check_availability`,
   `book_appointment`, `reschedule_appointment`, `cancel_appointment`,
   `get_patient_lab_history` (backed by the query engine), `escalate_to_oncall`. Guardrails:
   never emit a diagnosis, always cite when referencing guideline content, escalate rather
   than answer on red-flag lab values. Conversation state persisted per patient session.
6. **Evaluation & guardrail tests**: ~20-question golden set over the knowledge base
   checked for citation-presence and faithfulness; red-flag-value test cases that must
   trigger escalation, not a plain answer; `make eval` runs and reports pass/fail.
7. **Observability + hardening**: per-request token/cost tracking, latency tracing, PHI-safe
   log redaction, rate limiting on `/chat`, retry/backoff on the SQS worker, full audit log
   of every AI-authored decision (draft, approval action, agent tool call).
8. *(Later, optional)* Split the chat agent into a supervisor pattern — intake agent,
   retrieval agent, scheduling agent, compliance/guardrail agent — each a LangGraph node,
   to demonstrate multi-agent orchestration once the single-agent version is solid.

## How I want to work — mentor mode (important)

I'm doing this project to LEARN, not to watch you code. For each phase:

- YOU build: project plumbing, configs, docker, interfaces/protocols, tests, and anything
  I've already mastered (FastAPI routes, Mongo access, SQS wiring).
- I build: the core AI logic. Leave **2–4 clearly marked TODO functions per phase** for me —
  e.g. the retrieval query in the query engine, the recommendation prompt template, the
  LangGraph approval graph's conditional edges, the red-flag escalation rule. Each TODO
  gets: a docstring explaining WHAT and WHY, hints about the approach (not the code), and a
  failing test that turns green when I've done it right.
- After I implement my TODOs and tests pass, review my code like a senior colleague
  (correctness first, then idiom), then we move to the next phase.
- When there's a genuine design trade-off (e.g. Chroma vs Atlas Vector Search, chain vs
  graph for a given step, how aggressive the red-flag escalation rule should be), present
  the options in 2–3 sentences each and let me pick.

If I say **"just build it"** at any point, drop mentor mode and implement everything yourself.

## Quality bar

- Everything runs locally with zero API keys (fakes/local vector store) — real
  Bedrock/Atlas behind env vars.
- `make test` green at the end of every phase; no phase is "done" without tests.
- README grows with each phase: what was built, key decisions, how to run it.
- Every AI-authored write (recommendation, approval action, booked appointment) is
  audit-logged with who/what/why — this is a medical system, treat it like one.
- Git: init the repo, commit at least once per phase with a meaningful message.

## Start here

Start with Phase 1 now: show me the planned repo layout first, then build it.
