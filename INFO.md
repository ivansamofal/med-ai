# INFO — How LlamaIndex, LangChain, LangGraph, RAG, and Recommendations Work Here

This is a learning-oriented walkthrough of the three frameworks used in this
project and the two "generative" flows built on top of them (RAG retrieval and
recommendation generation). Each section explains the general concept first,
then points at the exact file in this repo that implements it.

The one-line division of labor (from `medai_architecture_graph.txt`):

```
LlamaIndex   -> knowledge base ingestion + retrieval only
LangChain    -> single-shot retrieve -> prompt -> generate (no branching/state)
LangGraph    -> anything stateful/branching: approval workflow, tool-calling agent
```

---

## 1. LlamaIndex — the knowledge base (ingestion + retrieval)

**What it's for in general:** LlamaIndex specializes in turning unstructured
sources (PDFs, text, CSV/JSON rows) into a searchable *index*. Its core
pipeline is:

```
raw sources -> Documents -> chunked into Nodes -> embedded into vectors -> VectorStore -> Index -> Retriever
```

- **Document**: one loaded source (e.g. one guideline file), with metadata.
- **Node** (here `TextNode`): a chunk of a Document, small enough to embed and
  retrieve precisely, carrying the parent's metadata forward.
- **Embedding model**: turns text into a vector so "similar meaning" becomes
  "nearby vectors."
- **VectorStore**: where the vectors + text + metadata actually live
  (Chroma, Postgres/pgvector, MongoDB Atlas Vector Search, Pinecone, etc.).
- **Index**: LlamaIndex's object wrapping a vector store + embedding model,
  exposing `.as_retriever()` / `.as_query_engine()`.
- **Retriever vs. query engine**: a *retriever* returns raw ranked passages; a
  *query engine* additionally calls an LLM to synthesize a prose answer from
  those passages. This project deliberately only uses the retriever — LLM
  synthesis is LangChain's job here (see the division-of-labor line above).

**In this repo:**

| Step | File |
|---|---|
| Load guideline/drug/reference-range sources into `Document`s | `app/knowledge/loaders.py` |
| Split guideline `Document`s into retrieval-sized `TextNode`s (`SentenceSplitter`, chunk_size=256/overlap=32) | `app/knowledge/chunking.py` |
| Embedding model interface: `FastEmbedEmbedding` (real, local ONNX, no API key) vs `FakeEmbedding` (hash-based, tests) | `app/knowledge/embeddings.py` |
| Vector store interface: `ChromaStore` (default, local) vs `AtlasVectorSearch` (MongoDB Atlas, prod) | `app/knowledge/vector_store.py` |
| Build the index (`build_index`) / reopen it (`get_index`) | `app/knowledge/ingest.py` |
| The **only** public retrieval entry point: `query_knowledge_base()` — takes a question + optional `source_types` filter, returns `RetrievedPassage`s (text + citation + score) | `app/knowledge/query_engine.py` |

Everything downstream (the recommendation chain, the chat agent) calls
`query_knowledge_base()` — nothing else touches the vector store directly.
That's why it's described as "the knowledge base's single public entry point."

---

## 2. RAG (Retrieval-Augmented Generation) — how it actually works, end to end

RAG means: instead of asking an LLM to answer from what it memorized during
training, you retrieve real, current, domain-specific text first, and force
the LLM to ground its answer in that text. The generic recipe:

1. **Ingest** your knowledge base once (offline/batch): load → chunk → embed →
   store. (Section 1, `build_index`.)
2. **At request time**, turn the user's question/context into a *query
   string* (this is a judgment call, not just "pass the question through") and
   retrieve the top-k most relevant chunks, optionally filtered by metadata.
3. **Build a prompt** that includes those retrieved chunks verbatim (with
   their citations) plus an instruction like "only use what's in these
   passages, cite what you use, don't invent anything else."
4. **Call the LLM** with that prompt. The model's job is now synthesis and
   citation, not recall.
5. **Parse** the model's response into a structured shape you can store/display,
   rather than trusting free-form prose.

Why this matters for a clinical system specifically: an LLM's parametric
knowledge is not verifiable and not necessarily current or approved — a
guideline can be wrong or out of date in the model's weights. Retrieval makes
the actual source text the ground truth, and citations let a clinician verify
it.

**In this repo, two places do RAG, sharing the same retrieval layer:**

- **Recommendation generation** (Section 3 below) — `app/recommendations/retrieval.py`
  builds the query string from a lab result (test name/code, value, unit,
  abnormal/normal), decides which `source_types` to include (drug-interaction
  passages only pulled in when the result is abnormal, to avoid noise), and
  calls `query_knowledge_base()`.
- **Chat agent's lab-history tool** — `app/agent/tools.py::build_lab_history_context`
  does the same thing per-lab-result, but only fetches citations for abnormal/
  red-flagged results, and additionally surfaces a `red_flag` boolean the
  graph uses for routing (see Section 4).

Both places share the *retrieval trade-off* pattern: query too narrow and you
miss the passage that matters; query too broad and irrelevant guideline
chunks crowd out the ones that do.

---

## 3. LangChain — the single-shot recommendation chain

**What it's for in general:** LangChain provides composable building blocks —
prompt templates, chat models, output parsers — wired together with the `|`
(pipe) operator into an **LCEL** (LangChain Expression Language) chain:

```python
chain = prompt_template | chat_model | output_parser
result = await chain.ainvoke({...})
```

Each stage is swappable behind an interface (a real model vs. a fake one for
tests, e.g.), and the chain only ever flows one direction: input → prompt →
model → parsed output. There's no state carried between calls and no
conditional branching — that's precisely the boundary where this project
switches to LangGraph instead.

**In this repo** (`app/recommendations/`), the chain for turning one lab
result into a draft recommendation:

```
generate_recommendation(lab_result_id)
  1. fetch the LabResult from Mongo
  2. build_context_passages()  -> RAG retrieval (Section 2), via query_knowledge_base
  3. build_recommendation_prompt()  -> assembles lab result + cited passages into prompt text
  4. chain = ChatPromptTemplate | get_chat_model() | PydanticOutputParser
     parsed = await chain.ainvoke({prompt_body, format_instructions})
  5. persist Recommendation(text, citations, raw_llm_response) to Mongo, status=pending_review
```

- **`app/recommendations/prompt.py`** — builds the actual prompt content:
  states the lab result plainly, embeds each retrieved passage with its
  source title, and instructs the model "use only these passages, no outside
  knowledge, no diagnosis, this is a draft for a clinician."
- **`app/recommendations/parser.py`** — defines `ParsedRecommendation`
  (`text` + `citations: list[str]`) and wraps it in LangChain's
  `PydanticOutputParser`, whose `get_format_instructions()` is embedded in the
  prompt so the model is told the exact JSON schema to emit — no regex
  scraping of free text.
- **`app/llm/interface.py`** — the chat model itself sits behind
  `get_chat_model()`: `ChatBedrockConverse` (real, AWS Bedrock/Claude) in
  production, or LangChain's `FakeListChatModel` returning a canned, valid
  `ParsedRecommendation` JSON blob so the whole chain — parser included — runs
  in tests with zero API keys.
- **`app/recommendations/chain.py`** ties all of the above together as
  `generate_recommendation()`, invoked by `app/workers/recommendation_worker.py`
  off the `lab_result_created` SQS event.

This draft is never shown to a patient directly — every one goes through the
LangGraph approval flow next.

---

## 4. LangGraph — stateful, branching, human-in-the-loop workflows

**What it's for in general:** LangGraph models a workflow as a graph of nodes
(plain functions on a shared `State` TypedDict) and edges (including
*conditional* edges chosen by a routing function). Unlike a LangChain chain,
a LangGraph graph can:

- **Branch**: different next node depending on state (`add_conditional_edges`).
- **Pause and resume**: `interrupt()` suspends a node mid-run to wait on
  something external (a human decision); calling the graph again with
  `Command(resume=...)` continues exactly where it left off.
- **Checkpoint**: a `checkpointer` (here `MongoDBSaver`) persists state after
  every node, so a paused/interrupted graph survives a process restart — the
  pause isn't just "blocked in memory," it's durable.
- **Loop**: a node can route back to an earlier node (e.g. a tool-calling
  agent looping model → tools → model until the model stops requesting
  tools).

This project uses LangGraph for exactly the two things a one-directional
LangChain chain can't do:

### 4a. The approval graph (`app/approval/graph.py`)

Every AI-authored recommendation must be reviewed by a clinician before it
can reach a patient — this is the human-in-the-loop requirement.

```
draft_ready -> await_clinician --interrupt()--> [clinician POSTs a decision]
                                     |
                    (conditional edge on state["decision"])
                                     |
                 approved/edited -> notify_patient -> END
                 rejected        -> record_rejection -> END
```

- `await_clinician` calls `interrupt(...)`, which pauses the graph and
  surfaces the recommendation + citations to a review UI. The clinician's
  `POST /reviews/{id}/approve|edit|reject` resumes the graph with
  `Command(resume={"decision": ..., "reviewer": ..., ...})`.
- `_route_on_decision` is the conditional-edge function: rejected drafts go to
  `record_rejection` (audit-logged, but `notify_patient_of_recommendation` is
  **never** called — a rejected AI draft must never reach a patient);
  approved/edited drafts go to `notify_patient` (finalize in Mongo, write to
  the audit log, then actually notify).
- `MongoDBSaver` checkpoints state at every node — a pending approval survives
  a service restart because the graph's paused state lives in Mongo, not just
  in a process's memory.

### 4b. The chat/appointment agent (`app/agent/graph.py`)

A tool-calling ReAct-style agent with a safety-critical routing decision that
can't be left to the model's judgment alone.

```
agent --tools_condition--> [tools | END]
tools --_route_after_tools--> [agent (loop back) | escalate (terminal)]
```

- `agent` node: binds `CHAT_TOOLS` (`app/agent/tools.py`: appointment
  scheduling + `get_patient_lab_history` + `escalate_to_oncall`) to the chat
  model and invokes it with the running message history.
- `tools_condition` (LangGraph's prebuilt routing function) sends the flow to
  `ToolNode(CHAT_TOOLS)` if the model requested a tool call, else `END`.
- After tools run, `_tool_results_flagged` inspects the just-appended
  `ToolMessage`s for a `red_flag` field (set by
  `build_lab_history_context`/`detect_red_flag` in Section 2). If any tool
  result flagged a red-flag lab value, the graph **deterministically** routes
  to `escalate` — a terminal node emitting a fixed "flagged to on-call"
  message — instead of looping back to the model. This is the guardrail
  called out in the architecture doc: "red-flag lab value → escalate, don't
  answer," enforced by the graph itself so a weaker (or the offline fake)
  model can't just answer with the number anyway.
- `MongoDBSaver` again checkpoints so a patient's conversation state
  (`ChatState.messages`) survives across turns/restarts, scoped per chat
  session via the checkpointer's thread id.

---

## 5. Putting it together — the full event flow

```
Lab API -> Ingestion service -> MongoDB (raw lab result)
                                    |  publish "lab_result_created"
                                    v
                              AWS SQS / LocalStack
                                    |
                                    v
                     Recommendation worker (LangChain chain, Section 3)
                       - RAG-retrieves guideline/reference-range/drug-interaction
                         passages via LlamaIndex (Section 1/2)
                       - Bedrock (or FakeLLM) generates a cited draft
                       - writes Recommendation(status=pending_review) to Mongo
                                    |
                                    v
                     Approval graph (LangGraph, Section 4a)
                       - interrupts, waits for clinician review
                       - approved/edited -> notify_patient (final, audited)
                       - rejected -> record_rejection (never reaches patient)

                     (separately) Chat/appointment agent (LangGraph, Section 4b)
                       - tool-calling loop: scheduling + grounded lab history
                       - red-flag lab value -> forced escalation, never a
                         free-form answer
```

Evaluation harness (citation-presence + faithfulness checks over the golden
Q&A set, red-flag escalation checks over the golden lab-value set — Phase 6,
`app/eval/`) is built; see the README's Phase 6 section and Section 6 below
for where it fits alongside the rest.

Cross-cutting, not yet built (Phase 7 in `medai_architecture_graph.txt`):
observability (token/cost/latency), PHI-safe log redaction, and a full audit
log across every AI-authored write.

---

## 6. Key files to read, in order, to actually learn this

1. `app/knowledge/loaders.py`, `chunking.py`, `embeddings.py`, `vector_store.py`,
   `ingest.py`, `query_engine.py` — LlamaIndex ingestion + retrieval, start to finish.
2. `app/recommendations/retrieval.py`, `prompt.py`, `parser.py`, `chain.py` —
   RAG + LangChain: how a retrieved passage becomes a cited, structured draft.
3. `app/llm/interface.py` — the real/fake LLM split pattern used everywhere.
4. `app/approval/graph.py` — LangGraph basics: `StateGraph`, `interrupt()`,
   conditional edges, `MongoDBSaver` checkpointing.
5. `app/agent/graph.py`, `tools.py` — LangGraph tool-calling agent, prebuilt
   `tools_condition`/`ToolNode`, and a hand-rolled safety-routing edge on top.
6. `medai_architecture_graph.txt` — the one-page picture all of the above hangs off.
