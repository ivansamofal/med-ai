"""The knowledge base's single public entry point: a plain callable so it can
later be wrapped as a LangChain Tool (Phase 3) without any adaptation layer.

Deliberately retrieval-only, no LLM synthesis: this returns passages + citations,
not a generated answer. Phase 3's chain is what turns retrieved context into
prose — LlamaIndex here owns ingestion + retrieval only, per the project's
division of labor between LlamaIndex/LangChain/LangGraph.
"""

from llama_index.core import VectorStoreIndex
from pydantic import BaseModel

from app.knowledge.ingest import get_index


class RetrievedPassage(BaseModel):
    text: str
    source_title: str
    source_type: str
    score: float


def query_knowledge_base(
    question: str,
    top_k: int = 5,
    source_types: list[str] | None = None,
    index: VectorStoreIndex | None = None,
) -> list[RetrievedPassage]:
    """TODO (Phase 2, yours to implement): the retrieval query.

    WHAT: Retrieve the `top_k` most relevant knowledge-base passages for
    `question`, optionally restricted to one or more `source_types`
    ("guideline", "drug_interaction", "reference_range"), each returned as a
    `RetrievedPassage` carrying its citation (`source_title`, `source_type`)
    and similarity `score`.

    WHY: This is the one place in the whole system where retrieval-quality
    trade-offs live — top_k, metadata filtering, any score cutoff — and they
    directly decide whether Phase 3's recommendation chain is grounded in real
    guideline text or retrieves noise it'll cite anyway. Get this wrong and
    every downstream citation is untrustworthy.

    Hints:
    - `index` is a LlamaIndex `VectorStoreIndex` (defaults to `get_index()` —
      the `index` param exists so tests can inject a small fixture index).
    - Build a retriever via `index.as_retriever(similarity_top_k=top_k, filters=...)`
      rather than a full query engine — we want raw passages, not an
      LLM-synthesized answer.
    - `llama_index.core.vector_stores.MetadataFilters` / `MetadataFilter` let
      you restrict by the `source_type` metadata field the loaders attach.
    - Each retrieved `NodeWithScore` has `.node.metadata["source_title"]` /
      `["source_type"]` and a `.score` — that's your `RetrievedPassage`.
    """
    raise NotImplementedError("TODO(phase-2): implement query_knowledge_base retrieval")
