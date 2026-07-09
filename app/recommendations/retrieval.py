"""TODO (Phase 3, yours to implement): retrieval strategy for the
recommendation chain.
"""

from llama_index.core import VectorStoreIndex

from app.domain.lab_result import LabResult
from app.knowledge.query_engine import RetrievedPassage, query_knowledge_base


def build_context_passages(
    lab_result: LabResult,
    top_k: int = 5,
    index: VectorStoreIndex | None = None,
) -> list[RetrievedPassage]:
    """Decide what to ask the knowledge base for, given one specific lab result.

    WHAT: Call `query_knowledge_base` with a query string — and, if you choose,
    `source_types` — built from this `lab_result`, and return the retrieved
    passages that will ground the recommendation.

    WHY: This is a retrieval-strategy decision, not plumbing. Phase 2 built a
    generic retriever, but nothing yet decides *what* to ask it for a given lab
    result. Query too narrowly (e.g. just the test code) and you might miss
    the reference-range passage entirely; query too broadly and irrelevant
    guideline chunks crowd out the ones that matter. What you put in the query
    string, and which `source_types` you include (or don't), directly shapes
    what the LLM can ground its answer in.

    Hints:
    - `lab_result.test_name`, `.value`, `.unit`, `.is_abnormal` are your
      strongest signals for a query string that actually matches relevant
      guideline/reference-range text.
    - You don't have to restrict `source_types` — but consider whether always
      including `drug_interaction` passages for every lab result makes sense,
      versus only surfacing them when relevant.
    - `index` is passed straight through to `query_knowledge_base` — it exists
      so tests can inject a small fixture index instead of the real one.
    """
    raise NotImplementedError("TODO(phase-3): implement build_context_passages retrieval strategy")
