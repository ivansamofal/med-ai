"""TODO (Phase 2, yours to implement): guideline chunking strategy."""

from llama_index.core import Document
from llama_index.core.schema import TextNode


def build_guideline_nodes(documents: list[Document]) -> list[TextNode]:
    """Split each guideline `Document` into retrieval-sized `TextNode`s.

    WHAT: Guideline PDFs load as one `Document` per source file (potentially
    several pages of prose). Split each into smaller `TextNode`s so retrieval
    can return a focused passage instead of a whole document, while carrying
    forward `source_title`/`source_type` metadata onto every child node so it
    can still be cited back to where it came from.

    WHY: This is a core RAG trade-off. One vector per whole document makes
    retrieval imprecise — a single match drags in every unrelated paragraph in
    that guideline, and "citing" it just means "see the whole guideline."
    Chunk too small, on the other hand, and a passage loses the context it
    needs to stand on its own (e.g. a sentence without the condition it
    applies to). Chunk size/overlap is a knob you'll want to feel out.

    Hints:
    - `llama_index.core.node_parser.SentenceSplitter(chunk_size=..., chunk_overlap=...)`
      splits on sentence boundaries instead of mid-word/mid-sentence.
    - `splitter.get_nodes_from_documents(documents)` does the split and should
      propagate each parent `Document`'s `.metadata` onto its child nodes —
      verify that's actually happening for `source_title`/`source_type`.
    - A few hundred tokens per chunk with modest overlap is a reasonable
      starting point for clinical guideline prose; there's no single right
      answer, tune against the failing test and your own judgment.
    """
    raise NotImplementedError("TODO(phase-2): implement build_guideline_nodes chunking")
