"""Build (or reopen) the knowledge-base index: load the three sources, chunk
guidelines, embed everything, and write it into the configured `VectorStore`.
"""

from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import Document, TextNode

from app.knowledge.chunking import build_guideline_nodes
from app.knowledge.embeddings import get_embedding_model
from app.knowledge.loaders import (
    load_drug_interaction_documents,
    load_guideline_documents,
    load_lab_reference_range_documents,
)
from app.knowledge.vector_store import get_vector_store

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge_base"


def _documents_to_nodes(documents: list[Document]) -> list[TextNode]:
    return [TextNode(text=doc.text, metadata=doc.metadata) for doc in documents]


def build_index(data_dir: Path = DATA_DIR) -> VectorStoreIndex:
    """Load every knowledge-base source from `data_dir` and (re)build the index."""
    guideline_docs = load_guideline_documents(data_dir / "guidelines")
    drug_docs = load_drug_interaction_documents(data_dir / "drug_interactions.json")
    range_docs = load_lab_reference_range_documents(data_dir / "lab_reference_ranges.csv")

    nodes = build_guideline_nodes(guideline_docs)
    nodes += _documents_to_nodes(drug_docs)
    nodes += _documents_to_nodes(range_docs)

    storage_context = StorageContext.from_defaults(vector_store=get_vector_store())
    return VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=get_embedding_model(),
    )


def get_index() -> VectorStoreIndex:
    """Reopen the index bound to the existing (already-ingested) vector store."""
    return VectorStoreIndex.from_vector_store(
        get_vector_store(), embed_model=get_embedding_model()
    )


if __name__ == "__main__":
    index = build_index()
    print(f"Ingested knowledge base into '{index.vector_store.__class__.__name__}'.")
