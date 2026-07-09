import uuid

import chromadb
import pytest
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.knowledge.embeddings import FakeEmbedding
from app.knowledge.query_engine import RetrievedPassage, query_knowledge_base


@pytest.fixture
def test_index():
    nodes = [
        TextNode(
            text="Adults with type 2 diabetes should have glucose checked periodically.",
            metadata={"source_type": "guideline", "source_title": "Diabetes Guideline"},
        ),
        TextNode(
            text="Warfarin and Aspirin together increase bleeding risk.",
            metadata={"source_type": "drug_interaction", "source_title": "Warfarin / Aspirin interaction"},
        ),
        TextNode(
            text="Glucose reference range is 70-99 mg/dL.",
            metadata={"source_type": "reference_range", "source_title": "Glucose reference range"},
        ),
        TextNode(
            text="Hypertension patients on ACE inhibitors need potassium monitoring.",
            metadata={"source_type": "guideline", "source_title": "Hypertension Guideline"},
        ),
    ]
    client = chromadb.EphemeralClient()
    # chromadb's EphemeralClient shares an in-process store across instances
    # for a given collection name, so tests need a unique name to stay isolated.
    collection = client.get_or_create_collection(f"test_knowledge_base_{uuid.uuid4().hex}")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=FakeEmbedding())


def test_query_knowledge_base_respects_top_k(test_index):
    results = query_knowledge_base("glucose monitoring", top_k=2, index=test_index)

    assert len(results) == 2
    assert all(isinstance(r, RetrievedPassage) for r in results)


def test_query_knowledge_base_returns_citations(test_index):
    results = query_knowledge_base("glucose monitoring", top_k=4, index=test_index)

    assert len(results) == 4
    for r in results:
        assert r.source_title
        assert r.source_type
        assert r.text
        assert isinstance(r.score, float)


def test_query_knowledge_base_filters_by_source_type(test_index):
    results = query_knowledge_base(
        "glucose", top_k=4, source_types=["drug_interaction"], index=test_index
    )

    assert len(results) == 1
    assert results[0].source_type == "drug_interaction"
    assert "Warfarin" in results[0].text


def test_query_knowledge_base_filters_by_multiple_source_types(test_index):
    results = query_knowledge_base(
        "glucose", top_k=4, source_types=["guideline", "reference_range"], index=test_index
    )

    assert len(results) == 3
    assert all(r.source_type in ("guideline", "reference_range") for r in results)
