import uuid
from datetime import datetime, timezone

import chromadb
import pytest
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.domain.lab_result import LabResult, ReferenceRange
from app.knowledge.embeddings import FakeEmbedding
from app.recommendations.retrieval import build_context_passages


@pytest.fixture
def test_index():
    nodes = [
        TextNode(
            text="Adults with type 2 diabetes should have glucose checked periodically.",
            metadata={"source_type": "guideline", "source_title": "Diabetes Guideline"},
        ),
        TextNode(
            text="Glucose reference range is 70-99 mg/dL.",
            metadata={"source_type": "reference_range", "source_title": "Glucose reference range"},
        ),
        TextNode(
            text="Warfarin and Aspirin together increase bleeding risk.",
            metadata={"source_type": "drug_interaction", "source_title": "Warfarin / Aspirin interaction"},
        ),
    ]
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(f"test_retrieval_{uuid.uuid4().hex}")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=FakeEmbedding())


@pytest.fixture
def abnormal_glucose_result() -> LabResult:
    return LabResult(
        patient_id="p-123",
        external_order_id="ord-456",
        test_code="GLU",
        test_name="Glucose",
        value=250.0,
        unit="mg/dL",
        reference_range=ReferenceRange(low=70, high=99),
        is_abnormal=True,
        collected_at=datetime.now(timezone.utc),
        resulted_at=datetime.now(timezone.utc),
        source_lab="Quest Diagnostics",
        raw_payload={},
    )


def test_build_context_passages_returns_passages(abnormal_glucose_result, test_index):
    passages = build_context_passages(abnormal_glucose_result, top_k=5, index=test_index)

    assert len(passages) > 0
    assert all(p.text for p in passages)


def test_build_context_passages_respects_top_k(abnormal_glucose_result, test_index):
    passages = build_context_passages(abnormal_glucose_result, top_k=1, index=test_index)

    assert len(passages) == 1
