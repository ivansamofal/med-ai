import uuid
from datetime import datetime, timezone

import chromadb
import pytest
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.agent.tools import build_lab_history_context
from app.db.mongo import LabResultRepository, get_database
from app.domain.lab_result import LabResult, ReferenceRange
from app.knowledge.embeddings import FakeEmbedding


@pytest.fixture
def test_index():
    nodes = [
        TextNode(
            text="Glucose reference range is 70-99 mg/dL. Critical if below 50 or above 400 mg/dL.",
            metadata={"source_type": "reference_range", "source_title": "Glucose reference range"},
        ),
        TextNode(
            text="Adults with severe hyperglycemia need urgent evaluation.",
            metadata={"source_type": "guideline", "source_title": "Diabetes Guideline"},
        ),
    ]
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(f"test_tools_kb_{uuid.uuid4().hex}")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=FakeEmbedding())


async def _insert_lab_result(**overrides) -> str:
    db = get_database()
    defaults = dict(
        patient_id="p-123",
        external_order_id="ord-1",
        test_code="GLU",
        test_name="Glucose",
        value=105.0,
        unit="mg/dL",
        reference_range=ReferenceRange(low=70, high=99),
        is_abnormal=True,
        collected_at=datetime.now(timezone.utc),
        resulted_at=datetime.now(timezone.utc),
        source_lab="Quest Diagnostics",
        raw_payload={},
    )
    defaults.update(overrides)
    return await LabResultRepository(db).insert(LabResult(**defaults))


async def test_build_lab_history_context_includes_all_patient_results(test_index):
    await _insert_lab_result(value=105.0)
    await _insert_lab_result(test_code="HGB", test_name="Hemoglobin", value=15.0, is_abnormal=False)

    context = build_lab_history_context("p-123", index=test_index)

    assert context["patient_id"] == "p-123"
    assert len(context["results"]) == 2


async def test_build_lab_history_context_cites_abnormal_results(test_index):
    await _insert_lab_result(value=105.0, is_abnormal=True)

    context = build_lab_history_context("p-123", index=test_index)

    assert context["results"][0]["citations"]


async def test_build_lab_history_context_skips_citations_for_normal_results(test_index):
    await _insert_lab_result(value=85.0, is_abnormal=False)

    context = build_lab_history_context("p-123", index=test_index)

    assert context["results"][0]["citations"] == []


async def test_build_lab_history_context_flags_critical_value(test_index):
    await _insert_lab_result(value=420.0, is_abnormal=True)

    context = build_lab_history_context("p-123", index=test_index)

    assert context["red_flag"] is True
    assert context["results"][0]["red_flag"] is not None


async def test_build_lab_history_context_no_red_flag_for_normal_history(test_index):
    await _insert_lab_result(value=90.0, is_abnormal=False)

    context = build_lab_history_context("p-123", index=test_index)

    assert context["red_flag"] is False


async def test_build_lab_history_context_only_returns_requested_patient(test_index):
    await _insert_lab_result(patient_id="p-123")
    await _insert_lab_result(patient_id="p-other")

    context = build_lab_history_context("p-123", index=test_index)

    assert len(context["results"]) == 1
