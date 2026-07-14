import uuid

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.eval.models import GoldenQuestion
from app.eval.ragas_eval import evaluate_with_ragas
from app.knowledge.embeddings import FakeEmbedding


def _build_index(nodes: list[TextNode]) -> VectorStoreIndex:
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(f"test_ragas_{uuid.uuid4().hex}")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=FakeEmbedding())


def test_evaluate_with_ragas_scores_a_correct_reference_higher_than_a_wrong_one():
    good_index = _build_index(
        [TextNode(text="Glucose reference range is 70-99 mg/dL.", metadata={"source_type": "reference_range"})]
    )
    bad_index = _build_index(
        [TextNode(text="Warfarin and Aspirin together increase bleeding risk.", metadata={"source_type": "drug_interaction"})]
    )
    question = GoldenQuestion(
        id="q1",
        question="glucose range",
        expected_source_titles=["Glucose reference range"],
        required_keywords=["70", "99"],
        reference_context="Glucose reference range is 70-99 mg/dL.",
    )

    good_summary = evaluate_with_ragas([question], index=good_index, top_k=1)
    bad_summary = evaluate_with_ragas([question], index=bad_index, top_k=1)

    assert good_summary.mean_context_precision > bad_summary.mean_context_precision
    assert good_summary.mean_context_recall > bad_summary.mean_context_recall


def test_evaluate_with_ragas_returns_one_result_per_question():
    index = _build_index([TextNode(text="Glucose reference range is 70-99 mg/dL.", metadata={})])
    questions = [
        GoldenQuestion(id="q1", question="a", expected_source_titles=[], required_keywords=["70"]),
        GoldenQuestion(id="q2", question="b", expected_source_titles=[], required_keywords=["99"]),
    ]

    summary = evaluate_with_ragas(questions, index=index, top_k=1)

    assert [r.id for r in summary.results] == ["q1", "q2"]
