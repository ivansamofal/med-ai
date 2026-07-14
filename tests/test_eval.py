import uuid

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.eval.models import GoldenQuestion, GoldenRedFlagCase
from app.eval.qa_eval import evaluate_qa, load_golden_questions
from app.eval.redflag_eval import evaluate_red_flags, load_golden_red_flags
from app.knowledge.embeddings import FakeEmbedding
from app.knowledge.reference_ranges import CriticalThreshold


def test_load_golden_questions_covers_every_source_type():
    questions = load_golden_questions()

    assert len(questions) >= 15
    assert {t for q in questions for t in (q.source_types or [])} == {
        "guideline",
        "drug_interaction",
        "reference_range",
        "icd10_code",
        "medical_reference",
    }


def test_load_golden_red_flags_covers_both_outcomes():
    cases = load_golden_red_flags()

    assert len(cases) >= 15
    assert any(c.expect_red_flag for c in cases)
    assert any(not c.expect_red_flag for c in cases)


def _fixture_index() -> VectorStoreIndex:
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
    collection = client.get_or_create_collection(f"test_eval_kb_{uuid.uuid4().hex}")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=FakeEmbedding())


def test_evaluate_qa_passes_when_retrieval_cites_and_grounds_the_right_fact():
    question = GoldenQuestion(
        id="q1",
        question="Glucose reference range is 70-99 mg/dL. Critical if below 50 or above 400 mg/dL.",
        expected_source_titles=["Glucose reference range"],
        required_keywords=["50", "400"],
    )

    results = evaluate_qa([question], index=_fixture_index())

    assert results[0].passed


def test_evaluate_qa_fails_on_wrong_citation():
    question = GoldenQuestion(
        id="q2",
        question="Glucose reference range is 70-99 mg/dL. Critical if below 50 or above 400 mg/dL.",
        expected_source_titles=["Some Other Document"],
        required_keywords=["50"],
    )

    results = evaluate_qa([question], index=_fixture_index())

    assert not results[0].passed
    assert "Some Other Document" in results[0].detail


def test_evaluate_qa_fails_on_missing_grounding_keyword():
    question = GoldenQuestion(
        id="q3",
        question="Glucose reference range is 70-99 mg/dL. Critical if below 50 or above 400 mg/dL.",
        expected_source_titles=["Glucose reference range"],
        required_keywords=["9999"],
    )

    results = evaluate_qa([question], index=_fixture_index())

    assert not results[0].passed
    assert "9999" in results[0].detail


THRESHOLDS = {"K": CriticalThreshold(test_code="K", critical_low=2.5, critical_high=6.5)}


def test_evaluate_red_flags_passes_on_matching_expectation():
    case = GoldenRedFlagCase(
        id="c1", test_code="K", test_name="Potassium", value=7.0, unit="mmol/L", expect_red_flag=True
    )

    results = evaluate_red_flags([case], thresholds=THRESHOLDS)

    assert results[0].passed


def test_evaluate_red_flags_fails_on_mismatched_expectation():
    case = GoldenRedFlagCase(
        id="c2", test_code="K", test_name="Potassium", value=4.0, unit="mmol/L", expect_red_flag=True
    )

    results = evaluate_red_flags([case], thresholds=THRESHOLDS)

    assert not results[0].passed
    assert "expected red_flag=True" in results[0].detail


def test_golden_red_flags_all_pass_against_real_thresholds():
    """The authored golden set is a claim about production's real CSV
    thresholds, not a fixture — this is what `make eval` actually checks."""
    cases = load_golden_red_flags()

    results = evaluate_red_flags(cases)

    failures = [r for r in results if not r.passed]
    assert failures == []
