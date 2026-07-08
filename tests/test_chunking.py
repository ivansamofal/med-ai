from llama_index.core import Document
from llama_index.core.schema import TextNode

from app.knowledge.chunking import build_guideline_nodes


def _long_guideline_document() -> Document:
    paragraph = (
        "Adults with type 2 diabetes should have fasting plasma glucose and "
        "hemoglobin A1c checked at regular intervals. A fasting glucose "
        "reference range of 70-99 mg/dL is considered normal. Values "
        "persistently above 126 mg/dL are consistent with diabetes. "
    )
    return Document(
        text=paragraph * 20,  # long enough to force multiple chunks
        metadata={"source_type": "guideline", "source_title": "Diabetes Guideline"},
    )


def test_build_guideline_nodes_splits_long_document_into_multiple_nodes():
    nodes = build_guideline_nodes([_long_guideline_document()])

    assert all(isinstance(node, TextNode) for node in nodes)
    assert len(nodes) > 1


def test_build_guideline_nodes_propagates_metadata_to_every_chunk():
    nodes = build_guideline_nodes([_long_guideline_document()])

    assert len(nodes) > 0
    for node in nodes:
        assert node.metadata.get("source_title") == "Diabetes Guideline"
        assert node.metadata.get("source_type") == "guideline"
        assert node.text.strip() != ""
