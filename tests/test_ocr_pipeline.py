from app.db.mongo import ScannedDocumentRepository, get_database
from app.ocr.entities import ExtractedDocumentEntities
from app.ocr.interface import FakeOcrEngine
from app.ocr.pipeline import process_document
from app.ocr.validation import validate_entities


async def test_process_document_persists_a_clean_document():
    document_id = await process_document(b"irrelevant-bytes", "form.png", ocr_engine=FakeOcrEngine())

    db = get_database()
    document = await ScannedDocumentRepository(db).get_by_id(document_id)

    assert document.filename == "form.png"
    assert document.status == "clean"
    assert document.validation_issues == []
    assert document.extracted_entities["patient_name"] == "Jane Doe"
    assert document.extracted_entities["test_codes"] == ["GLU", "HBA1C"]


def test_validate_entities_flags_unknown_test_code_and_missing_field():
    entities = ExtractedDocumentEntities(
        patient_name="",
        document_date="2026-06-01",
        ordering_physician="Dr. Smith",
        test_codes=["NOT_A_REAL_CODE"],
    )

    issues = validate_entities(entities)

    assert any("missing patient name" in issue for issue in issues)
    assert any("unknown test code" in issue for issue in issues)


def test_validate_entities_flags_future_and_unparseable_dates():
    future = validate_entities(
        ExtractedDocumentEntities(
            patient_name="Jane Doe",
            document_date="2099-01-01",
            ordering_physician="Dr. Smith",
            test_codes=["GLU"],
        )
    )
    unparseable = validate_entities(
        ExtractedDocumentEntities(
            patient_name="Jane Doe",
            document_date="not-a-date",
            ordering_physician="Dr. Smith",
            test_codes=["GLU"],
        )
    )

    assert any("future" in issue for issue in future)
    assert any("unparseable" in issue for issue in unparseable)


def test_validate_entities_clean_case_has_no_issues():
    issues = validate_entities(
        ExtractedDocumentEntities(
            patient_name="Jane Doe",
            document_date="2026-06-01",
            ordering_physician="Dr. Smith",
            test_codes=["GLU", "HBA1C"],
        )
    )

    assert issues == []
