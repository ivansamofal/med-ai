"""OCR document pipeline: OCR -> extract entities -> validate -> persist.
Mirrors Phase 1's ingest->normalize and Phase 3's retrieve->prompt->generate
shapes, one level up: a single synchronous entry point a route (or a script)
calls with raw bytes and gets back a persisted `ScannedDocument` id.
"""

from __future__ import annotations

from app.db.mongo import ScannedDocumentRepository, get_database
from app.domain.scanned_document import ScannedDocument
from app.ocr.entities import extract_entities
from app.ocr.interface import OcrEngine, get_ocr_engine
from app.ocr.validation import validate_entities


async def process_document(file_bytes: bytes, filename: str, ocr_engine: OcrEngine | None = None) -> str:
    """Run the OCR pipeline for one uploaded document and return the new
    `ScannedDocument`'s id. `ocr_engine` defaults to `get_ocr_engine()` — the
    param exists so tests can inject `FakeOcrEngine` explicitly."""
    engine = ocr_engine if ocr_engine is not None else get_ocr_engine()
    raw_text = engine.extract_text(file_bytes)

    entities = await extract_entities(raw_text)
    issues = validate_entities(entities)

    document = ScannedDocument(
        filename=filename,
        raw_text=raw_text,
        extracted_entities=entities.model_dump(),
        validation_issues=issues,
        status="clean" if not issues else "needs_review",
    )
    db = get_database()
    return await ScannedDocumentRepository(db).insert(document)
