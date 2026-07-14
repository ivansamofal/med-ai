"""OCR document pipeline route: upload a scanned business document (image),
run it through OCR -> entity extraction -> validation -> persistence
(`app.ocr.pipeline.process_document`), and read back what's been processed.
Not part of the 8-phase plan — added to cover OCR/document-entity-extraction,
which the original spec never touched.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, UploadFile
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.api.schemas.dashboard import DocumentResponse
from app.db.mongo import ScannedDocumentRepository, get_database
from app.ocr.pipeline import process_document

router = APIRouter()


@router.post("/documents/ocr", response_model=DocumentResponse, status_code=201)
async def ocr_document(
    file: UploadFile, db: AsyncIOMotorDatabase = Depends(get_database)
) -> DocumentResponse:
    file_bytes = await file.read()
    document_id = await process_document(file_bytes, filename=file.filename or "upload")

    document = await ScannedDocumentRepository(db).get_by_id(document_id)
    return DocumentResponse(
        id=document_id,
        filename=document.filename,
        extracted_entities=document.extracted_entities,
        validation_issues=document.validation_issues,
        status=document.status,
        created_at=document.created_at.isoformat(),
    )


@router.get("/documents", response_model=list[DocumentResponse])
async def list_documents(db: AsyncIOMotorDatabase = Depends(get_database)) -> list[DocumentResponse]:
    documents = await ScannedDocumentRepository(db).list_recent()
    return [
        DocumentResponse(
            id=document["id"],
            filename=document["filename"],
            extracted_entities=document["extracted_entities"],
            validation_issues=document["validation_issues"],
            status=document["status"],
            created_at=document["created_at"].isoformat(),
        )
        for document in documents
    ]
