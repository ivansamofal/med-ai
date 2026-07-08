from fastapi import APIRouter, Depends, HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import ValidationError

from app.api.schemas.lab_result import LabResultIngestRequest, LabResultResponse
from app.db.mongo import LabResultRepository, get_database
from app.domain.lab_result import normalize_lab_result
from app.events.sqs import publish_lab_result_created
from app.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/lab-results", response_model=LabResultResponse, status_code=201)
async def ingest_lab_result(
    payload: LabResultIngestRequest,
    db: AsyncIOMotorDatabase = Depends(get_database),
) -> LabResultResponse:
    try:
        lab_result = normalize_lab_result(payload.model_dump())
    except (ValueError, ValidationError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=f"Malformed lab result payload: {exc}") from exc

    repository = LabResultRepository(db)
    lab_result_id = await repository.insert(lab_result)

    await publish_lab_result_created(
        lab_result_id=lab_result_id,
        patient_id=lab_result.patient_id,
        is_abnormal=lab_result.is_abnormal,
    )

    logger.info(
        "lab_result_ingested",
        lab_result_id=lab_result_id,
        test_code=lab_result.test_code,
        is_abnormal=lab_result.is_abnormal,
    )

    return LabResultResponse(
        id=lab_result_id,
        patient_id=lab_result.patient_id,
        test_code=lab_result.test_code,
        test_name=lab_result.test_name,
        value=lab_result.value,
        unit=lab_result.unit,
        is_abnormal=lab_result.is_abnormal,
        resulted_at=lab_result.resulted_at.isoformat(),
    )
