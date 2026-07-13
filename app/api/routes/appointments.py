from fastapi import APIRouter, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.api.schemas.dashboard import AppointmentResponse
from app.db.mongo import AppointmentRepository, get_database

router = APIRouter()


@router.get("/appointments", response_model=list[AppointmentResponse])
async def list_appointments(db: AsyncIOMotorDatabase = Depends(get_database)) -> list[AppointmentResponse]:
    documents = await AppointmentRepository(db).list_recent()
    return [
        AppointmentResponse(
            id=document["id"],
            patient_id=document["patient_id"],
            doctor=document["doctor"],
            scheduled_at=document["scheduled_at"].isoformat(),
            duration_minutes=document["duration_minutes"],
            status=document["status"],
            created_at=document["created_at"].isoformat(),
        )
        for document in documents
    ]
