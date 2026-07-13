from fastapi import APIRouter, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.api.schemas.dashboard import AuditLogResponse
from app.db.mongo import AuditLogRepository, get_database

router = APIRouter()


@router.get("/audit-log", response_model=list[AuditLogResponse])
async def list_audit_log(db: AsyncIOMotorDatabase = Depends(get_database)) -> list[AuditLogResponse]:
    documents = await AuditLogRepository(db).list_recent()
    return [
        AuditLogResponse(
            id=document["id"],
            recommendation_id=document["recommendation_id"],
            action=document["action"],
            actor=document["actor"],
            reason=document["reason"],
            created_at=document["created_at"].isoformat(),
        )
        for document in documents
    ]
