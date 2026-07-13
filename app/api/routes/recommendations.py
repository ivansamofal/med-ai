from fastapi import APIRouter, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.api.schemas.dashboard import RecommendationResponse
from app.db.mongo import RecommendationRepository, get_database

router = APIRouter()


@router.get("/recommendations", response_model=list[RecommendationResponse])
async def list_recommendations(
    db: AsyncIOMotorDatabase = Depends(get_database),
) -> list[RecommendationResponse]:
    documents = await RecommendationRepository(db).list_recent()
    return [
        RecommendationResponse(
            id=document["id"],
            lab_result_id=document["lab_result_id"],
            patient_id=document["patient_id"],
            recommendation_text=document["recommendation_text"],
            citations=document["citations"],
            status=document["status"],
            created_at=document["created_at"].isoformat(),
        )
        for document in documents
    ]
