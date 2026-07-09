from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import settings
from app.domain.lab_result import LabResult
from app.domain.recommendation import Recommendation

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.mongo_uri)
    return _client


def get_database() -> AsyncIOMotorDatabase:
    return get_client()[settings.mongo_db_name]


def close_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None


class LabResultRepository:
    """Persistence for normalized lab results."""

    COLLECTION = "lab_results"

    def __init__(self, db: AsyncIOMotorDatabase):
        self._collection = db[self.COLLECTION]

    async def insert(self, lab_result: LabResult) -> str:
        document = lab_result.model_dump()
        result = await self._collection.insert_one(document)
        return str(result.inserted_id)

    async def get_by_id(self, lab_result_id: str) -> LabResult:
        document = await self._collection.find_one({"_id": ObjectId(lab_result_id)})
        if document is None:
            raise ValueError(f"No lab result found for id={lab_result_id}")
        document.pop("_id")
        return LabResult(**document)


class RecommendationRepository:
    """Persistence for AI-authored recommendations."""

    COLLECTION = "recommendations"

    def __init__(self, db: AsyncIOMotorDatabase):
        self._collection = db[self.COLLECTION]

    async def insert(self, recommendation: Recommendation) -> str:
        document = recommendation.model_dump()
        result = await self._collection.insert_one(document)
        return str(result.inserted_id)
