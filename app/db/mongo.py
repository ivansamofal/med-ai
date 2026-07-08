from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import settings
from app.domain.lab_result import LabResult

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
