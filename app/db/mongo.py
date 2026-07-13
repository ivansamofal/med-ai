from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import settings
from app.domain.appointment import Appointment
from app.domain.audit_log import AuditLogEntry
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

    async def list_recent(self, limit: int = 50) -> list[dict]:
        """Dashboard read: most recent lab results, each carrying its `id`
        alongside the validated `LabResult` fields."""
        cursor = self._collection.find().sort("resulted_at", -1).limit(limit)
        results = []
        async for document in cursor:
            document_id = str(document.pop("_id"))
            results.append({"id": document_id, **LabResult(**document).model_dump()})
        return results


class RecommendationRepository:
    """Persistence for AI-authored recommendations."""

    COLLECTION = "recommendations"

    def __init__(self, db: AsyncIOMotorDatabase):
        self._collection = db[self.COLLECTION]

    async def insert(self, recommendation: Recommendation) -> str:
        document = recommendation.model_dump()
        result = await self._collection.insert_one(document)
        return str(result.inserted_id)

    async def get_by_id(self, recommendation_id: str) -> Recommendation:
        document = await self._collection.find_one({"_id": ObjectId(recommendation_id)})
        if document is None:
            raise ValueError(f"No recommendation found for id={recommendation_id}")
        document.pop("_id")
        return Recommendation(**document)

    async def list_recent(self, limit: int = 50) -> list[dict]:
        cursor = self._collection.find().sort("created_at", -1).limit(limit)
        results = []
        async for document in cursor:
            document_id = str(document.pop("_id"))
            results.append({"id": document_id, **Recommendation(**document).model_dump()})
        return results


class AppointmentRepository:
    """Read access for appointments (written by the chat agent's sync tools,
    via `app.agent.scheduling`) — Motor can read the same collection a
    pymongo write went to, so this stays async like the rest of the FastAPI
    read layer, no `asyncio.to_thread` needed."""

    COLLECTION = "appointments"

    def __init__(self, db: AsyncIOMotorDatabase):
        self._collection = db[self.COLLECTION]

    async def list_recent(self, limit: int = 50) -> list[dict]:
        cursor = self._collection.find().sort("scheduled_at", -1).limit(limit)
        results = []
        async for document in cursor:
            document_id = str(document.pop("_id"))
            results.append({"id": document_id, **Appointment(**document).model_dump()})
        return results


class AuditLogRepository:
    """Read access for the approval graph's audit trail (written by
    `app.approval.graph.write_audit_log`)."""

    COLLECTION = "audit_log"

    def __init__(self, db: AsyncIOMotorDatabase):
        self._collection = db[self.COLLECTION]

    async def list_recent(self, limit: int = 50) -> list[dict]:
        cursor = self._collection.find().sort("created_at", -1).limit(limit)
        results = []
        async for document in cursor:
            document_id = str(document.pop("_id"))
            results.append({"id": document_id, **AuditLogEntry(**document).model_dump()})
        return results
