"""Sync Mongo access for the LangGraph approval graph.

LangGraph (and `MongoDBSaver`, its Mongo-backed checkpointer) is sync-first —
there's no async Mongo checkpointer available yet. Rather than fight that, the
approval graph runs as a self-contained sync unit (idiomatic for LangGraph),
wrapped in `asyncio.to_thread` at the FastAPI boundary, same as the boto3 SQS
client elsewhere in this app. This module is that graph's Mongo access,
separate from the async Motor client the rest of the app uses.
"""

import pymongo
from pymongo.database import Database

from app.config import settings

_client: pymongo.MongoClient | None = None


def get_sync_client() -> pymongo.MongoClient:
    global _client
    if _client is None:
        _client = pymongo.MongoClient(settings.mongo_uri)
    return _client


def get_sync_database() -> Database:
    return get_sync_client()[settings.mongo_db_name]


def close_sync_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
