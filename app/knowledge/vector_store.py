"""Vector store behind an interface: `ChromaStore` (default, local, zero
external deps) or `AtlasVectorStore` (MongoDB Atlas Vector Search) selected by
`settings.vector_store_backend`. Both return a LlamaIndex `BasePydanticVectorStore`
so the rest of the app (indexing, retrieval) never branches on backend.
"""

from llama_index.core.vector_stores.types import BasePydanticVectorStore

from app.config import settings


def _chroma_store() -> BasePydanticVectorStore:
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = client.get_or_create_collection(settings.chroma_collection_name)
    return ChromaVectorStore(chroma_collection=collection)


def _atlas_store() -> BasePydanticVectorStore:
    from pymongo import AsyncMongoClient, MongoClient

    from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch

    if not settings.atlas_mongo_uri:
        raise RuntimeError("ATLAS_MONGO_URI must be set when VECTOR_STORE_BACKEND=atlas")

    return MongoDBAtlasVectorSearch(
        mongodb_client=MongoClient(settings.atlas_mongo_uri),
        async_mongodb_client=AsyncMongoClient(settings.atlas_mongo_uri),
        db_name=settings.atlas_db_name,
        collection_name=settings.atlas_collection_name,
        vector_index_name=settings.atlas_vector_index_name,
    )


def get_vector_store() -> BasePydanticVectorStore:
    if settings.vector_store_backend == "atlas":
        return _atlas_store()
    return _chroma_store()
