from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch

from app.config import settings
from app.knowledge.vector_store import get_vector_store


def test_get_vector_store_defaults_to_chroma(tmp_path):
    settings.chroma_persist_dir = str(tmp_path / "chroma")
    store = get_vector_store()
    assert isinstance(store, ChromaVectorStore)


def test_get_vector_store_selects_atlas_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "vector_store_backend", "atlas")
    monkeypatch.setattr(settings, "atlas_mongo_uri", "mongodb://localhost:27017")

    store = get_vector_store()

    assert isinstance(store, MongoDBAtlasVectorSearch)

    monkeypatch.setattr(settings, "vector_store_backend", "chroma")
