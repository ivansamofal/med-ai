from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central app configuration, loaded from environment / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_env: str = "local"
    log_level: str = "INFO"

    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db_name: str = "medai"

    aws_region: str = "us-east-1"
    # Set to the LocalStack endpoint locally; leave unset (None) to hit real AWS.
    aws_endpoint_url: str | None = "http://localhost:4566"
    aws_access_key_id: str = "test"
    aws_secret_access_key: str = "test"

    sqs_lab_result_queue_name: str = "lab-result-created"

    # Knowledge base (Phase 2)
    vector_store_backend: str = "chroma"  # "chroma" | "atlas"
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "medai_knowledge_base"

    atlas_mongo_uri: str | None = None
    atlas_db_name: str = "medai"
    atlas_collection_name: str = "knowledge_base"
    atlas_vector_index_name: str = "vector_index"

    embedding_backend: str = "fastembed"  # "fastembed" | "fake"
    fastembed_model_name: str = "BAAI/bge-small-en-v1.5"


settings = Settings()
