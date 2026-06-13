from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # OpenAI
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    embedding_dimensions: int = 1536  # text-embedding-3-small

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "rag_documents"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_chat_ttl_seconds: int = 86400

    # PostgreSQL
    database_url: str = "postgresql+asyncpg://raguser:ragpass@localhost:5432/ragdb"

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    max_file_size_mb: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()
