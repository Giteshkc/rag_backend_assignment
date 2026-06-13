from functools import lru_cache

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache
def get_qdrant_client() -> AsyncQdrantClient:
    settings = get_settings()
    kwargs: dict = {"url": settings.qdrant_url}
    if settings.qdrant_api_key:
        kwargs["api_key"] = settings.qdrant_api_key
    return AsyncQdrantClient(**kwargs)


async def ensure_collection_exists() -> None:
    """Create the Qdrant collection if it does not already exist."""
    settings = get_settings()
    client = get_qdrant_client()

    existing = await client.get_collections()
    names = [c.name for c in existing.collections]

    if settings.qdrant_collection not in names:
        await client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dimensions,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection '%s'", settings.qdrant_collection)
    else:
        logger.info("Qdrant collection '%s' already exists", settings.qdrant_collection)
