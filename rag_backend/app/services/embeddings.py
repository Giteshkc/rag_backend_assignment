"""OpenAI embedding generation with retry logic and batching."""

from __future__ import annotations

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_MAX_BATCH = 100  # OpenAI allows up to 2048 inputs but we keep batches small


def _get_client() -> AsyncOpenAI:
    settings = get_settings()
    return AsyncOpenAI(api_key=settings.openai_api_key)


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
async def _embed_batch(texts: list[str], model: str) -> list[list[float]]:
    client = _get_client()
    response = await client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.
    Automatically batches large inputs.
    """
    settings = get_settings()
    model = settings.openai_embedding_model

    if not texts:
        return []

    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), _MAX_BATCH):
        batch = texts[i : i + _MAX_BATCH]
        logger.debug("Embedding batch %d–%d / %d", i, i + len(batch), len(texts))
        embeddings = await _embed_batch(batch, model)
        all_embeddings.extend(embeddings)

    return all_embeddings


async def embed_query(text: str) -> list[float]:
    """Convenience wrapper for single-text embedding."""
    results = await embed_texts([text])
    return results[0]
