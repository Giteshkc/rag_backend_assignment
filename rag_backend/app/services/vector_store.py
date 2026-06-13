"""Qdrant vector store operations: upsert and query."""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.qdrant import get_qdrant_client

logger = get_logger(__name__)


@dataclass(slots=True)
class SearchResult:
    chunk_id: str
    text: str
    score: float
    document_id: str
    filename: str
    chunk_index: int


async def upsert_chunks(
    document_id: str,
    filename: str,
    chunk_ids: list[str],
    texts: list[str],
    embeddings: list[list[float]],
) -> None:
    """Upsert a batch of chunk vectors into Qdrant."""
    settings = get_settings()
    client = get_qdrant_client()

    points = [
        PointStruct(
            id=chunk_id,
            vector=embedding,
            payload={
                "document_id": document_id,
                "filename": filename,
                "text": text,
                "chunk_index": idx,
            },
        )
        for idx, (chunk_id, text, embedding) in enumerate(
            zip(chunk_ids, texts, embeddings)
        )
    ]

    await client.upsert(
        collection_name=settings.qdrant_collection,
        points=points,
    )
    logger.info("Upserted %d vectors for document '%s'", len(points), document_id)


async def similarity_search(
    query_vector: list[float],
    top_k: int = 5,
    document_id: str | None = None,
) -> list[SearchResult]:
    """
    Perform ANN search in Qdrant.
    Optionally filter by document_id for focused retrieval.
    """
    settings = get_settings()
    client = get_qdrant_client()

    query_filter: Filter | None = None
    if document_id:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        )

    hits = await client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )

    return [
        SearchResult(
            chunk_id=str(hit.id),
            text=hit.payload.get("text", ""),
            score=hit.score,
            document_id=hit.payload.get("document_id", ""),
            filename=hit.payload.get("filename", ""),
            chunk_index=hit.payload.get("chunk_index", 0),
        )
        for hit in hits
    ]
