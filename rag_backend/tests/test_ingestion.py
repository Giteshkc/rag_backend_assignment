"""
Tests for the Document Ingestion API.

Uses httpx AsyncClient with app in test mode.
External services (OpenAI, Qdrant, Redis, Postgres) are mocked.
"""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def txt_file() -> bytes:
    return b"The quick brown fox jumps over the lazy dog. " * 50


@pytest.fixture
def mock_db_session():
    session = AsyncMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock(scalars=lambda: MagicMock(all=lambda: [])))
    return session


# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_txt_fixed_chunking(txt_file, mock_db_session):
    with (
        patch("app.db.sql.get_db_session", return_value=mock_db_session),
        patch("app.services.embeddings.embed_texts", new_callable=AsyncMock) as mock_embed,
        patch("app.services.vector_store.upsert_chunks", new_callable=AsyncMock),
        patch("app.db.sql.create_tables", new_callable=AsyncMock),
        patch("app.db.qdrant.ensure_collection_exists", new_callable=AsyncMock),
    ):
        mock_embed.return_value = [[0.1] * 1536] * 10  # fake embeddings

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/ingest/upload",
                data={"chunking_strategy": "fixed", "chunk_size": "256", "chunk_overlap": "20"},
                files={"file": ("test.txt", io.BytesIO(txt_file), "text/plain")},
            )

    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "test.txt"
    assert data["chunking_strategy"] == "fixed"
    assert data["total_chunks"] > 0
    assert data["status"] == "ingested"


@pytest.mark.asyncio
async def test_upload_unsupported_type():
    transport = ASGITransport(app=app)
    with (
        patch("app.db.sql.create_tables", new_callable=AsyncMock),
        patch("app.db.qdrant.ensure_collection_exists", new_callable=AsyncMock),
    ):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/ingest/upload",
                data={"chunking_strategy": "fixed"},
                files={"file": ("test.docx", io.BytesIO(b"data"), "application/octet-stream")},
            )
    assert response.status_code == 415


@pytest.mark.asyncio
async def test_list_documents_empty(mock_db_session):
    with (
        patch("app.db.sql.get_db_session", return_value=mock_db_session),
        patch("app.db.sql.create_tables", new_callable=AsyncMock),
        patch("app.db.qdrant.ensure_collection_exists", new_callable=AsyncMock),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/ingest/documents")

    assert response.status_code == 200
    assert response.json()["total"] == 0
