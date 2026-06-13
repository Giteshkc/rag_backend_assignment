"""
Tests for the Conversational RAG API.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.schemas.conversation import SourceChunk


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mock_db():
    session = AsyncMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.execute = AsyncMock(
        return_value=MagicMock(scalars=lambda: MagicMock(all=lambda: []))
    )
    return session


# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_message_rag_response():
    fake_sources = [
        SourceChunk(
            chunk_id="c1",
            text="The company was founded in 2010.",
            score=0.92,
            document_id="d1",
            filename="company.pdf",
        )
    ]

    with (
        patch("app.db.sql.get_db_session", return_value=_mock_db()),
        patch("app.services.booking.extract_booking", new_callable=AsyncMock) as mock_book,
        patch("app.services.rag.run_rag", new_callable=AsyncMock) as mock_rag,
        patch("app.db.sql.create_tables", new_callable=AsyncMock),
        patch("app.db.qdrant.ensure_collection_exists", new_callable=AsyncMock),
    ):
        mock_book.return_value = {"intent": False}
        mock_rag.return_value = ("The company was founded in 2010.", fake_sources)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/chat/message",
                json={
                    "session_id": "test-session-001",
                    "message": "When was the company founded?",
                },
            )

    assert response.status_code == 200
    data = response.json()
    assert "2010" in data["answer"]
    assert len(data["sources"]) == 1
    assert data["booking"] is None


@pytest.mark.asyncio
async def test_chat_message_booking_complete():
    booking_data = {
        "intent": True,
        "complete": True,
        "name": "Alice Smith",
        "email": "alice@example.com",
        "date": "2025-08-01",
        "time": "14:00",
    }

    with (
        patch("app.db.sql.get_db_session", return_value=_mock_db()),
        patch("app.services.booking.extract_booking", new_callable=AsyncMock) as mock_book,
        patch("app.services.memory.get_history", new_callable=AsyncMock, return_value=[]),
        patch("app.services.memory.append_message", new_callable=AsyncMock),
        patch("app.db.sql.create_tables", new_callable=AsyncMock),
        patch("app.db.qdrant.ensure_collection_exists", new_callable=AsyncMock),
    ):
        mock_book.return_value = booking_data

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/chat/message",
                json={
                    "session_id": "book-session-001",
                    "message": "I'd like to book an interview for Alice Smith, alice@example.com, on August 1st at 2 PM.",
                },
            )

    assert response.status_code == 200
    data = response.json()
    assert data["booking"] is not None
    assert data["booking"]["name"] == "Alice Smith"
    assert data["booking"]["email"] == "alice@example.com"
    assert data["booking"]["status"] == "confirmed"


@pytest.mark.asyncio
async def test_chat_message_booking_incomplete():
    with (
        patch("app.db.sql.get_db_session", return_value=_mock_db()),
        patch("app.services.booking.extract_booking", new_callable=AsyncMock) as mock_book,
        patch("app.services.memory.get_history", new_callable=AsyncMock, return_value=[]),
        patch("app.services.memory.append_message", new_callable=AsyncMock),
        patch("app.db.sql.create_tables", new_callable=AsyncMock),
        patch("app.db.qdrant.ensure_collection_exists", new_callable=AsyncMock),
    ):
        mock_book.return_value = {
            "intent": True,
            "complete": False,
            "missing": ["date", "time"],
            "follow_up": "Could you provide the preferred date and time?",
        }

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/chat/message",
                json={"session_id": "book-session-002", "message": "I'd like to book an interview."},
            )

    assert response.status_code == 200
    data = response.json()
    assert "date" in data["answer"].lower() or "time" in data["answer"].lower()
    assert data["booking"] is None


@pytest.mark.asyncio
async def test_get_history_empty():
    with (
        patch("app.services.memory.get_history", new_callable=AsyncMock, return_value=[]),
        patch("app.db.sql.create_tables", new_callable=AsyncMock),
        patch("app.db.qdrant.ensure_collection_exists", new_callable=AsyncMock),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/chat/history/new-session")

    assert response.status_code == 200
    assert response.json()["messages"] == []


@pytest.mark.asyncio
async def test_delete_history():
    with (
        patch("app.services.memory.clear_history", new_callable=AsyncMock),
        patch("app.db.sql.create_tables", new_callable=AsyncMock),
        patch("app.db.qdrant.ensure_collection_exists", new_callable=AsyncMock),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.delete("/api/v1/chat/history/some-session")

    assert response.status_code == 204
