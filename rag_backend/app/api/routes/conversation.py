"""
Conversational RAG API
======================
POST   /api/v1/chat/message              – send a message (RAG + optional booking)
GET    /api/v1/chat/history/{session_id} – retrieve conversation history
DELETE /api/v1/chat/history/{session_id} – clear conversation history
GET    /api/v1/chat/bookings             – list all interview bookings
"""

from __future__ import annotations

import uuid
from datetime import date, time

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.sql import get_db_session
from app.models.sql_models import InterviewBooking
from app.schemas.conversation import (
    BookingInfo,
    BookingListResponse,
    BookingRecord,
    ChatRequest,
    ChatResponse,
    ConversationHistoryResponse,
)
from app.services.booking import extract_booking
from app.services.memory import clear_history, get_history
from app.services.rag import run_rag

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["conversation"])


# ── Message endpoint ───────────────────────────────────────────────────────────

@router.post(
    "/message",
    response_model=ChatResponse,
    summary="Send a message and receive a RAG-grounded answer",
)
async def chat_message(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db_session),
) -> ChatResponse:
    session_id = request.session_id

    # ── Check for booking intent first ─────────────────────────────────────────
    history = await get_history(session_id)
    booking_extraction = await extract_booking(request.message, history)

    if booking_extraction.get("intent"):
        if not booking_extraction.get("complete"):
            # Missing fields — return follow-up question without RAG
            follow_up = booking_extraction.get(
                "follow_up", "Could you provide the missing details?"
            )
            from app.services.memory import append_message

            await append_message(session_id, "user", request.message)
            await append_message(session_id, "assistant", follow_up)
            return ChatResponse(
                session_id=session_id,
                answer=follow_up,
                sources=[],
                booking=None,
            )

        # ── Persist booking ────────────────────────────────────────────────────
        booking_id = str(uuid.uuid4())
        interview_date = date.fromisoformat(booking_extraction["date"])
        interview_time = time.fromisoformat(booking_extraction["time"])

        booking_record = InterviewBooking(
            id=booking_id,
            session_id=session_id,
            name=booking_extraction["name"],
            email=booking_extraction["email"],
            interview_date=interview_date,
            interview_time=interview_time,
            status="confirmed",
        )
        db.add(booking_record)
        await db.flush()

        confirmation = (
            f"Your interview has been booked! ✅\n"
            f"Name: {booking_extraction['name']}\n"
            f"Email: {booking_extraction['email']}\n"
            f"Date: {interview_date.strftime('%B %d, %Y')}\n"
            f"Time: {interview_time.strftime('%I:%M %p')}\n"
            f"Booking ID: {booking_id}"
        )

        from app.services.memory import append_message

        await append_message(session_id, "user", request.message)
        await append_message(session_id, "assistant", confirmation)

        logger.info("Booking confirmed | id=%s | session=%s", booking_id, session_id)

        return ChatResponse(
            session_id=session_id,
            answer=confirmation,
            sources=[],
            booking=BookingInfo(
                booking_id=booking_id,
                name=booking_extraction["name"],
                email=booking_extraction["email"],
                date=interview_date,
                time=interview_time,
                status="confirmed",
            ),
        )

    # ── Standard RAG flow ──────────────────────────────────────────────────────
    answer, sources = await run_rag(
        session_id=session_id,
        user_message=request.message,
        document_id=request.document_id,
    )

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        sources=sources,
        booking=None,
    )


# ── History endpoints ──────────────────────────────────────────────────────────

@router.get(
    "/history/{session_id}",
    response_model=ConversationHistoryResponse,
    summary="Get conversation history for a session",
)
async def get_conversation_history(session_id: str) -> ConversationHistoryResponse:
    messages = await get_history(session_id)
    return ConversationHistoryResponse(session_id=session_id, messages=messages)

@router.delete(
    "/history/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Clear conversation history for a session",
)
async def delete_conversation_history(session_id: str) -> dict:
    await clear_history(session_id)
    return {"message": "History cleared"}


# ── Bookings endpoint ──────────────────────────────────────────────────────────

@router.get(
    "/bookings",
    response_model=BookingListResponse,
    summary="List all interview bookings",
)
async def list_bookings(
    db: AsyncSession = Depends(get_db_session),
) -> BookingListResponse:
    result = await db.execute(
        select(InterviewBooking).order_by(InterviewBooking.created_at.desc())
    )
    bookings = result.scalars().all()
    return BookingListResponse(
        bookings=[BookingRecord.model_validate(b) for b in bookings],
        total=len(bookings),
    )
