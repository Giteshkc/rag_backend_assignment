from datetime import date, datetime, time
from typing import Literal

from pydantic import BaseModel, EmailStr, Field


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="UUID identifying the conversation session")
    message: str = Field(..., min_length=1, max_length=4096)
    document_id: str | None = Field(
        default=None, description="Optional: restrict retrieval to a specific document"
    )


class SourceChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    document_id: str
    filename: str


# ── Booking ───────────────────────────────────────────────────────────────────

class BookingInfo(BaseModel):
    booking_id: str
    name: str
    email: str
    date: date
    time: time
    status: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
    booking: BookingInfo | None = None


class ConversationHistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatMessage]


# ── Bookings list ─────────────────────────────────────────────────────────────

class BookingRecord(BaseModel):
    id: str
    session_id: str
    name: str
    email: str
    interview_date: date
    interview_time: time
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class BookingListResponse(BaseModel):
    bookings: list[BookingRecord]
    total: int
