"""
LLM-driven interview booking extraction.

The LLM is prompted to return structured JSON when the user's message
contains booking intent (name, email, date, time).  If the information is
incomplete the assistant asks a follow-up question instead of guessing.
"""

from __future__ import annotations

import json
from datetime import date, time

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.conversation import ChatMessage

logger = get_logger(__name__)

_BOOKING_SYSTEM_PROMPT = """\
You are a helpful assistant that extracts interview booking details from user messages.

When a user wants to book an interview, extract:
- name        (full name)
- email       (valid email address)
- date        (ISO 8601 format: YYYY-MM-DD)
- time        (24-hour format: HH:MM)

Reply ONLY with a valid JSON object in one of these two shapes:

If ALL four fields are present:
{
  "complete": true,
  "name": "...",
  "email": "...",
  "date": "YYYY-MM-DD",
  "time": "HH:MM"
}

If any field is missing or ambiguous:
{
  "complete": false,
  "missing": ["field1", "field2"],
  "follow_up": "A polite question asking only for the missing fields."
}

Do NOT output markdown, code fences, or any text outside the JSON object.
"""


def _detect_booking_intent(message: str) -> bool:
    keywords = [
        "book", "schedule", "interview", "appointment",
        "reserve", "set up a meeting", "arrange",
    ]
    lower = message.lower()
    return any(kw in lower for kw in keywords)


async def extract_booking(
    message: str,
    history: list[ChatMessage],
) -> dict:
    """
    Returns one of:
      {"complete": True, "name": ..., "email": ..., "date": ..., "time": ...}
      {"complete": False, "follow_up": "..."}
      {"intent": False}   ← no booking intent detected
    """
    if not _detect_booking_intent(message):
        return {"intent": False}

    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    # Build context from recent history + the current message
    context_messages = [
        {"role": msg.role, "content": msg.content}
        for msg in history[-6:]  # last 3 turns
    ]
    context_messages.append({"role": "user", "content": message})

    response = await client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": _BOOKING_SYSTEM_PROMPT},
            *context_messages,
        ],
        temperature=0,
        max_tokens=256,
    )

    raw = response.choices[0].message.content or "{}"
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Booking LLM returned non-JSON: %s", raw)
        return {"complete": False, "follow_up": "Could you please provide your booking details?"}

    result["intent"] = True
    return result
