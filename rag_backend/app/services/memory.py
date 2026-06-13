"""Redis-backed sliding-window chat history."""

from __future__ import annotations

import json

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.redis import get_redis_client
from app.schemas.conversation import ChatMessage

logger = get_logger(__name__)

_MAX_HISTORY = 20  # keep last N turns (user + assistant pairs)


def _key(session_id: str) -> str:
    return f"chat:history:{session_id}"


async def append_message(session_id: str, role: str, content: str) -> None:
    """Push a message onto the session's history list in Redis."""
    settings = get_settings()
    client = get_redis_client()
    key = _key(session_id)

    message = json.dumps({"role": role, "content": content})
    await client.rpush(key, message)  # type: ignore[arg-type]

    # Trim to keep only the last _MAX_HISTORY messages
    await client.ltrim(key, -_MAX_HISTORY, -1)

    # Reset TTL on every write
    await client.expire(key, settings.redis_chat_ttl_seconds)


async def get_history(session_id: str) -> list[ChatMessage]:
    """Return the full message history for a session."""
    client = get_redis_client()
    raw: list[str] = await client.lrange(_key(session_id), 0, -1)  # type: ignore[assignment]
    messages: list[ChatMessage] = []
    for item in raw:
        try:
            data = json.loads(item)
            messages.append(ChatMessage(role=data["role"], content=data["content"]))
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Skipping malformed history entry: %s", exc)
    return messages


async def clear_history(session_id: str) -> None:
    """Delete all history for a session."""
    client = get_redis_client()
    await client.delete(_key(session_id))
    logger.info("Cleared history for session '%s'", session_id)
