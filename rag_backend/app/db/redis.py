from functools import lru_cache

import redis.asyncio as aioredis

from app.core.config import get_settings


@lru_cache
def get_redis_client() -> aioredis.Redis:
    settings = get_settings()
    return aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
    )
