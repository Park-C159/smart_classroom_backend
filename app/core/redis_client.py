"""Redis connection and helper utilities — Redis is optional for dev."""
import logging

logger = logging.getLogger(__name__)

_redis = None


async def _get_redis():
    """Lazy-init Redis connection. Returns None if Redis is unavailable."""
    global _redis
    if _redis is not None:
        return _redis
    try:
        import redis.asyncio as aioredis
        from app.config import settings

        _redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        await _redis.ping()
        logger.info("Redis connected: %s", settings.REDIS_URL)
        return _redis
    except Exception:
        logger.warning("Redis unavailable — running without cache/blacklist")
        _redis = False  # mark as tried-but-failed
        return None


async def redis_close() -> None:
    """Close Redis connection if open."""
    global _redis
    if _redis and _redis is not False:
        await _redis.close()
        _redis = None


# ── Token blacklist ──

async def blacklist_token(token: str, ttl: int) -> None:
    """Add a token to the blacklist with TTL equal to its remaining lifetime."""
    r = await _get_redis()
    if r:
        await r.setex(f"blacklist:{token}", ttl, "1")


async def is_token_blacklisted(token: str) -> bool:
    """Check if a token has been revoked."""
    r = await _get_redis()
    if r:
        return await r.exists(f"blacklist:{token}") > 0
    return False


# ── Speech cache ──

async def cache_speech_result(audio_hash: str, text: str, ttl: int = 3600) -> None:
    """Cache speech recognition result by audio hash."""
    r = await _get_redis()
    if r:
        await r.setex(f"speech:{audio_hash}", ttl, text)


async def get_cached_speech(audio_hash: str) -> str | None:
    """Retrieve cached speech recognition result."""
    r = await _get_redis()
    if r:
        return await r.get(f"speech:{audio_hash}")
    return None


# ── Question cache ──

async def cache_qa_result(question_hash: str, answer: str, ttl: int = 1800) -> None:
    """Cache Q&A result for duplicate questions."""
    r = await _get_redis()
    if r:
        await r.setex(f"qa:{question_hash}", ttl, answer)


async def get_cached_qa(question_hash: str) -> str | None:
    """Retrieve cached Q&A result."""
    r = await _get_redis()
    if r:
        return await r.get(f"qa:{question_hash}")
    return None
