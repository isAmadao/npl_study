"""Redis client helpers."""

from __future__ import annotations

from typing import Optional

import redis
from redis import Redis
from redis.exceptions import RedisError

from app.config import get_settings


def _build_client(*, db: int = 0) -> Redis:
    settings = get_settings()
    password: Optional[str] = settings.redis_password or None
    return redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        password=password,
        db=db,
        decode_responses=True,
    )


def _ensure_connection(client: Redis) -> None:
    settings = get_settings()
    try:
        client.ping()
    except RedisError as exc:
        raise RuntimeError(
            f"Unable to connect to Redis at {settings.redis_host}:{settings.redis_port}. "
            "Please start Redis Stack with `docker compose up -d`."
        ) from exc


def get_redis_client() -> Redis:
    """Create the default Redis client and verify connectivity."""
    client = _build_client(db=0)
    _ensure_connection(client)
    return client


def get_test_client() -> Redis:
    """Create a Redis client for tests.

    RediSearch vector indices are limited to DB 0 in Redis Stack, so tests
    that cover SemanticCache must use DB 0 as well.
    """
    return _build_client(db=0)
