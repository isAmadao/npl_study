"""Smoke tests for V0.1.0 infrastructure."""

from __future__ import annotations

import pytest


def test_redis_connection(redis_client) -> None:
    assert redis_client.ping() is True
    redis_client.set("smoke:key", "ok")
    assert redis_client.get("smoke:key") == "ok"


@pytest.mark.asyncio
async def test_health_endpoint(async_client) -> None:
    response = await async_client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_index_page_accessible(async_client) -> None:
    response = await async_client.get("/")
    assert response.status_code == 200
    assert "简易版 RedisVL" in response.text
    assert "EmbeddingsCache" in response.text

