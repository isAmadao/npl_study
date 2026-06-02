"""Shared test fixtures for infrastructure smoke tests."""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.redis_client import get_test_client


@pytest_asyncio.fixture
async def async_client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.fixture(scope="session")
def redis_client():
    client = get_test_client()
    try:
        client.ping()
    except Exception as exc:  # pragma: no cover - depends on local runtime state.
        pytest.skip(f"Redis is not available for tests: {exc}")

    client.flushdb()
    yield client
    client.flushdb()


@pytest.fixture(autouse=True)
def isolate_redis(redis_client):
    redis_client.flushdb()
    yield
    redis_client.flushdb()
