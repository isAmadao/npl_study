"""Tests for V0.2.0 EmbeddingsCache and chat API behavior."""

from __future__ import annotations

import time

import pytest

from app.config import get_settings
from app.modules.embeddings_cache import EmbeddingsCache


def test_embeddings_cache_set_get_exists(redis_client) -> None:
    cache = EmbeddingsCache(redis_client=redis_client, name="test-basic")
    text = "hello embeddings cache"
    vector = [0.12, 0.34, 0.56]

    assert cache.exists(text) is False
    cache.set([text], [vector])

    assert cache.exists(text) is True
    result = cache.get([text])
    assert result[text] == vector


def test_embeddings_cache_batch_operations(redis_client) -> None:
    cache = EmbeddingsCache(redis_client=redis_client, name="test-batch")
    texts = ["text-a", "text-b", "text-c"]
    vectors = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    cache.set(texts, vectors)
    result = cache.get(texts)

    assert result["text-a"] == [1.0, 2.0]
    assert result["text-b"] == [3.0, 4.0]
    assert result["text-c"] == [5.0, 6.0]


def test_embeddings_cache_ttl_expire(redis_client) -> None:
    cache = EmbeddingsCache(redis_client=redis_client, name="test-ttl", ttl=1)
    text = "ttl-text"
    cache.set([text], [[0.1, 0.2]])

    assert cache.exists(text) is True
    time.sleep(1.2)
    assert cache.exists(text) is False
    assert cache.get([text])[text] is None


@pytest.mark.asyncio
async def test_api_chat_embeddings_hit_and_miss(async_client, redis_client, monkeypatch) -> None:
    from app import main as main_module
    from app.redis_client import get_test_client

    dim = get_settings().embedding_dim

    class FakeVectorizer:
        def embed(self, text: str) -> list[float]:
            vector = [0.0] * dim
            vector[0] = float(len(text))
            vector[1] = 1.5
            vector[2] = 2.5
            vector[3] = 3.5
            vector[4] = 4.5
            vector[5] = 5.5
            return vector

    redis_client.flushdb()
    monkeypatch.setattr(main_module, "vectorizer", FakeVectorizer())
    monkeypatch.setattr(main_module, "get_redis_client", get_test_client)
    monkeypatch.setattr(main_module, "DEFAULT_ROUTER_ROUTES", [])

    payload = {"session_id": "s1", "prompt": "hello world"}

    first = await async_client.post("/api/chat", json=payload)
    assert first.status_code == 200
    first_data = first.json()
    assert first_data["embeddings_cache"]["hit"] is False
    assert first_data["embeddings_cache"]["status"] == "miss"
    assert first_data["embeddings_cache"]["vector_dim"] == dim
    assert first_data["semantic_cache"]["hit"] is False
    assert first_data["message_history"]["count"] == 0
    assert first_data["message_history"]["relevant_messages"] == []
    assert first_data["message_history"]["stored"]["prompt"] == payload["prompt"]

    second = await async_client.post("/api/chat", json=payload)
    assert second.status_code == 200
    second_data = second.json()
    assert second_data["embeddings_cache"]["hit"] is True
    assert second_data["embeddings_cache"]["status"] == "hit"
    assert second_data["embeddings_cache"]["vector_dim"] == dim
    assert second_data["embeddings_cache"]["vector_preview"] == first_data["embeddings_cache"]["vector_preview"]
    assert second_data["semantic_cache"]["hit"] is True
    assert second_data["message_history"]["count"] == 1
    assert second_data["message_history"]["relevant_messages"][0]["prompt"] == payload["prompt"]
    assert second_data["message_history"]["stored"]["response"] == second_data["semantic_cache"]["response"]
    assert second_data["router"]["matched"] is False
    assert second_data["router"]["message"] == "未匹配到任何路由"
