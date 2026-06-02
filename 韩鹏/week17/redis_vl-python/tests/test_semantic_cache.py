"""Tests for V0.3.0 semantic cache behavior."""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.modules.semantic_cache import SemanticCache


def _sparse_vector(dim: int, values: dict[int, float]) -> list[float]:
    vector = [0.0] * dim
    for index, value in values.items():
        vector[index] = value
    return vector


class FakeVectorizer:
    def __init__(self, mapping: dict[str, list[float]], dim: int) -> None:
        self.mapping = mapping
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        if text not in self.mapping:
            raise KeyError(f"Missing fake vector for text: {text}")
        return self.mapping[text]


def test_semantic_cache_store_and_exact_hit(redis_client) -> None:
    dim = get_settings().embedding_dim
    mapping = {
        "什么是缓存": _sparse_vector(dim, {0: 1.0}),
    }
    vectorizer = FakeVectorizer(mapping=mapping, dim=dim)
    cache = SemanticCache(redis_client=redis_client, vectorizer=vectorizer, name="sem-exact", distance_threshold=0.2)

    cache.store("什么是缓存", "缓存可以复用结果")
    result = cache.check("什么是缓存")
    assert result is not None
    assert result["response"] == "缓存可以复用结果"
    assert result["distance"] <= 0.2


def test_semantic_cache_similar_prompt_hit(redis_client) -> None:
    dim = get_settings().embedding_dim
    base = _sparse_vector(dim, {0: 1.0})
    mapping = {
        "Python 缓存是什么": base,
        "什么叫做 Python 缓存": base,
    }
    vectorizer = FakeVectorizer(mapping=mapping, dim=dim)
    cache = SemanticCache(redis_client=redis_client, vectorizer=vectorizer, name="sem-similar", distance_threshold=0.2)

    cache.store("Python 缓存是什么", "用于避免重复计算")
    result = cache.check("什么叫做 Python 缓存")
    assert result is not None
    assert result["response"] == "用于避免重复计算"
    assert result["cached_prompt"] == "Python 缓存是什么"


def test_semantic_cache_unrelated_prompt_miss(redis_client) -> None:
    dim = get_settings().embedding_dim
    mapping = {
        "向量数据库是什么": _sparse_vector(dim, {0: 1.0}),
        "今天天气如何": _sparse_vector(dim, {1: 1.0}),
    }
    vectorizer = FakeVectorizer(mapping=mapping, dim=dim)
    cache = SemanticCache(redis_client=redis_client, vectorizer=vectorizer, name="sem-unrelated", distance_threshold=0.2)

    cache.store("向量数据库是什么", "用于语义检索")
    result = cache.check("今天天气如何")
    assert result is None


def test_semantic_cache_distance_threshold(redis_client) -> None:
    dim = get_settings().embedding_dim
    mapping = {
        "阈值测试-原问题": _sparse_vector(dim, {0: 1.0}),
        "阈值测试-相似问题": _sparse_vector(dim, {0: 0.98, 1: 0.2}),
    }
    vectorizer = FakeVectorizer(mapping=mapping, dim=dim)
    cache = SemanticCache(redis_client=redis_client, vectorizer=vectorizer, name="sem-threshold", distance_threshold=0.01)

    cache.store("阈值测试-原问题", "阈值测试答案")
    result = cache.check("阈值测试-相似问题")
    assert result is None


def test_semantic_cache_clear(redis_client) -> None:
    dim = get_settings().embedding_dim
    mapping = {
        "清理缓存-1": _sparse_vector(dim, {0: 1.0}),
        "清理缓存-2": _sparse_vector(dim, {1: 1.0}),
    }
    vectorizer = FakeVectorizer(mapping=mapping, dim=dim)
    cache = SemanticCache(redis_client=redis_client, vectorizer=vectorizer, name="sem-clear", distance_threshold=0.2)

    cache.store("清理缓存-1", "回复1")
    cache.store("清理缓存-2", "回复2")
    cleared = cache.clear()

    assert cleared >= 2
    assert cache.check("清理缓存-1") is None


@pytest.mark.asyncio
async def test_api_chat_semantic_cache_status(async_client, redis_client, monkeypatch) -> None:
    from app import main as main_module
    from app.redis_client import get_test_client

    dim = get_settings().embedding_dim
    semantic_base = _sparse_vector(dim, {0: 1.0})
    mapping = {
        "缓存是什么": semantic_base,
        "什么是缓存": semantic_base,
    }

    class ApiFakeVectorizer:
        def embed(self, text: str) -> list[float]:
            if text in mapping:
                return mapping[text]
            return _sparse_vector(dim, {2: 1.0})

    redis_client.flushdb()
    monkeypatch.setattr(main_module, "vectorizer", ApiFakeVectorizer())
    monkeypatch.setattr(main_module, "get_redis_client", get_test_client)
    monkeypatch.setattr(main_module, "DEFAULT_ROUTER_ROUTES", [])

    first = await async_client.post("/api/chat", json={"session_id": "s-sem", "prompt": "缓存是什么"})
    assert first.status_code == 200
    first_data = first.json()
    assert first_data["semantic_cache"]["hit"] is False
    assert "response" in first_data["semantic_cache"]
    assert first_data["message_history"]["count"] == 0
    assert first_data["message_history"]["relevant_messages"] == []
    assert first_data["router"]["matched"] is False

    second = await async_client.post("/api/chat", json={"session_id": "s-sem", "prompt": "什么是缓存"})
    assert second.status_code == 200
    second_data = second.json()
    assert second_data["semantic_cache"]["hit"] is True
    assert second_data["semantic_cache"]["cached_prompt"] == "缓存是什么"
    assert second_data["semantic_cache"]["distance"] <= 0.2
    assert second_data["message_history"]["count"] == 1
    assert second_data["message_history"]["relevant_messages"][0]["prompt"] == "缓存是什么"
    assert second_data["router"]["matched"] is False
