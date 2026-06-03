"""Tests for V0.4.0 SemanticMessageHistory."""

from __future__ import annotations

import time

import pytest

from app.config import get_settings
from app.modules.message_history import SemanticMessageHistory


def _vector(dim: int, index: int) -> list[float]:
    values = [0.0] * dim
    values[index] = 1.0
    return values


class FakeVectorizer:
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    def embed(self, text: str) -> list[float]:
        if text not in self.mapping:
            raise KeyError(f"Missing fake vector for text: {text}")
        return self.mapping[text]


def test_message_history_store_and_recent(redis_client) -> None:
    dim = get_settings().embedding_dim
    mapping = {
        "第一条": _vector(dim, 0),
        "第二条": _vector(dim, 1),
        "第三条": _vector(dim, 2),
    }
    history = SemanticMessageHistory(
        redis_client=redis_client,
        vectorizer=FakeVectorizer(mapping),
        name="hist-recent",
        session_tag="session-recent",
    )

    history.store("第一条", "回复1")
    time.sleep(0.02)
    history.store("第二条", "回复2")
    time.sleep(0.02)
    history.store("第三条", "回复3")

    recent = history.get_recent(top_k=2)
    assert len(recent) == 2
    assert recent[0]["prompt"] == "第三条"
    assert recent[1]["prompt"] == "第二条"


def test_message_history_relevant_search(redis_client) -> None:
    dim = get_settings().embedding_dim
    vector = _vector(dim, 0)
    mapping = {
        "什么是消息历史": vector,
        "消息历史怎么用": vector,
        "无关问题": _vector(dim, 1),
    }
    history = SemanticMessageHistory(
        redis_client=redis_client,
        vectorizer=FakeVectorizer(mapping),
        name="hist-relevant",
        session_tag="session-relevant",
    )

    history.store("什么是消息历史", "消息历史用于记录对话")
    relevant = history.get_relevant("消息历史怎么用", top_k=3)

    assert len(relevant) == 1
    assert relevant[0]["prompt"] == "什么是消息历史"
    assert relevant[0]["distance"] <= 0.5


def test_message_history_session_isolation(redis_client) -> None:
    dim = get_settings().embedding_dim
    mapping = {
        "会话A消息": _vector(dim, 0),
        "会话B消息": _vector(dim, 0),
    }
    vectorizer = FakeVectorizer(mapping)

    history_a = SemanticMessageHistory(
        redis_client=redis_client,
        vectorizer=vectorizer,
        name="hist-isolation",
        session_tag="session-a",
    )
    history_b = SemanticMessageHistory(
        redis_client=redis_client,
        vectorizer=vectorizer,
        name="hist-isolation",
        session_tag="session-b",
    )

    history_a.store("会话A消息", "A回复")
    history_b.store("会话B消息", "B回复")

    recent_a = history_a.get_recent(top_k=5)
    recent_b = history_b.get_recent(top_k=5)
    assert [item["prompt"] for item in recent_a] == ["会话A消息"]
    assert [item["prompt"] for item in recent_b] == ["会话B消息"]


def test_message_history_clear(redis_client) -> None:
    dim = get_settings().embedding_dim
    mapping = {
        "清除1": _vector(dim, 0),
        "清除2": _vector(dim, 1),
    }
    history = SemanticMessageHistory(
        redis_client=redis_client,
        vectorizer=FakeVectorizer(mapping),
        name="hist-clear",
        session_tag="session-clear",
    )

    history.store("清除1", "回复1")
    history.store("清除2", "回复2")
    cleared = history.clear()

    assert cleared >= 2
    assert history.get_recent(top_k=5) == []


@pytest.mark.asyncio
async def test_api_chat_message_history_structure(async_client, monkeypatch) -> None:
    from app import main as main_module
    from app.redis_client import get_test_client

    dim = get_settings().embedding_dim
    shared_vector = _vector(dim, 0)
    mapping = {
        "缓存是什么": shared_vector,
        "什么是缓存": shared_vector,
    }

    class ApiFakeVectorizer:
        def embed(self, text: str) -> list[float]:
            if text in mapping:
                return mapping[text]
            return _vector(dim, 1)

    monkeypatch.setattr(main_module, "vectorizer", ApiFakeVectorizer())
    monkeypatch.setattr(main_module, "get_redis_client", get_test_client)
    monkeypatch.setattr(main_module, "DEFAULT_ROUTER_ROUTES", [])

    first = await async_client.post("/api/chat", json={"session_id": "api-message-session", "prompt": "缓存是什么"})
    assert first.status_code == 200
    first_data = first.json()
    assert first_data["message_history"]["count"] == 0
    assert first_data["message_history"]["stored"]["prompt"] == "缓存是什么"
    assert first_data["router"]["matched"] is False

    second = await async_client.post("/api/chat", json={"session_id": "api-message-session", "prompt": "什么是缓存"})
    assert second.status_code == 200
    second_data = second.json()
    assert second_data["message_history"]["count"] == 1
    assert second_data["message_history"]["relevant_messages"][0]["prompt"] == "缓存是什么"
    assert second_data["message_history"]["stored"]["response"] == second_data["semantic_cache"]["response"]
    assert second_data["router"]["matched"] is False
