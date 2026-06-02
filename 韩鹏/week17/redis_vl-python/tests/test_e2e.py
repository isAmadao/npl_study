"""End-to-end tests for V0.6.0 integrated flows."""

from __future__ import annotations

import pytest

from app.config import get_settings


def _vector(dim: int, index: int) -> list[float]:
    values = [0.0] * dim
    values[index] = 1.0
    return values


@pytest.mark.asyncio
async def test_e2e_full_user_flow(async_client, monkeypatch) -> None:
    from app import main as main_module
    from app.redis_client import get_test_client

    dim = get_settings().embedding_dim
    mapping = {
        "Python 列表怎么排序": _vector(dim, 0),
        "Python 排序怎么写": _vector(dim, 0),
        "你好啊今天怎么样": _vector(dim, 1),
        "3.14 乘以 2": _vector(dim, 2),
    }

    class ApiFakeVectorizer:
        def embed(self, text: str) -> list[float]:
            if text in mapping:
                return mapping[text]
            return _vector(dim, 3)

    monkeypatch.setattr(main_module, "vectorizer", ApiFakeVectorizer())
    monkeypatch.setattr(main_module, "get_redis_client", get_test_client)
    monkeypatch.setattr(
        main_module,
        "DEFAULT_ROUTER_ROUTES",
        [
            {"name": "技术问答", "references": ["Python 列表怎么排序"], "distance_threshold": 0.3},
            {"name": "日常闲聊", "references": ["你好啊今天怎么样"], "distance_threshold": 0.35},
            {"name": "数学计算", "references": ["3.14 乘以 2"], "distance_threshold": 0.3},
        ],
    )

    first = await async_client.post("/api/chat", json={"session_id": "e2e-session-a", "prompt": "Python 列表怎么排序"})
    assert first.status_code == 200
    first_data = first.json()
    assert first_data["embeddings_cache"]["hit"] is False
    assert first_data["semantic_cache"]["hit"] is False
    assert first_data["router"]["route_name"] == "技术问答"
    assert first_data["message_history"]["count"] == 0
    assert first_data["message_history"]["recent_count"] == 1
    assert first_data["timing"]["flow"] == ["embeddings_cache", "semantic_cache", "semantic_router", "message_history"]

    second = await async_client.post("/api/chat", json={"session_id": "e2e-session-a", "prompt": "Python 列表怎么排序"})
    assert second.status_code == 200
    second_data = second.json()
    assert second_data["embeddings_cache"]["hit"] is True
    assert second_data["semantic_cache"]["hit"] is True
    assert second_data["message_history"]["count"] == 1
    assert len(second_data["message_history"]["relevant_messages"]) == 1
    assert second_data["message_history"]["recent_count"] == 2

    similar = await async_client.post("/api/chat", json={"session_id": "e2e-session-a", "prompt": "Python 排序怎么写"})
    assert similar.status_code == 200
    similar_data = similar.json()
    assert similar_data["semantic_cache"]["hit"] is True
    assert similar_data["router"]["route_name"] == "技术问答"
    assert similar_data["message_history"]["count"] >= 2

    chat_route = await async_client.post("/api/chat", json={"session_id": "e2e-session-a", "prompt": "你好啊今天怎么样"})
    assert chat_route.status_code == 200
    assert chat_route.json()["router"]["route_name"] == "日常闲聊"

    math_route = await async_client.post("/api/chat", json={"session_id": "e2e-session-a", "prompt": "3.14 乘以 2"})
    assert math_route.status_code == 200
    assert math_route.json()["router"]["route_name"] == "数学计算"


@pytest.mark.asyncio
async def test_e2e_session_history_and_clear_actions(async_client, monkeypatch) -> None:
    from app import main as main_module
    from app.redis_client import get_test_client

    dim = get_settings().embedding_dim

    class ApiFakeVectorizer:
        def embed(self, text: str) -> list[float]:
            if "会话A" in text:
                return _vector(dim, 0)
            if "会话B" in text:
                return _vector(dim, 1)
            return _vector(dim, 2)

    monkeypatch.setattr(main_module, "vectorizer", ApiFakeVectorizer())
    monkeypatch.setattr(main_module, "get_redis_client", get_test_client)
    monkeypatch.setattr(main_module, "DEFAULT_ROUTER_ROUTES", [])

    await async_client.post("/api/chat", json={"session_id": "session-a", "prompt": "会话A 第一条"})
    await async_client.post("/api/chat", json={"session_id": "session-a", "prompt": "会话A 第二条"})
    await async_client.post("/api/chat", json={"session_id": "session-b", "prompt": "会话B 第一条"})

    history_a = await async_client.get("/api/session/history", params={"session_id": "session-a", "top_k": 5})
    assert history_a.status_code == 200
    history_a_data = history_a.json()
    assert history_a_data["count"] == 2
    assert history_a_data["recent_messages"][0]["prompt"] == "会话A 第二条"

    history_b = await async_client.get("/api/session/history", params={"session_id": "session-b", "top_k": 5})
    assert history_b.status_code == 200
    history_b_data = history_b.json()
    assert history_b_data["count"] == 1
    assert history_b_data["recent_messages"][0]["prompt"] == "会话B 第一条"

    cleared = await async_client.delete("/api/session/history", params={"session_id": "session-a"})
    assert cleared.status_code == 200
    assert cleared.json()["cleared"] >= 2

    history_after_clear = await async_client.get("/api/session/history", params={"session_id": "session-a", "top_k": 5})
    assert history_after_clear.status_code == 200
    assert history_after_clear.json()["count"] == 0


@pytest.mark.asyncio
async def test_e2e_semantic_cache_clear_and_boundary_inputs(async_client, monkeypatch) -> None:
    from app import main as main_module
    from app.redis_client import get_test_client

    dim = get_settings().embedding_dim

    class ApiFakeVectorizer:
        def embed(self, text: str) -> list[float]:
            if "特殊字符" in text:
                return _vector(dim, 4)
            return _vector(dim, 0)

    monkeypatch.setattr(main_module, "vectorizer", ApiFakeVectorizer())
    monkeypatch.setattr(main_module, "get_redis_client", get_test_client)
    monkeypatch.setattr(
        main_module,
        "DEFAULT_ROUTER_ROUTES",
        [{"name": "技术问答", "references": ["超长输入"], "distance_threshold": 0.3}],
    )

    first = await async_client.post("/api/chat", json={"session_id": "clear-cache-session", "prompt": "超长输入"})
    assert first.status_code == 200
    assert first.json()["semantic_cache"]["hit"] is False

    second = await async_client.post("/api/chat", json={"session_id": "clear-cache-session", "prompt": "超长输入"})
    assert second.status_code == 200
    assert second.json()["semantic_cache"]["hit"] is True

    cleared = await async_client.delete("/api/cache/semantic")
    assert cleared.status_code == 200
    assert cleared.json()["cleared"] >= 1

    third = await async_client.post("/api/chat", json={"session_id": "clear-cache-session", "prompt": "超长输入"})
    assert third.status_code == 200
    assert third.json()["semantic_cache"]["hit"] is False

    blank = await async_client.post("/api/chat", json={"session_id": "blank-session", "prompt": "   "})
    assert blank.status_code == 400

    long_prompt = "超长输入" * 120
    long_response = await async_client.post("/api/chat", json={"session_id": "long-session", "prompt": long_prompt})
    assert long_response.status_code == 200
    assert long_response.json()["session_id"] == "long-session"

    special = await async_client.post(
        "/api/chat",
        json={"session_id": "特殊 / session ? id", "prompt": "特殊字符测试"},
    )
    assert special.status_code == 200
    special_data = special.json()
    assert special_data["normalized_session_id"] == "特殊___session___id"
