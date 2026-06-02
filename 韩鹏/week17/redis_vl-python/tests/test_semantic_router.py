"""Tests for V0.5.0 SemanticRouter."""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.modules.semantic_router import SemanticRouter


def _vector(dim: int, values: dict[int, float]) -> list[float]:
    vector = [0.0] * dim
    for index, value in values.items():
        vector[index] = value
    return vector


class FakeVectorizer:
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    def embed(self, text: str) -> list[float]:
        if text not in self.mapping:
            raise KeyError(f"Missing fake vector for text: {text}")
        return self.mapping[text]


def test_semantic_router_match_and_miss(redis_client) -> None:
    dim = get_settings().embedding_dim
    mapping = {
        "Python 怎么用": _vector(dim, {0: 1.0}),
        "你好吗": _vector(dim, {1: 1.0}),
        "1+1 等于几": _vector(dim, {2: 1.0}),
        "Python 列表怎么排序": _vector(dim, {0: 1.0}),
        "给我推荐一个投资策略": _vector(dim, {0: 0.57735027, 1: 0.57735027, 2: 0.57735027}),
    }
    router = SemanticRouter(
        redis_client=redis_client,
        vectorizer=FakeVectorizer(mapping),
        name="router-match",
    )

    router.add_routes(
        [
            {"name": "技术问答", "references": ["Python 怎么用"], "distance_threshold": 0.3},
            {"name": "日常闲聊", "references": ["你好吗"], "distance_threshold": 0.35},
            {"name": "数学计算", "references": ["1+1 等于几"], "distance_threshold": 0.3},
        ]
    )

    matched = router.route("Python 列表怎么排序")
    assert matched is not None
    assert matched["matched"] is True
    assert matched["route_name"] == "技术问答"
    assert matched["matched_reference"] == "Python 怎么用"
    assert matched["distance"] == pytest.approx(0.0, abs=1e-6)

    missed = router.route("给我推荐一个投资策略")
    assert missed is None


def test_semantic_router_threshold_control(redis_client) -> None:
    dim = get_settings().embedding_dim
    mapping = {
        "阈值-原问题": _vector(dim, {0: 1.0}),
        "阈值-相似问题": _vector(dim, {0: 0.8, 1: 0.6}),
    }
    router = SemanticRouter(
        redis_client=redis_client,
        vectorizer=FakeVectorizer(mapping),
        name="router-threshold",
    )

    router.add_routes(
        [
            {"name": "技术问答", "references": ["阈值-原问题"], "distance_threshold": 0.25},
        ]
    )

    hit = router.route("阈值-相似问题")
    assert hit is not None
    assert hit["route_name"] == "技术问答"
    assert hit["distance"] == pytest.approx(0.2, abs=1e-4)

    router.add_routes(
        [
            {"name": "技术问答", "references": ["阈值-原问题"], "distance_threshold": 0.15},
        ]
    )

    assert router.route("阈值-相似问题") is None


def test_semantic_router_add_and_modify_routes(redis_client) -> None:
    dim = get_settings().embedding_dim
    mapping = {
        "Python 怎么用": _vector(dim, {0: 1.0}),
        "Redis 是什么": _vector(dim, {1: 1.0}),
        "Python 列表怎么排序": _vector(dim, {0: 1.0}),
        "Redis 缓存是什么": _vector(dim, {1: 1.0}),
    }
    router = SemanticRouter(
        redis_client=redis_client,
        vectorizer=FakeVectorizer(mapping),
        name="router-modify",
    )

    router.add_routes(
        [
            {"name": "技术问答", "references": ["Python 怎么用"], "distance_threshold": 0.3},
        ]
    )
    first = router.route("Python 列表怎么排序")
    assert first is not None
    assert first["matched_reference"] == "Python 怎么用"

    router.add_routes(
        [
            {"name": "技术问答", "references": ["Redis 是什么"], "distance_threshold": 0.3},
        ]
    )
    assert router.route("Python 列表怎么排序") is None

    second = router.route("Redis 缓存是什么")
    assert second is not None
    assert second["matched_reference"] == "Redis 是什么"


def test_semantic_router_sync_routes_skips_unchanged_config(redis_client) -> None:
    dim = get_settings().embedding_dim
    mapping = {
        "Python 怎么用": _vector(dim, {0: 1.0}),
        "Python 列表怎么排序": _vector(dim, {0: 1.0}),
    }
    router = SemanticRouter(
        redis_client=redis_client,
        vectorizer=FakeVectorizer(mapping),
        name="router-sync",
    )
    routes = [
        {"name": "技术问答", "references": ["Python 怎么用"], "distance_threshold": 0.3},
    ]

    first_sync = router.sync_routes(routes)
    second_sync = router.sync_routes(routes)

    assert first_sync is True
    assert second_sync is False
    assert router.route("Python 列表怎么排序") is not None


@pytest.mark.asyncio
async def test_api_chat_router_status(async_client, monkeypatch) -> None:
    from app import main as main_module
    from app.redis_client import get_test_client

    dim = get_settings().embedding_dim
    mapping = {
        "Python 怎么用": _vector(dim, {0: 1.0}),
        "你好吗": _vector(dim, {1: 1.0}),
        "Python 列表怎么排序": _vector(dim, {0: 1.0}),
        "今天天气不错": _vector(dim, {1: 1.0}),
    }

    class ApiFakeVectorizer:
        def embed(self, text: str) -> list[float]:
            if text not in mapping:
                raise KeyError(f"Missing fake vector for text: {text}")
            return mapping[text]

    monkeypatch.setattr(main_module, "vectorizer", ApiFakeVectorizer())
    monkeypatch.setattr(main_module, "get_redis_client", get_test_client)
    monkeypatch.setattr(
        main_module,
        "DEFAULT_ROUTER_ROUTES",
        [
            {
                "name": "技术问答",
                "references": ["Python 怎么用"],
                "distance_threshold": 0.3,
            },
            {
                "name": "日常闲聊",
                "references": ["你好吗", "今天天气不错"],
                "distance_threshold": 0.35,
            },
        ],
    )

    response = await async_client.post(
        "/api/chat",
        json={"session_id": "router-session", "prompt": "Python 列表怎么排序"},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["router"]["matched"] is True
    assert data["router"]["route_name"] == "技术问答"
    assert data["router"]["matched_reference"] == "Python 怎么用"
    assert data["router"]["distance"] == pytest.approx(0.0, abs=1e-6)
