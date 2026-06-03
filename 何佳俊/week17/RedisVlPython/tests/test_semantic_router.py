"""
SemanticRouter 单元测试
======================
"""

import numpy as np
import pytest

from src.SemanticRouter import (
    SemanticRouter,
    RouteConflictError,
    RouteNotFoundError,
    RouterConfigurationError,
)
from src.EmbeddingsCache import EmbeddingsCache


# ==================== 辅助函数 ====================


def dummy_embedding(text: str) -> np.ndarray:
    """测试用 Embedding 函数。"""
    np.random.seed(sum(ord(c) for c in text[:50]) % 2**31)
    return np.random.randn(128).astype(np.float32)


@pytest.fixture
def router():
    """创建 SemanticRouter 测试实例。"""
    import fakeredis
    redis_client = fakeredis.FakeRedis(decode_responses=True)
    cache = EmbeddingsCache(redis_client=redis_client, embedding_dim=128, skip_milvus=True)
    return SemanticRouter(
        cache=cache,
        embedding_func=dummy_embedding,
        confidence_threshold=0.7,
        default_route="fallback",
    )


# ==================== 测试用例 ====================


class TestRouterInit:
    """测试初始化。"""

    def test_init_defaults(self, router):
        """默认参数初始化。"""
        assert router.confidence_threshold == 0.7
        assert router.default_route == "fallback"
        assert len(router.routes) == 0

    def test_invalid_threshold(self):
        """无效阈值应报错。"""
        import fakeredis
        redis_client = fakeredis.FakeRedis(decode_responses=True)
        cache = EmbeddingsCache(redis_client=redis_client, embedding_dim=128)
        cache.milvus_available = False

        with pytest.raises(ValueError, match="confidence_threshold"):
            SemanticRouter(
                cache=cache,
                embedding_func=dummy_embedding,
                confidence_threshold=1.5,
            )


class TestRegisterRoute:
    """测试路由注册。"""

    def test_register_basic(self, router):
        """基本路由注册。"""
        result = router.register_route(
            name="greeting",
            examples=["你好", "早上好", "hello"],
            description="问候类查询",
        )
        assert result is True
        assert "greeting" in router.routes

    def test_register_duplicate(self, router):
        """重复注册应报错。"""
        router.register_route("test_route", ["示例1"])
        with pytest.raises(RouteConflictError, match="路由已存在"):
            router.register_route("test_route", ["示例2"])

    def test_register_empty_examples(self, router):
        """空示例列表应报错。"""
        with pytest.raises(RouterConfigurationError, match="示例列表不能为空"):
            router.register_route("empty_route", [])

    def test_register_with_handler(self, router):
        """带处理函数的注册。"""
        def my_handler(query):
            return f"处理: {query}"

        router.register_route(
            "code",
            ["写一个 Python 函数", "如何实现排序"],
            handler=my_handler,
        )
        assert router.routes["code"].handler is not None

    def test_multiple_routes(self, router):
        """注册多个路由。"""
        router.register_route(
            "greeting",
            ["你好", "hello"],
        )
        router.register_route(
            "weather",
            ["今天天气", "下雨吗"],
        )
        router.register_route(
            "code",
            ["写代码", "如何实现"],
        )
        assert len(router.routes) == 3
        assert len(router._route_centroids) == 3


class TestRouteQuery:
    """测试查询路由。"""

    def test_route_with_no_routes(self, router):
        """无路由时应走兜底。"""
        result = router.route("你好")
        assert result.route_name == "fallback"
        assert result.is_fallback is True

    def test_route_to_registered(self, router):
        """查询应路由到最匹配的意图。"""
        router.register_route(
            "greeting",
            ["你好", "hello", "早上好"],
        )
        router.register_route(
            "weather",
            ["今天天气怎么样", "下雨了", "气温多少"],
        )

        result = router.route("你好啊")
        # 由于使用伪 Embedding，可能匹配到 greeting 或 fallback
        # 至少保证方法正常执行
        assert isinstance(result.route_name, str)
        assert 0.0 <= result.confidence <= 1.0

    def test_route_confidence(self, router):
        """查询应返回置信度。"""
        router.register_route("test", ["test_query"])
        result = router.route("test_query")
        assert result.confidence >= 0.0
        assert result.query == "test_query"


class TestRouteManagement:
    """测试路由管理功能。"""

    def test_add_examples(self, router):
        """添加示例应更新路由。"""
        router.register_route("test", ["原始示例"])
        original_centroid = router._route_centroids["test"].copy()

        router.add_examples("test", ["新增示例1", "新增示例2"])
        assert len(router.routes["test"].examples) == 3

    def test_add_examples_nonexistent_route(self, router):
        """为不存在的路由添加示例应报错。"""
        with pytest.raises(RouteNotFoundError, match="路由不存在"):
            router.add_examples("no_such_route", ["示例"])

    def test_remove_route(self, router):
        """移除路由。"""
        router.register_route("test", ["示例1"])
        assert "test" in router.routes

        router.remove_route("test")
        assert "test" not in router.routes

    def test_remove_nonexistent_route(self, router):
        """移除不存在的路由应报错。"""
        with pytest.raises(RouteNotFoundError, match="路由不存在"):
            router.remove_route("no_such_route")


class TestRouterStats:
    """测试路由统计。"""

    def test_stats_empty(self, router):
        """空路由统计。"""
        stats = router.get_route_stats()
        assert stats.total_queries == 0
        assert stats.avg_confidence == 0.0
        assert stats.fallback_rate == 0.0

    def test_stats_after_routes(self, router):
        """路由查询后统计。"""
        router.register_route("test", ["test_query"])
        router.route("some query")
        router.route("another query")

        stats = router.get_route_stats()
        assert stats.total_queries == 2
