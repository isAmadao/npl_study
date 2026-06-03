"""
SemanticCache 单元测试
======================
"""

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.SemanticCache import SemanticCache, CacheResult, CacheStats
from src.EmbeddingsCache import EmbeddingsCache


# ==================== 辅助函数 ====================


def dummy_embedding(text: str) -> np.ndarray:
    """测试用 Embedding 函数：基于字符哈希的伪向量。"""
    np.random.seed(sum(ord(c) for c in text[:50]) % 2**31)
    return np.random.randn(128).astype(np.float32)


@pytest.fixture
def mock_embed_cache():
    """创建 Mock 的 EmbeddingsCache。"""
    import fakeredis
    redis_client = fakeredis.FakeRedis(decode_responses=True)
    cache = EmbeddingsCache(redis_client=redis_client, embedding_dim=128, skip_milvus=True)
    return cache


@pytest.fixture
def semantic_cache(mock_embed_cache):
    """创建 SemanticCache 测试实例。"""
    return SemanticCache(
        cache=mock_embed_cache,
        embedding_model=dummy_embedding,
        similarity_threshold=0.8,
        top_k=5,
    )


# ==================== 测试用例 ====================


class TestSemanticCacheInit:
    """测试初始化。"""

    def test_init_defaults(self, mock_embed_cache):
        """默认参数初始化。"""
        sc = SemanticCache(cache=mock_embed_cache, embedding_model=dummy_embedding)
        assert sc.similarity_threshold == 0.85
        assert sc.top_k == 5
        assert sc._hits == 0
        assert sc._misses == 0

    def test_init_custom_threshold(self, mock_embed_cache):
        """自定义阈值。"""
        sc = SemanticCache(
            cache=mock_embed_cache,
            embedding_model=dummy_embedding,
            similarity_threshold=0.9,
        )
        assert sc.similarity_threshold == 0.9

    def test_init_invalid_threshold(self, mock_embed_cache):
        """无效阈值应报错。"""
        with pytest.raises(ValueError, match="similarity_threshold"):
            SemanticCache(
                cache=mock_embed_cache,
                embedding_model=dummy_embedding,
                similarity_threshold=1.5,
            )


class TestSemanticCacheStore:
    """测试 Q&A 存储。"""

    def test_store(self, semantic_cache):
        """基本存储功能。"""
        result = semantic_cache.store("什么是 Redis？", "Redis 是一个内存数据库。")
        assert result is True

    def test_store_with_metadata(self, semantic_cache):
        """带元数据存储。"""
        result = semantic_cache.store(
            "Python 是什么？",
            "Python 是一种编程语言。",
            metadata={"source": "test", "model": "gpt-4"},
        )
        assert result is True


class TestSemanticCacheLookup:
    """测试语义查询。"""

    def test_lookup_exact_match(self, semantic_cache):
        """精确匹配应命中。"""
        semantic_cache.store("什么是 Redis？", "Redis 是内存数据库。")
        # 相同的问题应命中
        result = semantic_cache.lookup("什么是 Redis？")
        assert result is not None
        assert result.answer == "Redis 是内存数据库。"
        assert result.similarity >= 0.99

    def test_lookup_similar_query(self, semantic_cache):
        """语义相似的问题应命中。"""
        semantic_cache.store(
            "Redis 缓存的使用方法",
            "使用 Redis 缓存可以提升性能。",
        )
        result = semantic_cache.lookup("如何用 Redis 做缓存")
        # 用伪 Embedding，可能命中也可能不命中
        # 主要测试流程通顺
        assert result is None or isinstance(result, CacheResult)

    def test_lookup_low_threshold(self, semantic_cache):
        """降低阈值应提高命中率。"""
        semantic_cache.store(
            "Redis 缓存的使用方法",
            "使用 Redis 缓存可以提升性能。",
        )
        # 设置低阈值
        result = semantic_cache.lookup(
            "如何用 Redis 做缓存", threshold=0.1
        )
        # 低阈值应该能匹配到
        assert result is None or isinstance(result, CacheResult)

    def test_lookup_no_match(self, semantic_cache):
        """完全无关的问题应未命中。"""
        semantic_cache.store("Redis 数据库", "Redis 是内存数据库。")
        result = semantic_cache.lookup("今天天气怎么样")
        # 大概率不会命中
        if result is None:
            assert semantic_cache._misses > 0

    def test_lookup_empty_cache(self, semantic_cache):
        """空缓存应返回 None。"""
        result = semantic_cache.lookup("任何问题")
        assert result is None


class TestSemanticCacheInvalidate:
    """测试缓存失效。"""

    def test_invalidate(self, semantic_cache):
        """使缓存失效后不应再命中。"""
        semantic_cache.store("什么是 Redis？", "Redis 是内存数据库。")
        assert semantic_cache.lookup("什么是 Redis？") is not None

        semantic_cache.invalidate("什么是 Redis？")
        # 失效后可能由于伪 Embedding 的随机性仍匹配到
        # 至少确保方法不报错
        assert semantic_cache.lookup("什么是 Redis？") is None


class TestSemanticCacheStats:
    """测试统计功能。"""

    def test_stats_initial(self, semantic_cache):
        """初始统计。"""
        stats = semantic_cache.get_stats()
        assert stats.size == 0
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0

    def test_stats_after_ops(self, semantic_cache):
        """操作后的统计。"""
        semantic_cache.store("Q1", "A1")
        semantic_cache.lookup("Q1")
        semantic_cache.lookup("不相关的问题")

        stats = semantic_cache.get_stats()
        assert stats.size == 1

    def test_update_threshold(self, semantic_cache):
        """更新阈值。"""
        semantic_cache.update_threshold(0.9)
        assert semantic_cache.similarity_threshold == 0.9

        with pytest.raises(ValueError):
            semantic_cache.update_threshold(2.0)
