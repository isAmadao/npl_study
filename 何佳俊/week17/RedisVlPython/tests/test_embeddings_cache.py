"""
EmbeddingsCache 单元测试
========================
使用 fakeredis 模拟 Redis，使用 MagicMock 模拟 Milvus。
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.EmbeddingsCache import EmbeddingsCache, ConnectionError, NotFoundError


# ==================== 辅助函数 ====================


@pytest.fixture
def mock_redis():
    """创建 fakeredis 客户端用于测试。"""
    import fakeredis
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def embedding_cache(mock_redis):
    """创建 EmbeddingsCache 测试实例（使用 fakeredis，跳过 Milvus）。"""
    cache = EmbeddingsCache(
        redis_client=mock_redis,
        embedding_dim=128,
        skip_milvus=True,
    )
    return cache


@pytest.fixture
def sample_vector():
    """生成 128 维测试向量。"""
    np.random.seed(42)
    return np.random.randn(128).astype(np.float32)


# ==================== 测试用例 ====================


class TestEmbeddingsCacheInit:
    """测试初始化逻辑。"""

    def test_init_with_defaults(self, mock_redis):
        """默认参数初始化。"""
        cache = EmbeddingsCache(redis_client=mock_redis, embedding_dim=128)
        assert cache.dim == 128
        assert cache.redis is not None
        assert cache.milvus_available is False

    def test_init_custom_dim(self, mock_redis):
        """自定义向量维度。"""
        cache = EmbeddingsCache(redis_client=mock_redis, embedding_dim=256)
        assert cache.dim == 256


class TestEmbeddingsCacheSetGet:
    """测试向量的写入和读取。"""

    def test_set_and_get(self, embedding_cache, sample_vector):
        """写入后应能正确读取。"""
        key = "test_vector_1"
        result = embedding_cache.set(key, sample_vector)
        assert result is True

        retrieved = embedding_cache.get(key)
        assert retrieved is not None
        np.testing.assert_array_almost_equal(retrieved, sample_vector)

    def test_get_nonexistent_key(self, embedding_cache):
        """不存在的 key 应返回 None。"""
        result = embedding_cache.get("nonexistent")
        assert result is None

    def test_set_with_metadata(self, embedding_cache, sample_vector):
        """带元数据的写入。"""
        metadata = {"model": "test", "prompt": "hello", "tokens": 100}
        embedding_cache.set("key_meta", sample_vector, metadata=metadata)

        retrieved_meta = embedding_cache.get_metadata("key_meta")
        assert retrieved_meta is not None
        assert retrieved_meta["model"] == "test"
        assert retrieved_meta["tokens"] == 100

    def test_set_with_ttl(self, embedding_cache, sample_vector):
        """设置 TTL 后，过期后应无法读取。"""
        embedding_cache.set("key_ttl", sample_vector, ttl=1)
        assert embedding_cache.get("key_ttl") is not None
        time.sleep(1.1)
        assert embedding_cache.get("key_ttl") is None

    def test_set_invalid_dimension(self, embedding_cache):
        """维度不匹配应报错。"""
        wrong_vector = np.random.randn(64).astype(np.float32)
        with pytest.raises(ValueError, match="向量维度不匹配"):
            embedding_cache.set("bad_dim", wrong_vector)


class TestEmbeddingsCacheDelete:
    """测试删除逻辑。"""

    def test_delete_existing(self, embedding_cache, sample_vector):
        """删除已有数据。"""
        embedding_cache.set("to_delete", sample_vector)
        assert embedding_cache.exists("to_delete") is True

        result = embedding_cache.delete("to_delete")
        assert result is True
        assert embedding_cache.exists("to_delete") is False

    def test_delete_nonexistent(self, embedding_cache):
        """删除不存在的 key 返回 False。"""
        result = embedding_cache.delete("not_exist")
        assert result is False


class TestEmbeddingsCacheBatch:
    """测试批量操作。"""

    def test_batch_get(self, embedding_cache):
        """批量读取。"""
        vectors = {
            f"batch_{i}": np.random.randn(128).astype(np.float32)
            for i in range(5)
        }
        for key, vec in vectors.items():
            embedding_cache.set(key, vec)

        keys = list(vectors.keys())
        results = embedding_cache.batch_get(keys)
        assert len(results) == 5
        for i, key in enumerate(keys):
            np.testing.assert_array_almost_equal(results[i], vectors[key])

    def test_batch_get_mixed(self, embedding_cache):
        """混合存在的和不存在的 key。"""
        embedding_cache.set("exists", np.random.randn(128).astype(np.float32))
        results = embedding_cache.batch_get(["exists", "not_exist"])
        assert results[0] is not None
        assert results[1] is None


class TestEmbeddingsCacheStats:
    """测试统计信息。"""

    def test_get_stats_empty(self, embedding_cache):
        """空缓存的统计。"""
        stats = embedding_cache.get_stats()
        assert stats["vector_count"] == 0
        assert stats["milvus_available"] is False

    def test_get_stats_after_ops(self, embedding_cache, sample_vector):
        """操作后的统计。"""
        for i in range(3):
            embedding_cache.set(f"stat_{i}", sample_vector)
        stats = embedding_cache.get_stats()
        assert stats["vector_count"] >= 3


class TestEmbeddingsCacheSerialization:
    """测试序列化/反序列化。"""

    def test_serialize_deserialize(self, embedding_cache, sample_vector):
        """序列化再反序列化后应一致。"""
        serialized = embedding_cache._serialize_vector(sample_vector)
        deserialized = embedding_cache._deserialize_vector(serialized)
        np.testing.assert_array_almost_equal(sample_vector, deserialized)

    def test_deserialize_invalid(self, embedding_cache):
        """无效数据反序列化应报错。"""
        with pytest.raises(Exception):
            embedding_cache._deserialize_vector(b"invalid_data")
