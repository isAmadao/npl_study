"""
Redis 持久化示例 — 展示数据持久化和重启恢复能力
=============================================

运行要求：
  - Redis 运行中（默认 localhost:6379）
  - DASHSCOPE_API_KEY
"""

import logging
import os
import sys
import time
from typing import List

from dotenv import load_dotenv
load_dotenv()

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.EmbeddingsCache import EmbeddingsCache

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def get_embedding():
    from dashscope import TextEmbedding
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("EMBEDDING_API_KEY")
    if not api_key:
        raise RuntimeError("请设置 DASHSCOPE_API_KEY")

    def embed(text: str) -> np.ndarray:
        resp = TextEmbedding.call(model="text-embedding-v2", input=text, api_key=api_key)
        if resp.status_code != 200:
            raise RuntimeError(f"API 错误: {resp.message}")
        return np.array(resp.output["embeddings"][0]["embedding"], dtype=np.float32)
    return embed


def demo_persistence(cache: EmbeddingsCache):
    """读写验证。"""
    print("\n" + "=" * 60)
    print("💾 持久化读写")
    print("=" * 60)

    for i in range(5):
        key = f"persist_{i}"
        vec = np.random.randn(1536).astype(np.float32)
        cache.set(key, vec, metadata={"index": i, "label": f"测试 #{i}"})
        print(f"   ✅ {key}")

    for i in range(5):
        v = cache.get(f"persist_{i}")
        m = cache.get_metadata(f"persist_{i}")
        print(f"   🔍 persist_{i}: 维度 {v.shape}, 标签 {m.get('label','?')}")

    for i in range(5):
        cache.delete(f"persist_{i}")
    print(f"   🧹 已清理")


def demo_restore(cache: EmbeddingsCache):
    """重启恢复。"""
    print("\n" + "=" * 60)
    print("🔄 重启恢复")
    print("=" * 60)

    key = "restore_test"
    vec = np.random.randn(1536).astype(np.float32)
    cache.set(key, vec, metadata={"source": "restore", "created_at": time.time()})
    print(f"   📝 已写入 {key}")

    # 模拟重启：创建新连接
    import redis
    new_r = redis.Redis(decode_responses=True)
    new_cache = EmbeddingsCache(redis_client=new_r, embedding_dim=1536, skip_milvus=True)

    restored = new_cache.get(key)
    if restored is not None:
        print(f"   ✅ 重启后恢复成功! 维度: {restored.shape}")
    else:
        print(f"   ❌ 未恢复（检查 Redis 连接）")

    cache.delete(key)


def demo_ttl(cache: EmbeddingsCache):
    """TTL 过期。"""
    print("\n" + "=" * 60)
    print("⏰ TTL 过期")
    print("=" * 60)

    key = "ttl_test"
    cache.set(key, np.random.randn(1536).astype(np.float32), ttl=3)
    print(f"   📝 {key} (TTL=3s)")
    print(f"   🔍 立即: {'✅ 存在' if cache.get(key) is not None else '❌ 无'}")
    time.sleep(4)
    print(f"   🔍 4s后: {'✅ 存在' if cache.get(key) is not None else '❌ 已过期'}")


def demo_batch(cache: EmbeddingsCache):
    """批量操作。"""
    print("\n" + "=" * 60)
    print("📦 批量操作")
    print("=" * 60)

    items = [(f"batch_{i}", np.random.randn(1536).astype(np.float32), {"i": i}) for i in range(50)]
    t0 = time.time()
    cache.batch_set(items)
    print(f"   ✅ 批量写入 50 条: {time.time()-t0:.3f}s")

    t0 = time.time()
    vs = cache.batch_get([f"batch_{i}" for i in range(5)])
    print(f"   ✅ 批量读取 5 条: {time.time()-t0:.3f}s")
    print(f"      成功 {sum(1 for v in vs if v is not None)}/5")

    for i in range(50):
        cache.delete(f"batch_{i}")
    print(f"   🧹 已清理")


def main():
    if not (os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("EMBEDDING_API_KEY")):
        print("❌ 请设置 DASHSCOPE_API_KEY")
        return

    print("=" * 60)
    print("💾 Redis 持久化演示")
    print("=" * 60)

    try:
        cache = EmbeddingsCache(embedding_dim=1536)
        print(f"   ✅ Redis 已连接")
    except Exception as e:
        print(f"❌ Redis 连接失败: {e}")
        return

    cache.redis.flushdb()

    demo_persistence(cache)
    demo_restore(cache)
    demo_ttl(cache)
    demo_batch(cache)

    print(f"\n{'=' * 60}")
    print("✅ 演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
