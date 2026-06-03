"""
基础使用示例
=============
展示四个核心模块的基本用法（真实 Redis + Qwen Embedding 模式）。

运行要求：
  - Redis 运行中（默认 localhost:6379）
  - DASHSCOPE_API_KEY 或 EMBEDDING_API_KEY 已设置
"""

import logging
import os
import sys

import numpy as np

sys.path.insert(0, "..")

from src.EmbeddingsCache import EmbeddingsCache
from src.SemanticCache import SemanticCache
from src.SemanticMessageHistory import SemanticMessageHistory
from src.SemanticRouter import SemanticRouter

logging.basicConfig(level=logging.WARNING)
logging.getLogger("pymilvus").setLevel(logging.ERROR)


def get_embedding() -> callable:
    """初始化 Qwen Embedding。"""
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


def demo_embeddings_cache():
    """EmbeddingsCache 读写演示。"""
    print("\n" + "=" * 60)
    print("📦 EmbeddingsCache")
    print("=" * 60)

    cache = EmbeddingsCache(embedding_dim=1536)

    vec = np.random.randn(1536).astype(np.float32)
    cache.set("demo_vec", vec, metadata={"source": "demo"})
    print(f"   ✅ 写入向量")

    retrieved = cache.get("demo_vec")
    print(f"   ✅ 读取向量: 维度 {retrieved.shape}")

    stats = cache.get_stats()
    print(f"   📊 缓存条目: {stats['vector_count']}")

    cache.delete("demo_vec")
    print(f"   ✅ 删除成功")


def demo_semantic_cache():
    """SemanticCache 语义匹配演示。"""
    print("\n" + "=" * 60)
    print("🎯 SemanticCache")
    print("=" * 60)

    embed_fn = get_embedding()
    cache = EmbeddingsCache(embedding_dim=1536)
    sem_cache = SemanticCache(cache=cache, embedding_model=embed_fn, similarity_threshold=0.6)

    # 写入
    sem_cache.store("什么是 Redis？", "Redis 是一个开源的内存数据结构存储系统。")
    print(f"   ✅ 已存储 Q&A")

    # 精确匹配
    result = sem_cache.lookup("什么是 Redis？")
    print(f"   ✅ 精确匹配: {result.answer[:50] if result else '未命中'}...")

    # 语义匹配
    result = sem_cache.lookup("Redis 是什么软件？")
    if result:
        print(f"   ✅ 语义匹配 (相似度: {result.similarity:.2%})")


def demo_message_history():
    """SemanticMessageHistory 对话记录演示。"""
    print("\n" + "=" * 60)
    print("💬 SemanticMessageHistory")
    print("=" * 60)

    embed_fn = get_embedding()
    cache = EmbeddingsCache(embedding_dim=1536)
    history = SemanticMessageHistory(cache=cache, embedding_func=embed_fn)

    history.add_message("s1", "user", "Python 怎么读文件？")
    history.add_message("s1", "assistant", "使用 open() 函数。")

    msgs = history.get_history("s1")
    print(f"   📜 {len(msgs)} 条消息")

    stats = history.get_session_stats("s1")
    print(f"   📊 Token: {stats['total_tokens']}")

    sessions = history.list_sessions()
    print(f"   📂 活跃会话: {sessions}")


def demo_semantic_router():
    """SemanticRouter 路由演示。"""
    print("\n" + "=" * 60)
    print("🧭 SemanticRouter")
    print("=" * 60)

    embed_fn = get_embedding()
    cache = EmbeddingsCache(embedding_dim=1536)
    router = SemanticRouter(
        cache=cache, embedding_func=embed_fn, confidence_threshold=0.5,
    )

    def handler(q: str) -> str:
        return f"路由处理: {q}"

    router.register_route("greeting", ["你好", "hello", "早上好"], handler=handler)
    router.register_route("weather", ["今天天气", "下雨了吗"], handler=handler)

    for q in ["你好啊", "今天会下雨吗"]:
        result = router.route(q)
        route = "🔄 兜底" if result.is_fallback else f"➡️ {result.route_name}"
        print(f"   '{q}' → {route} (置信度: {result.confidence:.2f})")


def main():
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("EMBEDDING_API_KEY")
    if not api_key:
        print("❌ 请设置 DASHSCOPE_API_KEY")
        return

    print("🚀 RedisVL Agent Cache — 基础功能演示")
    print("   后端: Redis + Qwen Embedding")

    demo_embeddings_cache()
    demo_semantic_cache()
    demo_message_history()
    demo_semantic_router()

    print("\n" + "=" * 60)
    print("✅ 演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
