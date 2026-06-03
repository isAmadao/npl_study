"""
生产部署示例 — 连接远程 Redis + Milvus 服务

运行前:
  docker compose up -d
  export DASHSCOPE_API_KEY=sk-xxx
  python examples/production_deploy.py
"""

import logging
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.EmbeddingsCache import EmbeddingsCache
from src.SemanticCache import SemanticCache

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def get_embedding():
    from dashscope import TextEmbedding
    import numpy as np
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("EMBEDDING_API_KEY")

    def embed(text: str) -> np.ndarray:
        resp = TextEmbedding.call(model="text-embedding-v2", input=text, api_key=api_key)
        if resp.status_code != 200:
            raise RuntimeError(f"API 错误: {resp.message}")
        return np.array(resp.output["embeddings"][0]["embedding"], dtype=np.float32)
    return embed


def health_check(cache: EmbeddingsCache) -> dict:
    """健康检查。"""
    status = {"redis": False, "milvus": False, "latency_ms": {}}
    try:
        t0 = time.time()
        cache.redis.ping()
        status["redis"] = True
        status["latency_ms"]["redis"] = round((time.time() - t0) * 1000, 2)
    except Exception as e:
        status["redis_error"] = str(e)
    if cache.milvus_available and cache.milvus_client:
        try:
            t0 = time.time()
            cache.milvus_client.list_collections()
            status["milvus"] = True
            status["latency_ms"]["milvus"] = round((time.time() - t0) * 1000, 2)
        except Exception as e:
            status["milvus_error"] = str(e)
    return status


def demo_connect():
    """连接与健康检查。"""
    print("\n" + "=" * 60)
    print("🔌 连接与健康检查")
    print("=" * 60)

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    milvus_uri = os.environ.get("MILVUS_URI", "http://localhost:19530")
    print(f"\n   Redis:  {redis_url}")
    print(f"   Milvus: {milvus_uri}")

    from redis import ConnectionPool, Redis
    pool = ConnectionPool.from_url(redis_url, max_connections=20, socket_timeout=5)
    r = Redis(connection_pool=pool)

    try:
        cache = EmbeddingsCache(redis_client=r, milvus_uri=milvus_uri, embedding_dim=1536)
    except Exception as e:
        print(f"❌ {e}")
        return None

    h = health_check(cache)
    print(f"\n📊 健康状态:")
    print(f"   Redis:  {'✅' if h['redis'] else '❌'} {h['latency_ms'].get('redis','?')}ms")
    print(f"   Milvus: {'✅' if h['milvus'] else '❌'} {h['latency_ms'].get('milvus','?')}ms")
    return cache


def demo_production(cache: EmbeddingsCache):
    """生产写入与查询。"""
    print("\n" + "=" * 60)
    print("📝 生产读写")
    print("=" * 60)

    sem_cache = SemanticCache(
        cache=cache, embedding_model=get_embedding(), similarity_threshold=0.6,
    )

    qa = [
        ("订单状态怎么看", "登录账号 → 订单中心 → 查看订单状态"),
        ("如何申请退款", "订单中心 → 申请退款 → 填写原因 → 提交审核"),
        ("发货时间是什么", "工作日 16:00 前下单当天发货"),
        ("密码忘了怎么办", "登录页 → 忘记密码 → 重置"),
    ]
    for q, a in qa:
        sem_cache.store(q, a, metadata={"source": "demo"})
    print(f"   ✅ 存入 {len(qa)} 条")

    for q in ["怎么看订单状态", "退款怎么操作"]:
        r = sem_cache.lookup(q)
        print(f"   {'✅' if r else '❌'} '{q}' → {r.answer[:60] if r else '未命中'}...")


def main():
    if not (os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("EMBEDDING_API_KEY")):
        print("❌ 请设置 DASHSCOPE_API_KEY")
        return

    print("=" * 60)
    print("🚀 生产部署示例")
    print("=" * 60)

    cache = demo_connect()
    if cache:
        demo_production(cache)
        cache.clear()

    print(f"\n{'=' * 60}")
    print("✅ 演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
