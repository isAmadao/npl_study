"""
Web 监控演示 — 启动带数据的监控面板

运行要求:
  pip install fastapi uvicorn
  export DASHSCOPE_API_KEY=sk-xxx
  python examples/monitoring_demo.py

打开 http://localhost:8000
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.EmbeddingsCache import EmbeddingsCache
from src.SemanticCache import SemanticCache
from src.SemanticMessageHistory import SemanticMessageHistory
from src.SemanticRouter import SemanticRouter

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


def generate_demo_data(embed_cache, embedding_fn):
    """生成演示数据。"""
    sem_cache = SemanticCache(cache=embed_cache, embedding_model=embedding_fn, similarity_threshold=0.6)
    history = SemanticMessageHistory(cache=embed_cache, embedding_func=embedding_fn)
    router = SemanticRouter(cache=embed_cache, embedding_func=embedding_fn, confidence_threshold=0.5)

    router.register_route("knowledge",
        ["什么是Redis", "Milvus是什么", "Python特点", "Docker"],
        description="技术知识问答")
    router.register_route("chitchat",
        ["你好", "你是谁", "谢谢"], description="日常对话")
    router.register_route("technical",
        ["部署失败", "报错排查", "性能优化"], description="技术支持")

    qa = [
        ("什么是 Redis？", "Redis 是一个开源的内存数据结构存储系统。"),
        ("Milvus 是什么", "Milvus 是一款开源向量数据库。"),
        ("Docker 是什么", "Docker 是一个开源的容器化平台。"),
        ("Python 有什么特点", "Python 是一种高级、解释型、面向对象的编程语言。"),
    ]
    for q, a in qa:
        sem_cache.store(q, a, metadata={"source": "demo"})
    print(f"   📦 {len(qa)} 条缓存数据")

    sessions = {
        "user_001": [
            ("user", "什么是 Redis？"),
            ("assistant", "Redis 是一个开源的内存数据结构存储系统。"),
            ("user", "Redis 一般用来做什么"),
        ],
        "user_002": [("user", "你好"), ("assistant", "你好！我是知识库助手。"), ("user", "Docker 是什么")],
    }
    for sid, msgs in sessions.items():
        for role, content in msgs:
            history.add_message(sid, role, content)
    print(f"   💬 {sum(len(v) for v in sessions.values())} 条对话消息")

    for q in ["什么是Redis", "Redis一般用来做什么", "你好啊", "Docker是什么"]:
        router.route(q)
        cache_hit = sem_cache.lookup(q)
        history.add_message("user_001", "user", q)
        history.add_message("user_001", "assistant",
                            cache_hit.answer if cache_hit else "模拟回答")

    return sem_cache, router, history


def main():
    if not (os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("EMBEDDING_API_KEY")):
        print("❌ 请设置 DASHSCOPE_API_KEY")
        return

    print("📊 Web 监控面板演示")
    print("=" * 60)

    try:
        embed_cache = EmbeddingsCache(embedding_dim=1536)
        print(f"   ✅ Redis 已连接")
    except Exception as e:
        print(f"❌ Redis 连接失败: {e}")
        return

    sem_cache, router, history = generate_demo_data(embed_cache, get_embedding())

    from src.monitoring import init_monitoring
    init_monitoring(cache=embed_cache, sem_cache=sem_cache, router=router, history=history)

    print(f"\n   → http://localhost:8000")
    print(f"   → http://localhost:8000/docs")
    print(f"   Ctrl+C 停止\n")

    import uvicorn
    uvicorn.run("src.monitoring:app", host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
