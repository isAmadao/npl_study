"""
真实场景：智能知识库 Agent（带语义缓存 + 路由 + 对话历史）
=========================================================
模拟一个企业知识库智能问答系统，用户提问后：
  1. SemanticRouter 判断问题类别（知识问答 / 闲聊 / 技术支持）
  2. SemanticCache 查找是否已有相似问答的缓存
  3. 缓存命中 → 直接返回（零成本）
  4. 缓存未命中 → 调用 Qwen LLM → 存入缓存
  5. SemanticMessageHistory 记录对话上下文

运行要求：
  - Redis 服务运行中（默认 localhost:6379）
  - DASHSCOPE_API_KEY 或 EMBEDDING_API_KEY 已设置

运行方式：
  export DASHSCOPE_API_KEY=sk-xxx
  python examples/real_world_demo.py
"""

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.EmbeddingsCache import EmbeddingsCache
from src.SemanticCache import SemanticCache
from src.SemanticMessageHistory import SemanticMessageHistory
from src.SemanticRouter import SemanticRouter

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("pymilvus").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


# ==================== 1. Embedding ====================


def qwen_embedding(text: str) -> np.ndarray:
    """通过 DashScope SDK 调用 Qwen text-embedding-v2。

    Args:
        text: 输入文本。

    Returns:
        1536 维归一化向量。

    Raises:
        RuntimeError: API Key 未设置或调用失败。
    """
    from dashscope import TextEmbedding

    api_key = (
        os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("EMBEDDING_API_KEY")
    )
    if not api_key:
        raise RuntimeError("请设置 DASHSCOPE_API_KEY 或 EMBEDDING_API_KEY")
    resp = TextEmbedding.call(
        model="text-embedding-v2",
        input=text,
        api_key=api_key,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"DashScope API 错误: {resp.message}")
    return np.array(resp.output["embeddings"][0]["embedding"], dtype=np.float32)


# ==================== 2. LLM ====================


class LLM:
    """通过 LLMClient 调用 Qwen 对话 API。"""

    def __init__(self):
        from src.llm_client import LLMClient

        self.client = LLMClient(model="qwen-turbo", enable_stats=True)
        self.model = self.client.model

    def ask(self, question: str) -> Tuple[str, Dict[str, Any]]:
        """调用 Qwen 生成回答。

        Returns:
            (回答文本, 含 tokens/cost/latency 的统计字典)。
        """
        response = self.client.chat(
            messages=[{"role": "user", "content": question}],
            temperature=0.7,
            max_tokens=500,
        )
        return response.text, response.__dict__

    def get_stats(self) -> Dict[str, Any]:
        return self.client.get_stats().to_dict()


# ==================== 3. 存储引擎 ====================


def create_storage() -> EmbeddingsCache:
    """创建 EmbeddingsCache（必须使用真实 Redis）。

    Returns:
        EmbeddingsCache 实例。

    Raises:
        RuntimeError: Redis 连接失败。
    """
    try:
        cache = EmbeddingsCache(embedding_dim=1536)
    except Exception as e:
        raise RuntimeError(f"Redis 连接失败，请确保 Redis 已启动: {e}")

    print(f"   ✅ Redis: 已连接")
    if cache.milvus_available:
        print(f"   ✅ Milvus: 可用")
    else:
        print(f"   ℹ️  Milvus: 不可用（使用 Redis 暴力搜索）")
    return cache


# ==================== 4. 缓存预热 ====================


def prewarm_cache(sem_cache: SemanticCache) -> int:
    """预热语义缓存，让初始对话即有缓存命中。"""
    prewarm_qa = [
        ("什么是 Redis？",
         "Redis 是一个开源的内存数据结构存储系统，常用作数据库、缓存和消息中间件。"),
        ("Redis 有哪些应用场景",
         "Redis 常用于缓存加速、会话管理、消息队列、实时排行榜等场景。"),
        ("Docker 是什么？",
         "Docker 是一个开源的容器化平台，用于开发、交付和运行应用程序。"),
        ("Docker 和虚拟机的区别",
         "Docker 容器共享宿主机内核，启动快、资源占用少；虚拟机有独立内核，隔离性更强。"),
        ("Python 的特点有哪些",
         "Python 是一种高级、解释型、面向对象的编程语言，语法简洁清晰。"),
        ("Milvus 是什么",
         "Milvus 是一款开源向量数据库，专为 AI 应用设计，支持万亿级向量相似性搜索。"),
        ("语义缓存是什么",
         "语义缓存（Semantic Cache）是一种基于向量相似度的缓存策略，"
         "可减少 30%~80% 的 LLM API 调用成本。"),
        ("容器化部署失败怎么排查",
         "容器化部署失败排查步骤：1. 检查容器日志 2. 确认网络配置 3. 验证资源限制 4. 检查依赖版本。"),
    ]
    count = 0
    for q, a in prewarm_qa:
        try:
            sem_cache.store(q, a, metadata={"source": "prewarm"})
            count += 1
        except Exception as e:
            logger.debug("预热失败 (%s): %s", q[:20], e)
    if count > 0:
        print(f"   ✅ 预热缓存: {count}/{len(prewarm_qa)} 条 Q&A")
    return count


# ==================== 5. 最相似度查询 ====================


def best_similarity(sem_cache: SemanticCache, query: str) -> Optional[float]:
    """返回与缓存中最相似问题的相似度（不判断阈值）。"""
    try:
        results = sem_cache.search(query, top_k=1)
        return results[0].similarity if results else None
    except Exception:
        return None


# ==================== 6. 路由处理 ====================


def handle_knowledge(query: str, llm: LLM) -> Tuple[str, Dict[str, Any]]:
    text, stats = llm.ask(query)
    return text, {
        "tokens": stats.get("total_tokens", 0),
        "cost": stats.get("cost", 0),
        "latency": stats.get("latency", 0),
    }


def handle_chitchat(query: str) -> Tuple[str, None]:
    """闲聊路由（预设回复，零成本）。"""
    replies = {
        "你好": "你好！我是知识库助手，有什么可以帮助你的？",
        "你是谁": "我是基于语义缓存的智能知识库助手。",
        "你能做什么": "我可以回答技术问题、提供知识查询、帮你排查部署问题！",
        "你会什么": "我可以回答技术问题、提供知识查询、帮你排查部署问题！",
        "谢谢": "不客气！有其他问题随时问我。",
        "感谢": "不客气！有其他问题随时问我。",
    }
    for key, reply in replies.items():
        if key in query:
            return reply, None
    return f"哈哈，{query} 是个好问题！我们聊点技术相关的吧？", None


def handle_technical(query: str, llm: LLM) -> Tuple[str, Dict[str, Any]]:
    text, stats = llm.ask(query)
    return f"[技术支持]\n{text}", {
        "tokens": stats.get("total_tokens", 0),
        "cost": stats.get("cost", 0),
        "latency": stats.get("latency", 0),
    }


# ==================== 7. 主流程 ====================


def main():
    """运行智能知识库 Agent 演示。"""
    print("=" * 70)
    print("🚀 智能知识库 Agent — 真实场景演示")
    print("   核心链路: Qwen Embedding → 语义缓存 → Qwen LLM → Redis 持久化")
    print("=" * 70)

    # ----- 存储引擎 -----
    print("\n📦 正在初始化存储引擎...")
    embed_cache = create_storage()

    # ----- Embedding -----
    print(f"\n🧠 正在初始化 Embedding 模型...")
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("EMBEDDING_API_KEY")
    if not api_key:
        print("❌ 未设置 DASHSCOPE_API_KEY")
        print("   请执行: export DASHSCOPE_API_KEY=sk-xxx")
        return

    try:
        test_vec = qwen_embedding("test")
        embedding_fn = qwen_embedding
        embedding_dim = len(test_vec)
        print(f"   ✅ Qwen text-embedding-v2 ({embedding_dim} 维)")
    except Exception as e:
        print(f"❌ Embedding 初始化失败: {e}")
        return

    # 维度校验
    if embed_cache.dim != embedding_dim:
        print(f"   缓存维度 ({embed_cache.dim}) != 模型维度 ({embedding_dim})，重新创建缓存...")
        from redis import Redis
        r = Redis(decode_responses=True)
        embed_cache = EmbeddingsCache(
            redis_client=r, embedding_dim=embedding_dim, skip_milvus=True,
        )

    # ----- 阈值 -----
    SIMILARITY_THRESHOLD = 0.62
    ROUTER_THRESHOLD = 0.50

    # 清空旧数据
    try:
        embed_cache.redis.flushdb()
        print("   🧹 已清理 Redis 旧数据")
    except Exception:
        pass

    # ----- 语义缓存 -----
    sem_cache = SemanticCache(
        cache=embed_cache,
        embedding_model=embedding_fn,
        similarity_threshold=SIMILARITY_THRESHOLD,
    )
    print(f"\n🎯 语义缓存")
    print(f"   相似度阈值: {sem_cache.similarity_threshold}")
    prewarm_cache(sem_cache)

    # ----- 消息历史 -----
    history = SemanticMessageHistory(cache=embed_cache, embedding_func=embedding_fn)
    print(f"   消息历史: 最多 {history.max_history} 条")

    # ----- 路由 -----
    router = SemanticRouter(
        cache=embed_cache,
        embedding_func=embedding_fn,
        confidence_threshold=ROUTER_THRESHOLD,
    )
    router.register_route(
        "knowledge",
        ["什么是Redis", "Milvus是什么", "Python有什么特点",
         "解释Docker", "语义缓存是什么", "向量数据库",
         "什么是缓存", "Redis使用场景", "容器技术介绍"],
        description="技术知识问答",
    )
    router.register_route(
        "chitchat",
        ["你好", "你是谁", "谢谢", "很高兴认识你",
         "你能做什么", "你会什么", "今天天气", "再见"],
        description="日常对话",
    )
    router.register_route(
        "technical",
        ["容器部署失败", "系统报错怎么排查", "性能如何优化",
         "环境部署问题", "应用崩溃", "服务宕机"],
        description="技术问题排查",
    )
    print(f"   语义路由: {len(router.routes)} 条路由")
    for name, info in router.routes.items():
        print(f"     - {name}: {len(info.examples)} 个示例")

    # ----- LLM -----
    print(f"\n🤖 正在初始化 LLM...")
    try:
        llm = LLM()
        print(f"   ✅ qwen-turbo")
    except Exception as e:
        print(f"❌ LLM 初始化失败: {e}")
        return

    # ----- 对话 -----
    total_questions = 0
    cache_hits = 0
    total_tokens_saved = 0
    session_id = "demo_user_001"

    conversations = [
        ("你好！", "chitchat"),
        ("什么是 Redis？", "knowledge"),
        ("Redis 一般用来做什么", "knowledge"),
        ("Docker 是什么？", "knowledge"),
        ("docker 怎么用", "knowledge"),
        ("你能做什么？", "chitchat"),
        ("Python 的特点有哪些", "knowledge"),
        ("python 是解释型语言吗", "knowledge"),
        ("容器化部署失败了怎么办", "technical"),
        ("谢谢你的帮助", "chitchat"),
        ("语义缓存是什么", "knowledge"),
        ("Milvus 向量数据库有什么特点", "knowledge"),
    ]

    print("\n" + "=" * 70)
    print(f"💬 开始对话（共 {len(conversations)} 轮）")
    print("   ✅ 绿色 = 缓存命中（零成本） | 🔴 红色 = 调用 LLM")
    print("=" * 70)

    for i, (question, expected_route) in enumerate(conversations, 1):
        print(f"\n{'─' * 60}")
        print(f"👤 [{i}/{len(conversations)}] 用户: {question}")

        route_result = router.route(question)
        route_name = route_result.route_name
        is_fallback = route_result.is_fallback

        cache_result = sem_cache.lookup(question)
        total_questions += 1
        answer = None
        token_info = None

        if cache_result:
            cache_hits += 1
            answer = cache_result.answer
            total_tokens_saved += 200
            print(f"   📍 {route_name} {is_fallback * '⚠️兜底'}")
            print(f"   ✅ 缓存命中! (相似度: {cache_result.similarity:.2%})")
            print(f"   🤖 {answer[:150]}...")
            print(f"   💰 节省 ~200 tokens")
        else:
            best_sim = best_similarity(sem_cache, question)

            if route_name == "chitchat":
                answer, _ = handle_chitchat(question)
                print(f"   📍 {route_name} (闲聊 → 零成本)")
                print(f"   💬 {answer[:150]}...")
            else:
                answer, token_info = handle_knowledge(question, llm)
                tag = " ⚠️兜底" if is_fallback else ""
                print(f"   📍 {route_name}{tag}")
                if best_sim is not None:
                    print(f"   🔴 未命中 (最佳相似度: {best_sim:.2%}, 阈值: {SIMILARITY_THRESHOLD})")
                else:
                    print(f"   🔴 未命中 (缓存为空)")
                print(f"   🤖 qwen-turbo: {answer[:150]}...")
                if token_info:
                    print(f"   📊 Token: {token_info['tokens']} | "
                          f"¥{token_info['cost']:.6f} | {token_info['latency']:.2f}s")

            sem_cache.store(question, answer, metadata={
                "route": route_name, "model": "qwen-turbo",
            })

        history.add_message(session_id, "user", question)
        history.add_message(session_id, "assistant", answer)

    # ----- 统计 -----
    print("\n" + "=" * 70)
    print("📊 统计报告")
    print("=" * 70)

    cache_stats = sem_cache.get_stats()
    hit_rate = cache_hits / total_questions * 100
    final_stats = llm.get_stats()

    print(f"\n🎯 语义缓存")
    print(f"   总请求: {total_questions}")
    print(f"   缓存命中: {cache_hits}")
    print(f"   命中率: {hit_rate:.1f}%")
    print(f"   节省 LLM 调用: {cache_hits} 次")
    print(f"   节省 Token: ~{total_tokens_saved}")
    print(f"   缓存条目: {cache_stats.size}")

    print(f"\n🧭 语义路由")
    router_stats = router.get_route_stats()
    for name, count in sorted(router_stats.route_counts.items()):
        print(f"   {name}: {count} 次 ({count/router_stats.total_queries*100:.0f}%)")
    print(f"   兜底率: {router_stats.fallback_rate:.0%}")

    print(f"\n💬 对话")
    s = history.get_session_stats(session_id)
    print(f"   消息数: {s['message_count']}")

    print(f"\n💰 成本")
    print(f"   qwen-turbo 调用: {final_stats['total_calls']} 次")
    print(f"   总 Token: {final_stats['total_tokens']}")
    print(f"   总成本: ¥{final_stats['total_cost']:.6f}")
    print(f"   总耗时: {final_stats['total_latency']:.2f}s")
    estimated = total_questions * 300
    est_cost = estimated * 0.0003 / 1000 * 0.8
    saving = (1 - final_stats['total_cost'] / est_cost) * 100 if est_cost > 0 else 0
    print(f"\n   📈 无缓存反事实: ~{estimated} Token, ¥{est_cost:.6f}")
    print(f"   📈 节省比例: {saving:.1f}%")

    print(f"\n{'=' * 70}")
    print("🏁 演示结束")
    print(f"   💾 Redis 持久化 ✅ | LLM: qwen-turbo ✅ | Embedding: Qwen v2 ✅")
    print(f"   ⚡ 相似问题自动命中缓存 → 零成本秒回")
    print("=" * 70)


if __name__ == "__main__":
    main()
