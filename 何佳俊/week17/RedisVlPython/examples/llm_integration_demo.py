"""
LLM 集成示例 — 展示 LLMClient 的完整功能
=========================================

演示内容：
  1. 非流式对话（单轮 / 多轮）
  2. 流式对话（实时输出）
  3. Token 计数和成本统计
  4. 多 Provider 切换

运行要求：
  DASHSCOPE_API_KEY 或 LLM_API_KEY 或 OPENAI_API_KEY
"""

import logging
import os
import sys
import time
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_client import (
    LLMClient,
    LLMProvider,
    Message,
    MessageRole,
    LLMClientError,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def get_api_key() -> Optional[str]:
    for k in ["DASHSCOPE_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY", "EMBEDDING_API_KEY"]:
        v = os.environ.get(k)
        if v:
            return v
    return None


def demo_basic_chat(llm: LLMClient):
    """单轮对话。"""
    print("\n" + "=" * 60)
    print("📝 单轮对话")
    print("=" * 60)

    for q in ["什么是 Redis？", "Milvus 有哪些主要特性？"]:
        print(f"\n👤 {q}")
        r = llm.chat(messages=[Message(role=MessageRole.USER, content=q)])
        print(f"🤖 {r.text[:150]}...")
        print(f"   📊 {r.total_tokens} tokens | ¥{r.cost:.6f} | {r.latency:.2f}s")


def demo_streaming(llm: LLMClient):
    """流式对话。"""
    print("\n" + "=" * 60)
    print("🌊 流式对话")
    print("=" * 60)

    print(f"\n👤 请用三句话介绍 Python")
    print("🤖 ", end="", flush=True)

    collected = ""
    for chunk in llm.chat_stream(
        messages=[Message(role=MessageRole.USER, content="请用三句话介绍 Python")],
        temperature=0.7, max_tokens=200,
    ):
        print(chunk, end="", flush=True)
        collected += chunk
    print()


def demo_multi_turn(llm: LLMClient):
    """多轮对话。"""
    print("\n" + "=" * 60)
    print("💬 多轮对话")
    print("=" * 60)

    questions = [
        "什么是向量数据库？",
        "它和传统数据库有什么区别？",
        "有哪些流行的向量数据库？",
    ]

    for i, q in enumerate(questions):
        print(f"\n👤 ({i+1}) {q}")
        r = llm.chat(
            messages=[Message(role=MessageRole.USER, content=q)],
            system_prompt="你是一个技术助手，用简洁的语言回答问题。" if i == 0 else None,
        )
        print(f"🤖 {r.text[:150]}...")
        print(f"   📊 {r.total_tokens} tokens | ¥{r.cost:.6f}")


def demo_stats(llm: LLMClient):
    """统计信息。"""
    print("\n" + "=" * 60)
    print("💰 调用统计")
    print("=" * 60)

    for text in ["Hello", "Redis 是一个开源的内存数据库。", "基于 Redis 的语义缓存。" * 5]:
        print(f"   「{text[:30]}...」→ ~{llm.count_tokens(text)} tokens")

    s = llm.get_stats()
    print(f"\n📊 全局累计:")
    print(f"   调用: {s.total_calls} 次")
    print(f"   Token: {s.total_input_tokens + s.total_output_tokens}")
    print(f"   成本: ¥{s.total_cost:.6f}")
    print(f"   模型分布: {s.model_breakdown}")


def main():
    key = get_api_key()
    if not key:
        print("❌ 未设置 API Key")
        print("   export DASHSCOPE_API_KEY=sk-xxx")
        return

    llm = LLMClient(model="qwen-turbo", enable_stats=True)
    print(f"✅ LLMClient: {llm.model} ({llm.provider.value})")

    demo_basic_chat(llm)
    demo_streaming(llm)
    demo_multi_turn(llm)
    demo_stats(llm)

    print(f"\n{'=' * 60}")
    print("✅ 演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
