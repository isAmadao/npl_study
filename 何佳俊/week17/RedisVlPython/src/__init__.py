"""
RedisVL Agent Cache — 基于 Redis + Milvus 的 LLM Agent 语义缓存组件。

核心模块：
    - EmbeddingsCache:        嵌入向量缓存引擎（Redis + Milvus 双后端）
    - SemanticCache:          语义相似度缓存（Q&A 匹配）
    - SemanticMessageHistory: 语义消息历史（对话上下文管理）
    - SemanticRouter:         语义路由（查询意图分发）
    - LLMClient:              大语言模型调用客户端（Qwen / OpenAI）
    - AutoTuner:              阈值自动调优模块
    - monitoring:              Web 监控面板（FastAPI）

版本：0.2.0
"""

__version__ = "0.2.0"

from .EmbeddingsCache import EmbeddingsCache
from .SemanticCache import SemanticCache
from .SemanticMessageHistory import SemanticMessageHistory
from .SemanticRouter import SemanticRouter
from .llm_client import LLMClient, Message, LLMResponse

__all__ = [
    "EmbeddingsCache",
    "SemanticCache",
    "SemanticMessageHistory",
    "SemanticRouter",
    "LLMClient",
    "Message",
    "LLMResponse",
]
