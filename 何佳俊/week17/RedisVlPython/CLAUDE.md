# RedisVL Agent Cache — Claude Code 项目指南

## 项目概述

基于 Redis（低版本兼容）+ Milvus 的 LLM Agent 语义缓存组件。四个核心模块通过语义相似度匹配避免重复的大模型调用。

## 核心模块

| 模块 | 职责 | 关键依赖 |
|------|------|---------|
| `src/EmbeddingsCache.py` | 嵌入向量存储引擎，统一管理 Redis + Milvus | redis, pymilvus, numpy |
| `src/SemanticCache.py` | 语义缓存，相似度匹配 Q&A | EmbeddingsCache, dashscope / openai |
| `src/SemanticMessageHistory.py` | 对话历史管理，语义上下文检索 | EmbeddingsCache, redis |
| `src/SemanticRouter.py` | 查询意图路由 | EmbeddingsCache |

## Embedding 模型

**默认使用 Qwen（通义千问）DashScope API**：
- `text-embedding-v1` (1536 维)
- `text-embedding-v2` (1536 维，推荐)

两种调用方式（自动选择）：
1. **DashScope SDK** — `pip install dashscope`，设置 `DASHSCOPE_API_KEY`
2. **OpenAI 兼容接口** — 自动降级，使用 `openai` 库 + `api_key`

其他支持的 provider：`openai`、`sentence_transformers`

## 架构原则

1. **低版本兼容** — 兼容 Redis 6.x/7.x，无需 Redis Stack 模块（不依赖 RediSearch 模块）
2. **双存储后端** — Redis 存元数据和字符串，Milvus 存向量索引
3. **松耦合** — 四个模块通过依赖注入组合，可独立测试和使用
4. **容错设计** — Milvus 不可用时降级到 Redis 纯内存模式

## 开发规范

- **类型注解**：所有公开 API 必须包含完整类型标注（`typing` 模块）
- **文档字符串**：Google 风格 docstring（Args/Returns/Raises 三要素）
- **错误处理**：自定义异常层级，统一异常包装
- **日志**：使用 `logging` 模块，模块级 logger
- **测试**：pytest + pytest-cov，核心逻辑行覆盖率 > 90%

## 代码风格

```python
# Google 风格 docstring 示例
def lookup(self, query: str, threshold: float = None) -> Optional["CacheResult"]:
    """根据语义相似度查找缓存的回答。

    Args:
        query: 用户查询文本。
        threshold: 相似度阈值，覆盖默认值。

    Returns:
        如果找到匹配项返回 CacheResult，否则返回 None。

    Raises:
        ConnectionError: Redis/Milvus 连接失败。
    """
```

## 异常层级

```
CacheError (Base)
├── ConnectionError — Redis/Milvus 连接异常
├── EmbeddingError — Embedding 生成失败
├── SerializationError — 向量序列化/反序列化失败
├── NotFoundError — 键/向量不存在
└── ConfigurationError — 配置错误
```

## 测试策略

- 单元测试 mock Redis/Milvus 连接（使用 fakeredis + MagicMock）
- 集成测试需要实际的 Redis/Milvus 实例（通过环境变量控制）
- 每个模块的测试文件对应 `tests/test_<module>.py`

## 关键术语

- **Semantic Cache** — 语义缓存，通过向量相似度匹配的缓存策略
- **Embedding** — 嵌入向量，文本的数值表示
- **Cosine Similarity** — 余弦相似度，衡量向量间相似程度
- **Cache Hit** — 缓存命中，找到语义匹配的缓存项
- **Cache Miss** — 缓存未命中，需要调用 LLM 生成
- **Vector Index** — 向量索引，Milvus 中加速搜索的数据结构
- **TTL** — Time To Live，缓存项的生存时间
