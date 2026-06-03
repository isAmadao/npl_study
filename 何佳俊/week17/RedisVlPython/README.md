# 🧠 RedisVL Agent Cache 系统

> **基于 Redis（低版本兼容） + Milvus 的 LLM Agent 语义缓存组件**
>
> 通过语义相似度匹配，为 AI Agent 调用提供智能缓存加速，避免重复的大模型调用，降低延迟与成本。

---

## 📋 目录

- [项目背景](#-项目背景)
- [系统架构](#-系统架构)
- [核心模块](#-核心模块)
  - [EmbeddingsCache](#1-embeddingscachepy-嵌入向量缓存)
  - [SemanticCache](#2-semanticcachepy-语义缓存)
  - [SemanticMessageHistory](#3-semanticmessagehistorypy-语义消息历史)
  - [SemanticRouter](#4-semanticrouterpy-语义路由)
- [快速开始](#-快速开始)
- [配置说明](#-配置说明)
- [API 文档](#-api-文档)
- [性能指标](#-性能指标)
- [项目结构](#-项目结构)
- [开发指南](#-开发指南)
- [许可证](#-许可证)

---

## 🎯 项目背景

在大语言模型（LLM）Agent 应用中，存在大量重复或高度相似的查询请求。每次请求都调用大模型 API 会导致：

| 问题 | 影响 |
|------|------|
| ⏱ **高延迟** | API 调用通常耗时 1~10 秒 |
| 💰 **高成本** | Token 消耗直接产生费用 |
| 🔄 **重复计算** | 相似的查询反复调用模型 |
| 📊 **资源浪费** | 相同的回答被反复生成 |

本项目通过 **语义缓存（Semantic Caching）** 技术解决上述问题：

1. 将历史 Q&A 及其 Embedding 向量存入 Redis + Milvus
2. 新查询到来时，计算其 Embedding 并与缓存进行语义相似度匹配
3. 匹配度超过阈值时直接返回缓存结果，跳过 LLM 调用
4. 不匹配时调用 LLM，并将新结果存入缓存

---

## 🏗 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                       用户 / Agent                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    SemanticRouter                            │
│               (语义路由 — 判断查询意图)                        │
└──────┬──────────────────────┬───────────────────────────────┘
       │                      │
       ▼                      ▼
┌──────────────┐    ┌──────────────────────────────────────┐
│  直接处理     │    │          SemanticCache                 │
│  (无需缓存)   │    │    (语义缓存 — 相似度匹配)               │
└──────────────┘    └──────┬───────────────────┬────────────┘
                          │                   │
                    缓存命中             缓存未命中
                          │                   │
                          ▼                   ▼
              ┌────────────────┐    ┌──────────────────┐
              │  返回缓存结果    │    │  调用 LLM 生成     │
              │  + 历史上下文    │    │   + 存入缓存       │
              └────────────────┘    └──────────────────┘
                          │                   │
                          └────────┬──────────┘
                                   ▼
              ┌──────────────────────────────────────┐
              │       EmbeddingsCache                 │
              │   (嵌入向量缓存 — 底层存储引擎)          │
              │   ┌────────────┐  ┌────────────┐      │
              │   │   Redis    │  │   Milvus   │      │
              │   │ (低版本兼容) │  │ (向量数据库) │      │
              │   └────────────┘  └────────────┘      │
              └──────────────────────────────────────┘
                                   ▲
                                   │
              ┌──────────────────────────────────────┐
              │     SemanticMessageHistory             │
              │   (语义消息历史 — 对话上下文管理)         │
              └──────────────────────────────────────┘
```

### 数据流说明

1. **查询路由** → `SemanticRouter` 分析查询意图，决定走哪个处理流程
2. **语义匹配** → `SemanticCache` 计算查询 Embedding，与缓存库进行向量相似度搜索
3. **缓存命中** → 直接返回缓存的回答，结合 `SemanticMessageHistory` 补充上下文
4. **缓存未命中** → 调用 LLM 生成回答，结果通过 `EmbeddingsCache` 存入 Redis + Milvus
5. **异步索引** → Embedding 向量异步写入 Milvus，元数据同步到 Redis

---

## 📦 核心模块

### 1. `EmbeddingsCache.py` — 嵌入向量缓存

**功能**：底层存储引擎，统一管理 Redis（低版本兼容）和 Milvus 的读写。

| 特性 | 说明 |
|------|------|
| 双存储后端 | Redis 存储元数据 + Milvus 存储向量索引 |
| 低版本兼容 | 兼容 Redis 6.x/7.x（无需 Redis Stack 模块） |
| TTL 管理 | 支持键级别的过期时间设置 |
| LRU 淘汰 | 基于 Redis 近似 LRU 的缓存淘汰策略 |
| 批量操作 | 支持向量和元数据的批量读写 |
| 连接池 | 复用 Redis/Milvus 连接，避免频繁创建 |

**核心 API**：

```python
class EmbeddingsCache:
    def get(self, key: str) -> Optional[np.ndarray]
    def set(self, key: str, vector: np.ndarray, metadata: dict = None, ttl: int = None)
    def delete(self, key: str) -> bool
    def exists(self, key: str) -> bool
    def clear(self) -> bool
    def get_stats(self) -> dict
    def batch_get(self, keys: List[str]) -> List[Optional[np.ndarray]]
    def batch_set(self, items: List[Tuple[str, np.ndarray, dict]])
```

---

### 2. `SemanticCache.py` — 语义缓存

**功能**：核心语义缓存模块，根据语义相似度匹配查询并返回缓存的 LLM 回答。

| 特性 | 说明 |
|------|------|
| 语义匹配 | 基于余弦相似度的向量搜索 |
| 动态阈值 | 可配置相似度阈值（默认 0.85） |
| Top-K 搜索 | 返回 K 个最相似的缓存结果 |
| 缓存策略 | 支持 LRU、LFU、TTL 组合策略 |
| 命中率统计 | 实时监控缓存命中/未命中情况 |
| 数据持久化 | 缓存数据重启后自动恢复 |

**核心 API**：

```python
class SemanticCache:
    def lookup(self, query: str, threshold: float = None) -> Optional[CacheResult]
    def store(self, question: str, answer: str, metadata: dict = None)
    def search(self, query: str, top_k: int = 5) -> List[CacheResult]
    def invalidate(self, query: str) -> bool
    def update_threshold(self, threshold: float)
    def get_stats(self) -> CacheStats
    def clear(self) -> bool
```

---

### 3. `SemanticMessageHistory.py` — 语义消息历史

**功能**：存储和管理 Agent 对话历史，支持基于语义的上下文检索。

| 特性 | 说明 |
|------|------|
| 会话管理 | 按 session_id 隔离不同对话 |
| 语义检索 | 从历史消息中找到语义相关的上下文 |
| 窗口管理 | 限制上下文窗口大小（Token / 条数） |
| 消息压缩 | 历史消息自动摘要压缩 |
| 多轮对话 | 完整支持多轮对话的上下文构建 |
| 过期清理 | 会话级别 TTL 自动清理 |

**核心 API**：

```python
class SemanticMessageHistory:
    def add_message(self, session_id: str, role: str, content: str, metadata: dict = None)
    def get_history(self, session_id: str, limit: int = None) -> List[Message]
    def search_similar(self, query: str, session_id: str, top_k: int = 5) -> List[Message]
    def get_context(self, session_id: str, max_tokens: int = 4096) -> str
    def clear_session(self, session_id: str) -> bool
    def list_sessions(self) -> List[str]
    def delete_session(self, session_id: str) -> bool
    def get_session_stats(self, session_id: str) -> dict
```

---

### 4. `SemanticRouter.py` — 语义路由

**功能**：基于语义相似度将查询路由到不同的处理流程或模块。

| 特性 | 说明 |
|------|------|
| 意图分类 | 基于示例查询定义路由意图 |
| 动态注册 | 运行时注册新的路由和处理函数 |
| 兜底机制 | 无匹配时使用默认路由 |
| 置信度评分 | 返回匹配的置信度分数 |
| 路由统计 | 各路由调用频率统计 |
| 热加载 | 支持配置文件变更后重新加载路由 |

**核心 API**：

```python
class SemanticRouter:
    def route(self, query: str) -> RouteResult
    def register_route(self, name: str, examples: List[str], handler: Callable, description: str = None)
    def add_examples(self, route_name: str, examples: List[str])
    def remove_route(self, route_name: str) -> bool
    def list_routes(self) -> List[RouteInfo]
    def get_route_stats(self) -> dict
    def reload_routes(self, config_path: str = None)
```

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- Redis 6.x / 7.x（低版本兼容，无需 Redis Stack）
- Milvus 2.x（Standalone 或 Cluster）
- 支持的 Embedding 模型：
  - **Qwen**（通义千问）`text-embedding-v1` / `text-embedding-v2`（默认推荐）
  - OpenAI `text-embedding-ada-002` / `text-embedding-3-small` / `text-embedding-3-large`
  - 本地模型（如 `BAAI/bge-*`、`GanymedeNil/text2vec-large-chinese` 等）

### 安装

```bash
# 克隆项目
git clone <repo-url>
cd RedisVlPython

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件填写 Redis、Milvus 和 Embedding 配置
```

### 基础使用

```python
from src.cache import EmbeddingsCache, SemanticCache
from src.router import SemanticRouter
from src.history import SemanticMessageHistory

# 1. 初始化 Embedding 缓存
emb_cache = EmbeddingsCache(
    redis_url="redis://localhost:6379",
    milvus_host="localhost",
    milvus_port=19530,
    embedding_dim=768
)

# 2. 初始化语义缓存
sem_cache = SemanticCache(
    cache=emb_cache,
    embedding_model="qwen",              # 使用 Qwen 通义千问
    similarity_threshold=0.85
)

# 3. 存储 Q&A
sem_cache.store("什么是 Redis？", "Redis 是一个开源的内存数据结构存储系统...")

# 4. 语义搜索
result = sem_cache.lookup("请解释一下 Redis 是什么")
if result:
    print(f"缓存命中! 相似度: {result.similarity:.4f}")
    print(f"回答: {result.answer}")
else:
    print("缓存未命中，调用 LLM...")
```

> 详细示例请参见 [`examples/`](./examples/) 目录。

---

## ⚙️ 配置说明

### 环境变量（.env）

```ini
# === Redis 配置 (低版本兼容) ===
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=
REDIS_SOCKET_TIMEOUT=5
REDIS_SOCKET_CONNECT_TIMEOUT=5
REDIS_MAX_CONNECTIONS=50

# === Milvus 配置 ===
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_ALIAS=default
MILVUS_COLLECTION=agent_cache
MILVUS_INDEX_TYPE=IVF_FLAT
MILVUS_METRIC_TYPE=IP
MILVUS_NLIST=1024

# === Embedding 配置 ===
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_DIM=1536
EMBEDDING_API_KEY=sk-xxx
EMBEDDING_API_BASE=https://api.openai.com/v1

# === Semantic Cache 配置 ===
CACHE_SIMILARITY_THRESHOLD=0.85
CACHE_DEFAULT_TTL=86400
CACHE_MAX_SIZE=10000
CACHE_EVICTION_POLICY=lru

# === Semantic Router 配置 ===
ROUTER_CONFIDENCE_THRESHOLD=0.75
ROUTER_DEFAULT_ROUTE=fallback
```

> 详细配置说明参见 [`docs/configuration.md`](./docs/configuration.md)

---

## 📖 API 文档

各模块的完整 API 文档：

| 文档 | 说明 |
|------|------|
| [`docs/embeddings_cache.md`](./docs/embeddings_cache.md) | EmbeddingsCache 详细 API 与使用示例 |
| [`docs/semantic_cache.md`](./docs/semantic_cache.md) | SemanticCache 缓存策略与调优指南 |
| [`docs/message_history.md`](./docs/message_history.md) | SemanticMessageHistory 会话管理 |
| [`docs/semantic_router.md`](./docs/semantic_router.md) | SemanticRouter 路由规则与配置 |

---

## 📈 性能指标

| 场景 | 直接调用 LLM | 使用语义缓存 | 提升 |
|------|-------------|-------------|------|
| 精确匹配 | 1~5 秒 | <10 毫秒 | **100~500x** |
| 语义相似（>0.9） | 1~5 秒 | <50 毫秒 | **20~100x** |
| 语义相似（>0.8） | 1~5 秒 | <100 毫秒 | **10~50x** |
| 高并发场景 | 受限 API 限速 | 轻松处理千级 QPS | — |

> 以上数据基于测试环境，实际性能取决于配置、网络和数据量。

---

## 📁 项目结构

```
RedisVlPython/
├── README.md                    # 项目说明（本文件）
├── CLAUDE.md                    # Claude Code 项目指令
├── main.py                      # 入口文件
├── requirements.txt             # Python 依赖
├── .env.example                 # 环境变量模板
├── setup.py                     # 安装脚本
│
├── src/                         # 核心源码
│   ├── __init__.py
│   ├── EmbeddingsCache.py       # 嵌入向量缓存
│   ├── SemanticCache.py         # 语义缓存
│   ├── SemanticMessageHistory.py# 语义消息历史
│   ├── SemanticRouter.py        # 语义路由
│   └── config.py                # 配置管理
│
├── tests/                       # 单元测试
│   ├── __init__.py
│   ├── test_embeddings_cache.py
│   ├── test_semantic_cache.py
│   ├── test_message_history.py
│   └── test_semantic_router.py
│
├── docs/                        # 文档
│   ├── configuration.md
│   ├── embeddings_cache.md
│   ├── semantic_cache.md
│   ├── message_history.md
│   └── semantic_router.md
│
├── examples/                    # 使用示例
│   ├── basic_usage.py
│   ├── semantic_cache_demo.py
│   └── router_demo.py
│
└── .claude/                     # Claude Code 配置
    └── settings.json
```

---

## 💻 开发指南

### 开发环境搭建

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装开发依赖
pip install -r requirements.txt
pip install pytest pytest-cov mypy black isort

# 启动 Redis (Docker)
docker run -d --name redis -p 6379:6379 redis:7

# 启动 Milvus (Docker)
docker compose -f docker-compose.yml up -d
```

### 代码规范

- 类型注解：所有公开 API 必须包含完整类型标注
- 文档字符串：使用 Google 风格 docstring
- 测试覆盖：核心逻辑行覆盖率 > 90%
- 命名规范：类名使用 PascalCase，函数/变量使用 snake_case

### 测试运行

```bash
# 运行所有测试
pytest tests/ -v

# 运行指定测试
pytest tests/test_semantic_cache.py -v

# 带覆盖率报告
pytest tests/ --cov=src/ --cov-report=html
```

---

## 🔮 未来规划

- [x] 基础语义缓存架构
- [x] Redis + Milvus 双存储后端
- [ ] 缓存预热机制
- [ ] 自动 Embedding 模型切换
- [ ] 分布式缓存同步
- [ ] 缓存命中率预测
- [ ] 可视化监控面板
- [ ] 多模态缓存支持（图文）

---

## 📄 许可证

本项目仅供学习研究使用。

---

<p align="center">
  <b>RedisVL Agent Cache</b> — 让 AI Agent 更快、更省、更聪明 🚀
</p>
