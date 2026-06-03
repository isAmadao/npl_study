# EmbeddingsCache 文档

## 概述

EmbeddingsCache 是系统的底层存储引擎，统一管理 Redis 和 Milvus 的向量读写。

## 架构

```
应用层
   │
   ├─ Redis (String):  向量序列化二进制 + 元数据 Hash
   ├─ Milvus:          FLOAT_VECTOR 向量索引 + ANN 搜索
   └─ 降级模式:         Milvus 不可用时 → 纯 Redis 内存模式
```

## API 参考

### `__init__(redis_client, milvus_host, milvus_port, embedding_dim, settings)`

- 支持注入外部 Redis 客户端实例
- 自动连接 Milvus，失败时静默降级

### `get(key) -> np.ndarray | None`

从 Redis 读取序列化的向量。

### `set(key, vector, metadata, ttl) -> bool`

写入向量。自动同步到 Milvus（如果可用）。

### `delete(key) -> bool`

删除向量。同时清理 Redis 和 Milvus 中的数据。

### `get_stats() -> dict`

返回缓存统计信息，包含命中率、缓存大小等。

## 使用场景

- 作为 SemanticCache 和 SemanticMessageHistory 的底层存储
- 直接管理 Embedding 向量
- 需要精细控制 TTL 和淘汰策略的场景
