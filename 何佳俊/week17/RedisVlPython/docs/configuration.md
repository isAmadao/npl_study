# 配置说明

## 配置方式

本系统支持两种配置方式（优先级递减）：

1. **环境变量**（最高优先级）
2. **`.env` 文件**（推荐方式）
3. **`Settings` 类默认值**

## 配置项参考

### Redis 配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis 连接 URL |
| `REDIS_PASSWORD` | (空) | Redis 认证密码 |
| `REDIS_SOCKET_TIMEOUT` | `5` | Socket 超时时间（秒） |
| `REDIS_SOCKET_CONNECT_TIMEOUT` | `5` | 连接超时时间（秒） |
| `REDIS_MAX_CONNECTIONS` | `50` | 连接池最大连接数 |
| `REDIS_DECODE_RESPONSES` | `true` | 自动解码响应为字符串 |

### Milvus 配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `MILVUS_HOST` | `localhost` | Milvus 服务地址 |
| `MILVUS_PORT` | `19530` | Milvus 服务端口 |
| `MILVUS_ALIAS` | `default` | 连接别名 |
| `MILVUS_COLLECTION` | `cache_vectors` | 集合名称 |
| `MILVUS_INDEX_TYPE` | `IVF_FLAT` | 索引类型 |
| `MILVUS_METRIC_TYPE` | `IP` | 距离度量（IP/COSINE） |
| `MILVUS_NLIST` | `1024` | 索引聚类参数 |

### Embedding 配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `EMBEDDING_PROVIDER` | `openai` | Embedding 提供方 |
| `EMBEDDING_MODEL` | `text-embedding-ada-002` | 模型名称 |
| `EMBEDDING_DIM` | `1536` | 向量维度 |
| `EMBEDDING_API_KEY` | (空) | API 密钥 |
| `EMBEDDING_API_BASE` | `https://api.openai.com/v1` | API 基础地址 |

### Cache 配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `CACHE_SIMILARITY_THRESHOLD` | `0.85` | 相似度阈值 |
| `CACHE_DEFAULT_TTL` | `86400` | 默认 TTL（24 小时） |
| `CACHE_MAX_SIZE` | `10000` | 最大缓存条目数 |
| `CACHE_EVICTION_POLICY` | `lru` | 淘汰策略 |
| `CACHE_TOP_K` | `5` | Top-K 搜索结果数 |

## 低版本 Redis 配置说明

本项目兼容 Redis 6.x/7.x，**不依赖 Redis Stack 模块**。

关键点：
- 不使用 RediSearch 的向量搜索功能
- 向量存储使用 Redis String + pickle 序列化
- 向量的暴力搜索在应用层 Python 完成
- 适合中小规模数据（< 10 万条）
