# 生产部署指南

## 快速启动

### 1. 启动依赖服务

```bash
docker compose up -d
```

此命令会启动：
- **Redis** (端口 6379) — 缓存元数据存储
- **Milvus** (端口 19530) — 向量数据库
- **RedisInsight** (端口 5540，可选) — Redis 图形化管理界面

验证服务状态：

```bash
redis-cli ping
# → PONG

python -c "from pymilvus import MilvusClient; c = MilvusClient('http://localhost:19530'); print(c.list_collections())"
# → []
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

关键配置项：

| 变量 | 推荐值 | 说明 |
|------|--------|------|
| `REDIS_URL` | `redis://redis:6379/0` | Docker 内使用 service 名称 |
| `MILVUS_URI` | `http://milvus:19530` | Docker 内使用 service 名称 |
| `EMBEDDING_API_KEY` | `sk-xxx` | 你的 API Key |
| `CACHE_SIMILARITY_THRESHOLD` | `0.85` | 相似度阈值 |

### 3. 运行示例

```bash
python examples/production_deploy.py
```

---

## Docker Compose 详解

### 服务拓扑

```
┌─────────────┐     ┌─────────────┐
│   Redis     │─────│  RedisInsight│
│  (6379)     │     │  (5540)     │
└──────┬──────┘     └─────────────┘
       │
       │ 本地网络
       │
┌──────┴──────┐
│   Milvus    │
│  (19530)    │
│  (9091)     │
└─────────────┘
```

### 网络

三个服务通过 `redisvl-network` 桥接网络互通。在 Docker 内部，服务可以通过名称相互访问（如 `redis://redis:6379`）。

### 数据持久化

| 服务 | 挂载卷 | 说明 |
|------|--------|------|
| Redis | `redis-data:/data` | RDB + AOF 持久化 |
| Milvus | `milvus-data:/var/lib/milvus` | 向量数据持久化 |
| RedisInsight | `redis-insight-data:/data` | 配置持久化 |

### 健康检查

所有服务均配置了 healthcheck：

```bash
# 检查所有服务状态
docker compose ps

# 查看各服务健康日志
docker compose logs redis | grep health
docker compose logs milvus | grep health
```

---

## 配置优化

### Redis 配置

参考 `redis.conf` 文件，关键优化点：

```conf
maxmemory 512mb                    # 根据可用内存调整
maxmemory-policy allkeys-lru       # LRU 淘汰策略
appendonly yes                     # 开启 AOF 持久化
appendfsync everysec               # 每秒同步
```

**内存估算**：每个缓存条目约占用：
- 向量数据：`维度 × 4 bytes`（如 1536 维 ≈ 6 KB）
- 元数据：`~500 bytes`
- 总计：`~10 KB/条`

100 万条缓存条目 ≈ 10 GB 内存。

### Milvus 配置

Milvus 使用嵌入式 etcd 和 minio（适用于单机部署）。生产环境建议：

1. **分离 etcd 和 minio**为独立服务
2. **配置资源限制**：

```yaml
milvus:
  deploy:
    resources:
      limits:
        memory: 8G
```

3. **调整索引参数**：

```python
# 根据数据量调整 nlist
MILVUS_NLIST=1024       # < 100 万条
MILVUS_NLIST=4096       # > 100 万条
```

---

## 监控与运维

### 健康检查端点

应用层健康检查：

```python
from src.EmbeddingsCache import EmbeddingsCache

cache = EmbeddingsCache()

# Redis 健康
cache.redis.ping()

# Milvus 健康
cache.milvus_client.list_collections() if cache.milvus_available else None
```

### 关键指标

| 指标 | 采集方式 | 告警阈值 |
|------|----------|---------|
| Redis 内存使用 | `INFO memory` | > 80% maxmemory |
| Redis 命中率 | `INFO stats` | < 80% |
| Milvus 查询延迟 | 应用层统计 | > 100ms |
| 缓存命中率 | `get_stats()` | < 20% |
| LLM 调用成本 | `LLMClient.stats` | 按预算 |

### 备份策略

```bash
# Redis RDB 备份
cp /data/dump.rdb backup_$(date +%Y%m%d).rdb

# 使用 RedisInsight 导出
# 或直接备份 Docker 卷
docker run --rm -v redisvl_redis-data:/data -v $(pwd):/backup alpine cp /data/dump.rdb /backup/
```

---

## 扩容建议

### 垂直扩容（推荐）

| 组件 | 建议规格 |
|------|----------|
| Redis | 4C8G+（100 万条以内） |
| Milvus | 8C16G+（1000 万条以内） |

### 水平扩容

- Redis：使用 Redis Cluster（6.x 以上原生支持）
- Milvus：使用 Milvus Cluster（需独立 etcd + minio + rootcoord）

---

## 常见问题

### Q: 缓存命中率低怎么办？

1. 调低 `CACHE_SIMILARITY_THRESHOLD`（每次降 0.05）
2. 检查 Embedding 模型质量
3. 使用 Task 4 的 AutoTuner 自动调优

### Q: Redis 内存溢出？

1. 配置 `maxmemory` 和淘汰策略
2. 缩短 `CACHE_DEFAULT_TTL`
3. 减少 `CACHE_MAX_SIZE`

### Q: Milvus 连接超时？

1. 检查网络连通性
2. 增加连接超时配置
3. 系统会自动降级为纯 Redis 模式（暴力搜索）
