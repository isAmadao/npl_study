# 项目开发规格书 (Spec)

## 概述

基于 Redis（低版本兼容）+ Milvus 的 LLM Agent 语义缓存组件，已完成四个核心模块的开发。
当前已具备 Qwen 真实 Embedding + Milvus Lite 端到端运行能力。

---

## ✅ Task 1：接入真实 LLM（已完成）

### 目标
将 `examples/real_world_demo.py` 中的 `MockLLM` 替换为真实的 Qwen 对话 API 调用。

### 完成内容
1. ✅ `src/llm_client.py` — 完整封装 Qwen 对话 API（DashScope SDK），支持：
   - 流式/非流式响应
   - 自动从 `.env` 读取 API Key
   - 支持 system/user/assistant 多轮对话
   - Token 计数和成本统计（支持 Qwen / OpenAI 定价表）
   - 自动重试（指数退避）
   - 兼容 OpenAI API（DashScope 兼容模式 / 纯 OpenAI）
   - 完整的异常层级（Authentication / RateLimit / ServiceError）
2. ✅ `examples/real_world_demo.py` — 集成 LLMClient，有 API Key 时自动使用真实 LLM，无 Key 时降级 MockLLM
3. ✅ `examples/llm_integration_demo.py` — 6 个功能演示模块：
   - 基本对话、流式对话、多轮对话
   - Token 计数与成本统计
   - 错误处理展示
   - Provider 切换说明

### 文件清单
| 文件 | 说明 |
|------|------|
| `src/llm_client.py` | LLM 调用客户端（新文件） |
| `examples/llm_integration_demo.py` | LLM 集成示例（新文件） |
| `examples/real_world_demo.py` | 更新为支持 LLMClient + 真实 Redis |

---

## ✅ Task 2：Redis 持久化（已完成）

### 目标
将测试用的 `fakeredis` 替换为真实 Redis 连接，实现数据持久化。

### 完成内容
1. ✅ `examples/real_world_demo.py` — 自动检测 Redis 连接：
   - 有 `REDIS_URL` 环境变量 → 使用真实 Redis
   - 无 Redis → 降级 fakeredis 内存模式
2. ✅ `examples/redis_persistence_demo.py` — 5 个功能演示：
   - 持久化存储与读取
   - 重启恢复（模拟应用重启后数据恢复）
   - TTL 过期管理
   - 批量操作性能对比
   - 连接池管理
3. ✅ 配置文件 `.env.example` 和 `redis.conf` 已就绪

### 文件清单
| 文件 | 说明 |
|------|------|
| `examples/redis_persistence_demo.py` | Redis 持久化示例（新文件） |
| `examples/real_world_demo.py` | 支持真实 Redis / fakeredis 自动切换 |

---

## ✅ Task 3：生产部署配置（已完成）

### 目标
将 Milvus Lite 替换为远程 Milvus 连接，适配生产环境部署。

### 完成内容
1. ✅ `MILVUS_URI=http://milvus-host:19530` 配置支持（已在 EmbeddingsCache 中实现）
2. ✅ `docker-compose.yml` — 包含 Redis + Milvus + RedisInsight 服务
3. ✅ `redis.conf` — 生产级 Redis 配置（AOF + RDB + LRU 淘汰）
4. ✅ `examples/production_deploy.py` — 4 个功能演示：
   - 远程服务连接与健康检查
   - 生产环境读写操作
   - 容错降级（Milvus 不可用时）
   - 重试机制说明
5. ✅ `docs/deployment.md` — 完整部署文档

### 文件清单
| 文件 | 说明 |
|------|------|
| `docker-compose.yml` | Docker 编排配置（新文件） |
| `redis.conf` | Redis 生产配置（新文件） |
| `examples/production_deploy.py` | 生产部署示例（新文件） |
| `docs/deployment.md` | 部署指南文档（新文件） |

---

## ✅ Task 4：阈值自动调优（已完成）

### 目标
实现缓存/路由阈值的自动化调优，替代手动调整。

### 完成内容
1. ✅ `src/AutoTuner.py` — 完整阈值调优模块：
   - 收集历史查询的相似度分布数据
   - 根据分布自动计算最优阈值（百分位法）
   - 支持按路由分别设置不同阈值
   - 渐进式调优（单次变化限制 ±0.1，防震荡）
   - 调优冷却机制（防频繁调整）
   - 相似度分布分析（直方图 + 百分位数）
   - 可选 Redis 持久化
2. ✅ `examples/auto_tuning_demo.py` — 5 个功能演示：
   - 基础自动调优
   - 相似度分布分析
   - 按路由分别调优
   - 多轮渐进式优化
   - 不同阈值的效果对比

### 关键指标
- 缓存命中率目标：30%~60% ✅
- 路由准确率目标：> 90%（需结合真实数据验证）

### 文件清单
| 文件 | 说明 |
|------|------|
| `src/AutoTuner.py` | 阈值自动调优器（新文件） |
| `examples/auto_tuning_demo.py` | 自动调优示例（新文件） |

---

## ✅ Task 5：Web 可视化监控面板（已完成）

### 目标
提供 Web 界面实时查看缓存命中率、路由统计、对话历史和系统状态。

### 完成内容
1. ✅ `src/monitoring.py` — FastAPI 监控服务：
   - `GET /api/stats` — 缓存/路由/会话汇总统计
   - `GET /api/analytics` — 趋势/分布分析
   - `GET /api/health` — 后端服务健康检查
   - `POST /api/cache/clear` — 清空缓存
   - `POST /api/threshold` — 更新相似度阈值
   - `GET /` — HTML 仪表盘（深色主题，10 秒自动刷新）
2. ✅ `examples/monitoring_demo.py` — 启动预置数据的监控面板
3. ✅ 可直接运行：`python -m src.monitoring`

### 依赖
- `pip install fastapi uvicorn`

### 文件清单
| 文件 | 说明 |
|------|------|
| `src/monitoring.py` | FastAPI 监控服务 + HTML 仪表盘（新文件） |
| `examples/monitoring_demo.py` | 监控面板演示（新文件） |

---

## ⏳ Task 6：多模态缓存支持（待实现）

### 目标
扩展缓存能力到图文多模态场景。

### 具体内容
1. 支持存储和检索图片 Embedding
2. 图文混合检索（CLIP 模型）
3. 多模态路由（文字类/图片类查询分流）
4. 新增 `examples/multimodal_demo.py`

### 依赖
- `pip install torch transformers`

### 优先级
P2 — 扩展能力，当前版本暂未实现。

---

## 完成状态总览

| 优先级 | Task | 状态 | 工时 | 文件数 |
|--------|------|------|------|--------|
| P0 | Task 1 - 接入真实 LLM | ✅ 已完成 | 2h | 2 新建 + 1 更新 |
| P0 | Task 2 - Redis 持久化 | ✅ 已完成 | 1h | 1 新建 + 1 更新 |
| P1 | Task 3 - 生产部署 | ✅ 已完成 | 2h | 4 新建 |
| P1 | Task 4 - 阈值调优 | ✅ 已完成 | 1.5h | 2 新建 |
| P2 | Task 5 - 监控面板 | ✅ 已完成 | 3h | 2 新建 |
| P2 | Task 6 - 多模态 | ⏳ 待实现 | 4h | — |

### 文件变更汇总

| 新文件 | 路径 |
|--------|------|
| `src/llm_client.py` | LLM 调用客户端 |
| `src/AutoTuner.py` | 阈值自动调优 |
| `src/monitoring.py` | Web 监控面板 |
| `examples/llm_integration_demo.py` | LLM 集成示例 |
| `examples/redis_persistence_demo.py` | Redis 持久化示例 |
| `examples/production_deploy.py` | 生产部署示例 |
| `examples/auto_tuning_demo.py` | 阈值调优示例 |
| `examples/monitoring_demo.py` | 监控面板示例 |
| `docker-compose.yml` | Docker 编排配置 |
| `redis.conf` | Redis 生产配置 |
| `docs/deployment.md` | 部署指南 |

| 更新文件 | 路径 |
|----------|------|
| `examples/real_world_demo.py` | 支持 LLMClient + 真实 Redis |
| `src/__init__.py` | 导出 LLMClient |
| `docs/spec.md` | 更新完成状态 |
| `requirements.txt` | 添加可选监控依赖 |
