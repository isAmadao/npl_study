# 多模态 RAG 项目需求（作业 / Claude Code）

## 1. 业务目标

- 用户上传 PDF → 异步解析（MinerU）→ 文本/图 chunk 写入向量库
- 用户提问 → 检索相关 chunk → 生成带出处与图片链接的答案

## 2. HTTP 接口（已实现）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/upload/document` | multipart: `file`, `kb_id` |
| POST | `/retrieve` | JSON: `query`, `kb_id`, `top_k`, `mode` |
| POST | `/chat` | JSON: `query`, `kb_id`, `top_k` |
| GET | `/documents` | 列表，`?kb_id=default` |
| DELETE | `/documents/{doc_id}` | 删文件 + 向量 |
| GET | `/health` | 健康检查 |

## 3. Worker

- 模块：`app.worker.parse_document`
- 消费 Kafka `rag-data`；`MOCK_MODE=1` 时用 `doc1/content.md` 跳过 MinerU

## 4. 技术栈

- FastAPI + SQLite + Milvus + Kafka + Streamlit
- BGE / CLIP / Qwen（生产环境）

## 5. 测试

```bash
set MOCK_MODE=1
pytest tests/ -q
```

## 6. 评价维度（README 原文）

- 页面匹配度 0.25
- 文件名匹配度 0.25
- 答案 Jaccard 相似度 0.5

Phase 2 可补 `scripts/evaluate.py`。
