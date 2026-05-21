# Claude Code 项目说明

## 项目

多模态 RAG 知识库（`05-multimodal-rag-chatbot`）。README 描述完整业务；本仓库已实现 **FastAPI 接口 + Mock 测试 + Worker 骨架**。

## 约束

- 只在本目录修改代码
- 密钥必须用环境变量（见 `.env.example`），禁止提交真实 key
- 先跑 `pytest tests/ -q`（`MOCK_MODE=1`）
- 不要删除 `web_page_*.py` 根目录旧文件（兼容）；新 UI 在 `web/` 目录

## 目录

```
app/
  main.py           # FastAPI 入口
  api/              # upload, retrieve, chat, documents
  services/         # embedding, retrieval, generation, vector_store
  worker/           # parse_document Kafka 消费者
web/                # Streamlit 调 API
tests/              # pytest
```

## 常用命令

```bash
# 测试
set MOCK_MODE=1
pytest tests/ -q

# API
set MOCK_MODE=1
uvicorn app.main:app --reload --port 8000

# Worker（另开终端，需先上传触发队列；mock 下可直接跑）
set MOCK_MODE=1
python -m app.worker.parse_document

# 手动灌库（不跑 worker）
python scripts/seed_mock.py

# UI
set API_BASE=http://127.0.0.1:8000
streamlit run web/web_demo.py
```

## 待扩展（可选）

1. `POST /chat` 接入 Qwen-VL 多模态输入
2. `mode=hybrid` 融合 BGE + CLIP 检索分数
3. `scripts/evaluate.py` 对接 README 评价指标
4. 生产环境关闭 `MOCK_MODE`，配置 Milvus/Kafka/MinerU

## Step 提示（给 cc 续写）

- Step A: 跑通 pytest，修失败用例
- Step B: 集成真实 Milvus（`MILVUS_URI`/`TOKEN`）
- Step C: Kafka + MinerU 联调
- Step D: Qwen-VL 图文问答

每步结束列出变更文件与验证命令。
