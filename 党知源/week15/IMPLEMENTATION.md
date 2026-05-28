# 实现说明（第15周作业）

## 架构

```
Streamlit (web/)  --HTTP-->  FastAPI (app/main.py)
                                |
                    +-----------+-----------+
                    |           |           |
              upload/doc   retrieve/chat   documents
                    |           |
                 SQLite      vector_store (Milvus / MOCK)
                    |
                 Kafka (MOCK: 内存队列)
                    |
            worker/parse_document
                    |
              MinerU -> markdown -> chunk -> embed
```

## 与原版差异

| 原版 | 现版 |
|------|------|
| Streamlit 直连 DB/Kafka/Milvus | FastAPI 统一接口 |
| 逻辑散落在三个 py 文件 | `app/services` 分层 |
| 硬编码 API Key | `.env` / 环境变量 |
| 仅 Streamlit 内检索 | `POST /retrieve` 可单独调用 |

根目录 `web_page_*.py`、`offline_precess_worker.py` 保留作参考，请用 `web/` + `app/`。

## 快速开始

```powershell
cd 05-multimodal-rag-chatbot\05-multimodal-rag-chatbot
pip install -r requirements.txt
$env:MOCK_MODE="1"
pytest tests/ -q
python scripts/seed_mock.py
uvicorn app.main:app --reload
# 另一终端
$env:API_BASE="http://127.0.0.1:8000"
streamlit run web/web_demo.py
```

## Claude Code 必要文件

- `CLAUDE.md` — cc 项目规则
- `REQUIREMENTS.md` — 接口与验收
- `.env.example` — 环境变量模板
- `IMPLEMENTATION.md` — 本文件
- `tests/` — 测试驱动续写
