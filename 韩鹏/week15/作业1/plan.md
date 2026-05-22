# 图文知识库 MVP 开发计划

## 1. 业务闭环

目标是交付一个最小可用的图文知识库系统，让用户可以完成以下闭环：

1. 在 Streamlit 页面上传 `pdf/docx` 文件。
2. 系统将文件保存到 `uploads/`，并在 `db.db` 的 `file` 表记录元数据与处理状态。
3. 系统尝试把处理任务投递到 Kafka。
4. 异步 worker 消费任务，解析文档内容，抽取文本与可用图片资源。
5. 文本使用本地 `bge-small-zh-v1.5` 进行向量化，图片使用本地 `jina-clip-v2` 进行向量化。
6. 向量写入 Milvus 集合 `rag_data_new`，同时保留本地索引兜底，避免外部服务不可用时系统完全失效。
7. 在“图文对话”页面输入问题，系统先检索相关文本/图片上下文，再将上下文传给大模型生成答案。

## 2. MVP 范围

本次实现聚焦最小可运行版本：

- 前端：一个 Streamlit 应用，包含“文件管理”和“图文对话”两个菜单。
- 数据库：SQLite + SQLAlchemy，维护文件状态。
- 队列：优先接入 Kafka，Kafka 不可用时自动退化到本地队列目录。
- 文档解析：
  - `docx`：解析正文文本，并提取 `word/media` 中的图片。
  - `pdf`：解析文本内容。
- 向量存储：
  - 优先写入 Milvus。
  - 同时写入本地 JSON 索引，作为离线检索兜底。
- 问答：优先调用 DashScope 兼容接口；失败时返回检索摘要，保证页面可用。

## 3. 架构设计审核

### 3.1 模块划分

- `app.py`
  - Streamlit 入口。
- `worker.py`
  - 异步消费任务并执行入库。
- `src/rag_mvp/config.py`
  - 项目路径、模型路径、Kafka/Milvus/LLM 配置。
- `src/rag_mvp/db.py` + `src/rag_mvp/models.py`
  - SQLite 连接与 ORM 模型。
- `src/rag_mvp/services/queue_service.py`
  - Kafka 发布/消费与本地队列兜底。
- `src/rag_mvp/services/document_parser.py`
  - `pdf/docx` 文档解析。
- `src/rag_mvp/services/embedding_service.py`
  - 本地 embedding 模型加载与编码。
- `src/rag_mvp/services/vector_store.py`
  - Milvus 与本地向量索引。
- `src/rag_mvp/services/ingest_service.py`
  - 文档切分、向量化、写库。
- `src/rag_mvp/services/chat_service.py`
  - 检索增强问答。

### 3.2 关键设计决策

1. 统一把向量维度补齐到 1024 维：
   - BGE 文本向量为 512 维，右侧补零到 1024 维。
   - Jina CLIP 原生为 1024 维。
   - 这样 Milvus 只需维护一个 `embedding` 向量字段，简化集合设计。
2. Milvus 与本地索引双写：
   - 满足“向量库存储”的要求。
   - 同时避免外部 Milvus 网络波动时系统不可用。
3. Kafka 优先，本地队列兜底：
   - 满足“通知 Kafka”要求。
   - 同时让当前环境下的 MVP 可继续跑通。

## 4. 分步开发计划

1. 完成项目骨架、配置文件、计划文档。
2. 完成数据库、文件元数据模型和初始化逻辑。
3. 完成上传、删除、任务投递能力。
4. 完成文档解析、文本切分、向量化、向量存储。
5. 完成图文对话检索与大模型调用。
6. 完成 worker 脚本。
7. 编写单元测试并执行本地验证。

## 5. 验收标准

- 可以启动 Streamlit 页面。
- 可以上传 `pdf/docx` 到 `uploads/`。
- `db.db` 中存在 `file` 表且有状态字段。
- 有可运行的 `worker.py`。
- 有可运行的检索问答主流程。
- 有 `plan.md`。
- 有单元测试并能在当前环境完成基本验证。

