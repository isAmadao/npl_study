# Stock BI Agent 项目分析

## 1. 前后端分离
本项目采用 FastAPI（后端）+ Streamlit（前端） 的分离架构，通过 HTTP API + SSE 流式通信。

后端负责业务逻辑、数据库操作和大模型集成， 前端专注界面展示和交互。

### 优化建议（DDD 架构）：
当前项目按技术分层，建议转向领域驱动设计。将核心业务划分为独立领域（如 ChatDomain、StockDomain、UserDomain），每个领域内聚实体、值对象和领域服务；引入应用层协调用例，基础设施层隔离数据库和外部 API。
这样能降低耦合，提升业务逻辑的可测试性和可维护性。

## 2. 历史对话存储与上下文传递
采用**双层存储策略**：
- **sever.db**（SQLite）：持久化存储会话元数据和消息记录，支持业务查询和管理
- **conversations.db**（AdvancedSQLiteSession）：Agent SDK 自动管理对话状态，在调用大模型时自动拼接历史上下文

工作流程：前端通过 session_id 加载历史 → 用户发送新消息 → 后端存储到 sever.db → AdvancedSQLiteSession 自动从 conversations.db 读取完整上下文传给大模型 → 流式返回并保存回复。

### 优化建议：
1. 上下文压缩：引入滑动窗口或摘要机制，仅向大模型传递最近 N 轮对话或生成历史摘要，减少 Token 消耗并避免超出上下文限制。
2. 存储升级：将 SQLite 迁移至 PostgreSQL 或 MySQL 以支持高并发；对消息内容建立全文索引，实现快速历史检索。
3. 缓存加速：使用 Redis 缓存高频访问的会话上下文，减少对数据库的 I/O 压力，提升响应速度。
