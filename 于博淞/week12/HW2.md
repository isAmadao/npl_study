# 作业2：06-stock-bi-agent 代码分析

---

## 1. 什么是前后端分离？

前端（页面）和后端（服务）各自独立，通过 HTTP 接口通信。

本项目中：
- **后端**：FastAPI 提供接口，如 `/v1/chat`、`/v1/user/login`
- **前端**：只管调接口、展示结果，不关心后端逻辑

两者可以独立开发、独立部署。

---

## 2. 历史对话如何存储？如何传给大模型？

**存储**：每轮对话结束后，将用户消息和模型回答写入 SQLite 数据库（`chat_message` 表），字段包括 `role`（user/assistant）和 `content`（消息内容），通过 `session_id` 关联同一个会话。

**传给大模型**：下次提问时，用 `AdvancedSQLiteSession` 按 `session_id` 从数据库读出历史，自动拼成如下格式一起发给大模型：

```
[system]    你是AI助手...
[user]      第1轮问题
[assistant] 第1轮回答
[user]      第2轮问题（当前）
```

大模型拿到完整上下文，就能实现多轮对话。