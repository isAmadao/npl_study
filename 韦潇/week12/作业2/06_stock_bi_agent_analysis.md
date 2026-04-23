# Stock BI Agent 代码分析

## 1. 什么是前后端分离？

从代码可以看到这是一个典型的**前后端分离架构**：

### 后端（FastAPI服务）

```python
# main_server.py
app = FastAPI()
app.include_router(chat_routers)  # 聊天路由
app.mount("/stock", stock_app)    # 股票API
```

### 前端调用方式

```python
# routers/chat.py - API接口定义
@router.post("/")
async def chat(req: RequestForChat) -> StreamingResponse:
    # 流式返回给前端
    return StreamingResponse(content=chat_stream_generator(),
                           media_type="text/event-stream")
```

### 核心特点

| 特性 | 说明 |
|------|------|
| **API接口化** | 前后端通过JSON API通信 |
| **流式输出(SSE)** | 使用 `StreamingResponse` 实现打字机效果 |
| **独立部署** | 后端运行在 `:8000`，前端独立运行 |
| **工具调用** | MCP服务器提供工具，前端按需调用 |

---

## 2. 历史对话如何存储？

### 数据库存储（SQLite）

```python
# 使用两张表存储对话

class ChatSessionTable(Base):
    """会话表 - 存储会话元信息"""
    __tablename__ = 'chat_session'
    session_id: str          # 会话唯一ID
    title: str               # 对话标题（取用户第一个问题）
    user_id: int             # 关联用户

class ChatMessageTable(Base):
    """消息表 - 存储每条消息"""
    __tablename__ = 'chat_message'
    chat_id: int             # 关联会话
    role: str                # "user" / "assistant" / "system"
    content: Text            # 消息内容
```

### 存储函数

```python
# services/chat.py
def append_message2db(session_id: str, role: str, content: str):
    # 将消息追加到数据库
    message_recod = ChatMessageTable(
        chat_id=chat_id,
        role=role,
        content=content
    )
    session.add(message_recod)
```

---

## 3. 如何将历史对话作为大模型输入？

### 使用 AdvancedSQLiteSession

```python
# services/chat.py
from agents.extensions.memory import AdvancedSQLiteSession

# 创建session，自动关联历史
session = AdvancedSQLiteSession(
    session_id=session_id,                    # 与系统会话ID关联
    db_path="./assert/conversations.db",      # 存储路径
    create_tables=True
)

# 调用agent时传入session
result = Runner.run_streamed(
    agent,
    input=content,     # 当前用户输入
    session=session   # 携带历史记录的session
)
```

### 工作流程

```
用户发送消息
    ↓
append_message2db() 存储用户消息
    ↓
AdvancedSQLiteSession 自动加载该session_id的历史
    ↓
Runner.run_streamed() 将历史+当前输入一起发送给大模型
    ↓
大模型回复 → append_message2db() 存储助手消息
```

### 关键代码

```python
# 获取历史消息
def get_chat_sessions(session_id: str):
    chat_messages = session.query(ChatMessageTable)\
        .join(ChatSessionTable)\
        .filter(ChatSessionTable.session_id == session_id).all()
    # 返回该会话的所有历史消息
```

**核心原理**：`AdvancedSQLiteSession` 会自动从数据库读取同一 `session_id` 的历史对话，在调用大模型时作为上下文自动注入，无需手动拼接。

---

## 4. 核心文件结构

```
06-stock-bi-agent/
├── main_server.py          # FastAPI 主服务
├── routers/
│   └── chat.py            # 聊天API路由
├── services/
│   └── chat.py            # 聊天业务逻辑
├── models/
│   ├── orm.py             # 数据库模型定义
│   └── data_models.py     # 数据模型
├── agent/
│   ├── stock_agent.py     # 股票分析Agent
│   ├── csv_agent.py       # CSV处理Agent
│   ├── excel_agent.py     # Excel处理Agent
│   └── db_agent.py        # 数据库Agent
└── assert/
    ├── conversations.db   # 对话历史存储
    └── sever.db            # 系统数据库
```
