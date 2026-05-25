# AI狼人杀 — 技术架构

## 架构总览

```
┌──────────────────────────┐      WebSocket       ┌──────────────────────────┐
│      Vue 3 Frontend      │◄────────────────────►│     FastAPI Server        │
│                          │                      │                           │
│  ┌────────────────────┐  │                      │  ┌─────────────────────┐  │
│  │   GameBoard.vue    │  │                      │  │  /ws  WebSocket端点  │  │
│  │   玩家状态面板      │  │                      │  │  POST /games 创建游戏 │  │
│  └────────────────────┘  │                      │  │  GET /games/:id 状态  │  │
│  ┌────────────────────┐  │                      │  └─────────────────────┘  │
│  │   ChatLog.vue      │  │                      └────────────┬─────────────┘
│  │   实时发言/行动日志 │  │                                   │
│  └────────────────────┘  │                      ┌────────────┴─────────────┐
│  ┌────────────────────┐  │                      │      Game Engine         │
│  │   GameControl.vue  │  │                      │                           │
│  │   新游戏/速度控制   │  │                      │  StateMachine ─ 回合驱动  │
│  └────────────────────┘  │                      │  RuleEngine  ─ 胜负裁决  │
│                          │                      │  Role/Action ─ 角色行动   │
└──────────────────────────┘                      └────────────┬─────────────┘
                                                               │
                                              ┌────────────────┴─────────────┐
                                              │       Agent System           │
                                              │                              │
                                              │  InfoFilter ─ 信息可见性控制  │
                                              │  LLMClient  ─ DeepSeek API   │
                                              │  Prompts    ─ 角色Prompt模板  │
                                              └────────────────┬─────────────┘
                                                               │
                                                               ▼
                                              DeepSeek v4-flash API
                                              (api.deepseek.com/v1)
```

## 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| 前端框架 | Vue 3 + Vite + TypeScript | 观战UI |
| UI库 | Naive UI | 简洁组件 |
| 后端框架 | FastAPI | HTTP + WebSocket |
| 数据建模 | Pydantic v2 | 类型安全 |
| LLM客户端 | openai Python SDK | 调用DeepSeek API |
| 异步 | asyncio + uvicorn | 并发处理 |
| 测试 | pytest + pytest-asyncio | 自动化测试 |

## 项目结构

```
Werewolf-Game/
├── docs/                        # 开发文档
│   ├── 01-requirements.md       # 开发目标
│   ├── 02-architecture.md       # 技术架构（本文件）
│   ├── 03-development-plan.md   # 开发计划
│   ├── 04-progress.md           # 开发进度
│   └── 05-environment.md        # 环境配置
├── backend/                     # 后端项目
│   ├── werewolf/
│   │   ├── __init__.py
│   │   ├── engine.py            # 游戏引擎（状态机+规则+角色+行动）
│   │   ├── agent.py             # Agent系统（LLM客户端+信息过滤+Prompt）
│   │   ├── server.py            # FastAPI服务（HTTP+WebSocket）
│   │   └── main.py              # 终端入口
│   ├── tests/
│   │   └── test_engine.py       # 引擎单元测试
│   └── requirements.txt
├── frontend/                    # 前端项目
│   └── src/
│       ├── App.vue
│       ├── main.ts
│       └── components/
│           ├── GameBoard.vue    # 玩家状态面板
│           ├── ChatLog.vue      # 发言/行动日志
│           └── GameControl.vue  # 游戏控制
└── README.md
```

## 核心设计决策

### 1. 信息隔离机制

每个Agent在任何时刻只能访问：
- **公共信息**：白天发言记录、投票结果、死亡公告
- **私有信息**：自己的身份、夜晚行动结果
- **禁止访问**：其他角色的私有信息、其他Agent的内部推理

实现方式：`InfoFilter` 根据当前阶段和角色，从完整游戏状态中提取该Agent可见的子集。

### 2. 游戏状态机

```
[游戏开始] → [夜晚-狼人行动] → [夜晚-预言家] → [夜晚-女巫]
    → [天亮-死亡公告] → [白天-自由发言] → [投票阶段] → [天黑]
    → 循环直到胜负判定
```

### 3. Agent决策流程

```
收到可见信息 → 构造角色Prompt → 调用DeepSeek API → 解析结构化输出 → 执行行动
```

LLM输出格式（JSON）：
```json
{
  "reasoning": "我的推理过程...",
  "action": "kill|check|save|poison|vote|speak",
  "target": "player_id或null",
  "speech": "发言内容（仅speak行动时）"
}
```

### 4. 前端与后端通信

- WebSocket 单通道：后端推送游戏事件流
- 事件类型：`game_start`, `night_action`, `day_announcement`, `speech`, `vote`, `game_end`
- 前端不发送指令（纯观战模式MVP），仅接收和展示
