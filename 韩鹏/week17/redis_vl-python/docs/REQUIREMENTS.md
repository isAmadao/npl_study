# 开发要求文档

## 环境要求

### 操作系统
- Windows 11 本地开发

### Python 环境
- 使用 conda 管理，环境名称为 `py310`
- Python 版本 3.10+
- 所有 `pip install` 不加 `--break-system-packages`（conda 环境不需要）

### Redis
- 通过 Docker Desktop 运行 Redis Stack 容器
- 端口：6379
- 镜像：`redis/redis-stack:latest` 或 `redis/redis-stack-server:latest`
- 使用 docker-compose 管理（可选，直接 docker run 也可）

### HuggingFace 模型
- 优先使用 `C:\Users\peng\.cache\huggingface\hub` 下已缓存的模型
- 禁止在运行时自动下载新模型（避免网络问题）
- 启动时列出可用模型，让用户选择或使用默认

### 其他中间件
- 所有中间件必须通过 Docker 容器运行
- 不在 Windows 本地直接安装中间件

### Git
- 本地已安装 git，可以使用
- **禁止自动 commit 和 push**，必须用户手动操作

---

## 开发流程要求

### 版本迭代模式
1. 整个项目划分为多个版本（MVP 最小版本数原则）
2. 每个版本是一个独立可运行、可测试的功能增量
3. 用户发送 **"开发下一个版本"** 命令触发下一版本开发
4. AI 根据 `docs/ROADMAP.md` 和 `docs/versions/` 下的版本规划自动确定要开发的内容

### 版本开发流程
1. AI 查看 `docs/ROADMAP.md` 确定当前应开发的版本号
2. AI 查看对应版本规划文档 `docs/versions/Vx.x.x.md`
3. AI 按照版本文档中的任务清单逐项开发
4. 开发完成后自动运行测试
5. 更新 `docs/ROADMAP.md` 中的进度状态
6. 如测试失败，修复后重新测试，直到全部通过

### 代码管理
- 所有代码放在项目根目录下，按模块分目录
- 禁止自动 `git commit` 和 `git push`
- 代码风格：简洁优先，不追求过度抽象

---

## 代码规范

### 项目结构（推荐）
```
redis_vl-python/
├── docs/                    # 开发文档
│   ├── GOAL.md              # 最终目标
│   ├── ROADMAP.md           # 开发进度
│   ├── REQUIREMENTS.md      # 本文件
│   └── versions/            # 版本文档
├── app/                     # 后端应用
│   ├── __init__.py
│   ├── main.py              # FastAPI 入口
│   ├── config.py            # 配置
│   ├── modules/             # 四个功能模块
│   │   ├── __init__.py
│   │   ├── embeddings_cache.py
│   │   ├── semantic_cache.py
│   │   ├── message_history.py
│   │   └── semantic_router.py
│   ├── redis_client.py      # Redis 连接管理
│   ├── vectorizer.py        # 向量化封装
│   └── templates/           # 前端页面
│       └── index.html
├── tests/                   # 测试
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_embeddings_cache.py
│   ├── test_semantic_cache.py
│   ├── test_message_history.py
│   ├── test_semantic_router.py
│   └── test_integration.py
├── docker-compose.yml       # Redis 容器
├── requirements.txt         # Python 依赖
└── run.py                   # 启动脚本
```

### 编码原则
- 每个模块核心逻辑控制在 200 行以内
- 函数单一职责，命名清晰
- 配置与代码分离
- 错误处理：Redis 连接失败时返回明确的错误信息，不崩溃
- 不引入不必要的抽象层和设计模式

### 测试要求
- 使用 pytest + httpx 进行测试
- 每个模块必须有对应的测试文件
- 每个版本开发完成后，运行全量测试
- 测试不依赖外部服务（除了本地 Redis 容器）
- 使用 pytest fixtures 管理 Redis 连接

---

## 前端要求

### 页面风格
- 简洁实用，不花哨
- 单页面，所有结果在同一页展示
- 使用现代浏览器原生能力（fetch API、CSS Grid/Flexbox）

### 交互体验
- 输入问题后点击发送或按 Enter 键提交
- 发送后显示加载状态
- 四个模块的结果以卡片形式展示
- 卡片用颜色区分命中/未命中状态
- 提供 Session ID 输入框，方便切换会话

### 前端技术
- 原生 HTML + CSS + JavaScript（不使用 React/Vue 等框架）
- CSS 使用 Flexbox/Grid 布局
- 内联到单个 HTML 文件中（FastAPI 模板渲染或静态文件）

---

## 配置文件

### .env 示例
```
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
HF_CACHE_DIR=C:\\Users\\peng\\.cache\\huggingface\\hub
HF_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIM=384
```

### docker-compose.yml
```yaml
version: '3.8'
services:
  redis:
    image: redis/redis-stack:latest
    container_name: redis-stack
    ports:
      - "6379:6379"
      - "8001:8001"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
```

---

## 开发约束总结

| 约束项 | 要求 |
|--------|------|
| Python 环境 | conda py310 |
| Redis | Docker 容器 |
| 模型加载 | 优先本地缓存 |
| 其他中间件 | Docker 容器 |
| Git | 手动操作，禁止自动 commit/push |
| 版本开发 | 用户命令触发，AI 自动查找任务 |
| 测试 | 每版本开发完自动运行 |
| 前端 | 原生 HTML，单页面 |
| 模块复杂度 | 每个核心模块 ≤200 行 |
