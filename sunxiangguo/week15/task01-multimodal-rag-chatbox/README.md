# MultiModal RAG ChatBox - 多模态RAG聊天框

一个基于CLIP/Qwen-VL和多模态检索增强生成的智能问答系统。

## 功能特性

- **多模态文档解析**：PDF结构化解析，支持文本、表格、图像分离
- **跨模态检索**：文本查询召回相关图像/图表
- **智能问答**：整合多模态内容生成增强上下文，LLM回答
- **图文混排**：支持图像URL在答案中的展示
- **文档管理**：上传、查看、删除文档

## 技术栈

### 后端
- Python 3.10+
- FastAPI
- Uvicorn
- SQLAlchemy
- Milvus (向量数据库)
- MinIO (对象存储)
- Kafka (消息队列)
- CLIP / Qwen-VL
- MinerU (PDF解析)

### 前端
- React 18
- Vite
- Tailwind CSS
- Framer Motion
- Lucide React Icons
- Axios

## 快速开始

### 前置要求

- Python 3.10+
- Node.js 18+
- Docker (可选，用于Docker部署)

### 一键启动

#### Windows

```cmd
start.bat
```

#### Linux / Mac

```bash
chmod +x start.sh stop.sh
./start.sh
```

### 手动启动

#### 1. 后端启动

```bash
cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. 前端启动

```bash
cd frontend
npm install
npm run dev
```

### 访问地址

- 前端: http://localhost:3000
- 后端: http://localhost:8000
- API文档: http://localhost:8000/docs

## 项目结构

```
multimodal-rag-chatbox/
├── backend/              # 后端代码
│   ├── app/
│   │   ├── api/         # API路由
│   │   ├── core/        # 核心配置
│   │   ├── services/    # 业务服务
│   │   ├── clients/     # 外部客户端
│   │   ├── models/      # 数据模型
│   │   ├── repositories/ # 数据访问
│   │   ├── utils/       # 工具函数
│   │   └── workers/     # 异步任务
│   └── requirements.txt
├── frontend/            # 前端代码
│   ├── src/
│   │   ├── components/  # UI组件
│   │   ├── pages/       # 页面组件
│   │   └── services/    # API服务
│   └── package.json
├── docs/                # 项目文档
├── start.bat            # Windows启动脚本
├── start.sh             # Linux/Mac启动脚本
└── docker-compose.yml   # Docker配置
```

## 文档

详细文档请参考 [docs/](docs/) 目录：

- [产品需求文档](docs/01-prd.md)
- [技术架构文档](docs/02-tech-arch.md)
- [API接口设计](docs/03-api-design.md)
- [部署运维指南](docs/04-deployment.md)
- [项目结构说明](docs/05-project-structure.md)

## Docker部署

```bash
# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

## 开发指南

### 添加新的API路由

1. 在 `backend/app/api/v1/` 创建新的路由文件
2. 在 `backend/app/main.py` 中注册路由

### 添加新的组件

1. 在 `frontend/src/components/` 创建新组件
2. 在需要的地方导入使用

## 许可证

MIT License

---

### 项目背景

随着企业数字化转型的深入，核心知识资产不再局限于纯文本形式，大量专业信息以PDF报告、扫描件、财务图表、设计图稿等多模态形式存在。传统文本RAG或OCR技术存在诸多局限性：深度语义理解不足，无法精准把握图表数据趋势、图像对象关系及复杂PDF版面结构，导致知识丢失；跨模态检索能力欠缺，难以实现"用文本问题检索相关图像或图表"的高级需求；大模型适用性差，缺乏将多模态数据高效转化为LLM可用Context的处理管道。

### 项目目标

• 实现文档深度解析：构建PDF解析pipeline，不仅提取文本，还要借助ViT/CLIP等模型实现图表、图像的自动识别、理解和关键信息提取。
• 构建多模态向量库：利用CLIP或Qwen-VL等模型，实现文本和图像的统一向量化，搭建支持跨模态检索的向量数据库。
• 工程化落地多模态RAG：将多模态知识（文本+图像描述/结构化数据）整合为Context，输入LLM，实现图文混合的高级问答。
项目经历
• 多模态数据处理 Pipeline 构建：
    ◦ PDF结构化解析：采用本地部署的MinerU工具进行PDF文档解析，分离出纯文本、表格和图像三大要素。同时，引入Kafka传输待解析的文档，实现分布式处理，提升解析效率。
    ◦ 图像与图表理解：引入CLIP模型，对图像和图表进行特征抽取和描述生成，将视觉信息转化为文本形式的结构化知识，并将解析后的图像保存为本地URL路径，便于后续处理。
• 多模态统一表示与检索：
    ◦ 跨模态向量化：利用CLIP等多模态模型，将文本描述和图像内容映射到统一的向量空间，并存储于向量数据库。
    ◦ 多模态检索：实现"文本查询 -> 向量检索（图文）"的跨模态检索流程，提高在文档库中召回相关图表或图像的准确率。
• 多模态问答：修改切分逻辑，让图像和描述进行有效划分chunk，然后将检索到的原始文本、图像描述和图表结构化数据整合为增强上下文（Context），输入给LLM，实现如"分析Q4财报中的营收图表，总结关键结论"等高级问答，并在RAG问答中进行图文混排，提升问答效果和用户体验。
