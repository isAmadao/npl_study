# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## 仓库概述

这是一个课程作业提交仓库，采用按学生姓名和周次组织的结构：

```
于玲玲/
├── week01/ ~ week15/
│   ├── 作业1.py / 作业1.ipynb / 作业1.docx
│   └── 作业2.py / 作业2.ipynb / 作业2.docx
```

- 根目录: `/Users/yull/PycharmProjects/hub-eJLq/`
- 远程仓库: `https://github.com/yull-100/hub-eJLq`
- 当前分支: `main`

## 当前项目 (week15/作业1)：多模态RAG问答系统

### 项目背景

构建一个多模态检索增强生成（RAG）系统，能够从图文混排的PDF知识库中检索相关信息并回答需要结合图像和文本推理的问题。系统需要理解自然语言问题、检索图像和文本、关联推理，并生成可溯源到具体文档页面的答案。

### 项目任务

- **多模态信息理解**：同时理解用户问题和知识库中的图像/文本
- **跨模态检索**：从多个PDF文档中高效检索相关图像、图表、文本段落
- **图文关联推理**：关联和融合图像信息与文本信息进行深层逻辑推理
- **答案生成**：生成准确简洁的答案，并指明信息来源（PDF文件、页码、图表）

### 技术栈

- **编程语言**：Python
- **服务框架**：FastAPI
- **模型**：
  - Qwen-VL：多模态问答，图像内容理解和生成
  - CLIP：多模态检索，文本到图像/文本的跨模态检索
  - DeepSeek-OCR / MinerU：PDF文档解析，输出markdown和图片
  - BGE：常规文本编码（embedding）
- **中间件**：
  - SQLite / MySQL：元信息存储
  - Milvus / ES：向量存储与检索
  - Kafka：分布式消息队列，用于上传和离线解析的解耦

### 架构设计

三个核心服务：

1. **web_page_upload**（生产者，10并发）：文件上传，文件存本地，将待解析任务发送到Kafka
2. **offline_process_worker**（消费者，离线处理）：从Kafka消费，调用MinerU解析，chunk切分+编码，存入Milvus
3. **web_page_chat**：用户提问，触发RAG流程，检索文本+图片，调用Qwen-VL生成答案

### 接口定义

- `POST /upload/document` — 上传文档到指定知识库，存储PDF并将解析任务入队
- `POST /chat` — 多模态问答：获取提问+知识库ID → embedding检索（文本+图）→ 图文排版 → 答案生成

### 功能模块

- **PDF内容解析**：使用DeepSeek-OCR或MinerU将PDF解析为markdown和图片文件
- **内容存储**：解析结果（markdown/图片）存本地或云存储（OSS）
- **内容检索**：CLIP模型进行文本和图像的跨模态检索
- **内容问答**：Qwen-VL模型基于检索到的文本和图像进行多模态推理和答案生成
- **知识库管理**：文档上传、chunk文本向量化和图片向量化
- **权限管理**：知识库级别的访问控制

### 评价指标

- 页面匹配度（0.25分）
- 文件名匹配度（0.25分）
- 答案内容相似度（0.5分，Jaccard相似系数）

## 项目领域

根据历史作业内容，涉及方向包括：
- **NLP**: BERT微调、文本分类、情感对话数据集处理 (week02~week04)
- **图像处理**: 图像识别、PDF解析 (week10~week11)
- **数据分析**: 数据可视化 (week14)、SQLite数据库查询 (week12)
- **DevOps**: YAML配置文件编写 (week09)
- **多模态RAG**: PDF文档解析、CLIP检索、Qwen-VL问答 (week15)

## 常用命令

```bash
# 查看当前工作目录状态
git status

# 添加作业文件（使用具体路径，避免 git add -A）
git add 于玲玲/week15/作业1/xxx.py

# 提交
git commit -m "于玲玲第十五周作业提交"

# 推送
git push
```

## 注意事项

- 不要修改其他同学和老师的文件
- 文件名不要包含空格或 `.` 等特殊符号（可用 `-`、`_` 替代）
- 提交后等待老师审批，不要自行修改已提交的历史作业文件
- MinerU解析一个文件约需1分钟且需要GPU，不能实时处理，须设计为离线任务
- HTTP接口调用需在30秒内返回结果，请求并发和处理并发能力要对等