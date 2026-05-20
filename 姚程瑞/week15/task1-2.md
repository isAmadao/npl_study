# 05-multimodal-rag-chatbot 需求文档

## 项目概述

构建一个基于 RAG (Retrieval-Augmented Generation) 的多模态聊天机器人，能够理解文档内容并进行智能问答。

---

## 功能需求

### 1. 文档处理模块

| 功能 | 描述 |
|------|------|
| PDF 解析 | 支持 pdfplumber 和 MinerU 两种解析方式 |
| Markdown 解析 | 支持解析 Markdown 文档 |
| 文档分块 | 按标题、按大小两种分块策略 |
| 图片提取 | 从文档中提取图片并保存 |

### 2. 向量存储模块

| 功能 | 描述 |
|------|------|
| 向量添加 | 支持批量添加文档分块向量 |
| 向量检索 | 基于余弦相似度的 Top-K 检索 |
| 持久化 | 可选持久化到磁盘 |

### 3. 检索模块

| 功能 | 描述 |
|------|------|
| 向量检索 | 基础语义检索 |
| 混合检索 | 支持关键词+向量的混合检索（可选） |
| Rerank | 对检索结果进行重排序（可选） |

### 4. 对话模块

| 功能 | 描述 |
|------|------|
| 会话管理 | 支持多轮对话，会话历史记录 |
| 流式输出 | 支持流式生成回复 |
| 提示词工程 | 可配置的提示词模板 |
| 引用标注 | 在回复中标注引用来源 |

### 5. API 接口

| 接口 | 方法 | 描述 |
|------|------|------|
| /documents | POST | 上传并处理文档 |
| /chat | POST | 发送聊天消息 |
| /chat/stream | GET | 流式聊天接口 |
| /retrieve | POST | 仅检索相关文档 |

---

## 非功能需求

1. **可扩展性**：模块化设计，支持替换不同的 Embedding 模型和 LLM
2. **性能**：检索响应时间 < 500ms
3. **兼容性**：支持 Python 3.8+
4. **可测试性**：提供完整的单元测试和集成测试

---

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| PDF 解析 | pdfplumber / MinerU |
| 向量存储 | NumPy (内存) / FAISS / ChromaDB |
| 嵌入模型 | OpenAI Embedding / 通义千问 Embedding |
| LLM | OpenAI API / 通义千问 API |
| Web 框架 (可选) | FastAPI / Flask |

---

## 数据模型

```
Document:
  - doc_id: str
  - path: str
  - chunks: List[DocumentChunk]
  - metadata: Dict

DocumentChunk:
  - chunk_id: str
  - content: str
  - embedding: ndarray (可选)
  - image_path: str (可选)
  - metadata: Dict

Conversation:
  - conv_id: str
  - messages: List[ChatMessage]

ChatMessage:
  - role: str (user/assistant)
  - content: str
  - timestamp: datetime
```
