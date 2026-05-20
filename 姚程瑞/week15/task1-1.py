# 05-multimodal-rag-chatbot 测试文档

## 测试策略

采用单元测试 + 集成测试的方式，确保各个模块功能正常。

---

## 单元测试

### 1. 文档分块测试 (DocumentChunker)

| 测试用例 | 输入 | 期望输出 |
|----------|------|----------|
| test_chunk_by_header | Markdown 文本带标题 | 按标题分割的分块列表 |
| test_chunk_by_size | 长文本 | 按指定大小分割的分块列表 |
| test_chunk_empty_input | 空字符串 | 空列表 |

### 2. 向量存储测试 (SimpleVectorStore)

| 测试用例 | 输入 | 期望输出 |
|----------|------|----------|
| test_add_chunks | DocumentChunk 列表 | chunks 已添加到存储 |
| test_search | 查询向量 | Top-K 相关结果 |
| test_search_empty_store | 空存储查询 | 空结果 |

### 3. 嵌入模型测试 (MockEmbeddingModel)

| 测试用例 | 输入 | 期望输出 |
|----------|------|----------|
| test_embed_text | 文本字符串 | 维度正确的向量 |
| test_embed_texts | 文本列表 | 向量列表 |
| test_embed_image | 图片路径 | 维度正确的向量 |

### 4. 对话管理测试 (ChatManager)

| 测试用例 | 输入 | 期望输出 |
|----------|------|----------|
| test_create_conversation | 无 | 新的会话 ID |
| test_add_message | 会话ID+消息 | 消息已添加到会话 |
| test_build_prompt | 查询+检索结果 | 格式化的提示词 |

---

## 集成测试

### 1. 端到端聊天测试

| 测试步骤 | 操作 | 期望 |
|----------|------|------|
| 1 | 添加文档到知识库 | 文档处理成功 |
| 2 | 发送相关查询 | 检索到相关分块 |
| 3 | 获取回复 | 回复基于文档内容 |

### 2. 多轮对话测试

| 测试步骤 | 操作 | 期望 |
|----------|------|------|
| 1 | 发送第一个查询 | 正常回复 |
| 2 | 发送相关的第二个查询 | 理解上下文 |
| 3 | 验证历史记录 | 会话历史完整保存 |

---

## 测试运行

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定模块测试
python -m pytest tests/test_chunker.py

# 运行集成测试
python -m pytest tests/test_integration.py
```

---

## 测试数据

测试文档目录：`demo/documents/`
- `2309 vllm/2309 vllm.md` - VLLM 技术文档
- `2312 sglang/2312 sglang.md` - SGLang 技术文档
