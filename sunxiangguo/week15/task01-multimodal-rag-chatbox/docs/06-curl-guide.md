# MultiModal RAG ChatBox - curl 使用指南

本文档介绍如何使用 `curl` 命令直接调用多模态RAG聊天框的API接口。

## 基础信息

- **API基础地址**: `http://localhost:8000/api/v1`
- **认证方式**: 无（Demo版本）
- **数据格式**: JSON

---

## 目录

1. [文档管理接口](#1-文档管理接口)
2. [检索接口](#2-检索接口)
3. [问答接口](#3-问答接口)
4. [图像服务接口](#4-图像服务接口)
5. [向量库管理接口](#5-向量库管理接口)
6. [模型服务接口](#6-模型服务接口)
7. [健康检查](#7-健康检查)

---

## 1. 文档管理接口

### 1.1 上传文档

**请求**:
```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@/path/to/your/document.pdf"
```

**响应示例**:
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "document.pdf",
    "status": "pending",
    "created_at": "2024-01-01T12:00:00"
  }
}
```

### 1.2 获取文档列表

**请求**:
```bash
curl -X GET "http://localhost:8000/api/v1/documents?page=1&size=10"
```

**响应示例**:
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "items": [
      {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "name": "Q4财报.pdf",
        "page_count": 24,
        "status": "completed",
        "created_at": "2024-01-01T12:00:00"
      }
    ],
    "total": 1,
    "page": 1,
    "size": 10
  }
}
```

### 1.3 获取文档详情

**请求**:
```bash
curl -X GET http://localhost:8000/api/v1/documents/<document-id>
```

**响应示例**:
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "Q4财报.pdf",
    "page_count": 24,
    "status": "completed",
    "file_path": "minio://documents/documents/xxx/Q4财报.pdf",
    "created_at": "2024-01-01T12:00:00",
    "updated_at": "2024-01-01T12:05:00"
  }
}
```

### 1.4 删除文档

**请求**:
```bash
curl -X DELETE http://localhost:8000/api/v1/documents/<document-id>
```

**响应示例**:
```json
{
  "code": 200,
  "message": "success",
  "data": null
}
```

---

## 2. 检索接口

### 2.1 跨模态检索

**请求**:
```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "分析Q4财报中的营收图表",
    "top_k": 5,
    "document_ids": ["doc-id-1", "doc-id-2"]
  }'
```

**参数说明**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| query | string | 是 | 查询文本 |
| top_k | int | 否 | 返回数量，默认5 |
| document_ids | array | 否 | 指定文档ID列表，为空则搜索全部 |

**响应示例**:
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "query": "分析Q4财报中的营收图表",
    "results": [
      {
        "id": "chunk-1",
        "document_id": "doc-1",
        "content_type": "image",
        "content": "Q4营收柱状图，显示各季度营收增长趋势...",
        "image_url": "/api/v1/images/image-1",
        "page_number": 5,
        "similarity": 0.85
      },
      {
        "id": "chunk-2",
        "document_id": "doc-1",
        "content_type": "text",
        "content": "Q4营收达到100亿，同比增长25%...",
        "page_number": 3,
        "similarity": 0.78
      }
    ]
  }
}
```

---

## 3. 问答接口

### 3.1 发起问答

**请求**:
```bash
curl -X POST http://localhost:8000/api/v1/qa/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "分析Q4财报中的营收图表，总结关键结论",
    "conversation_id": "conv-xxx",
    "document_ids": ["doc-id-1"]
  }'
```

**参数说明**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| question | string | 是 | 用户问题 |
| conversation_id | string | 否 | 对话ID，不传则新建 |
| document_ids | array | 否 | 指定文档ID列表 |

**响应示例**:
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "conversation_id": "conv-xxx",
    "answer": "根据Q4财报图表分析，营收达到100亿，同比增长25%...",
    "references": [
      {
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "content_type": "image",
        "image_url": "/api/v1/images/image-1"
      }
    ],
    "created_at": "2024-01-01T12:30:00"
  }
}
```

### 3.2 获取对话列表

**请求**:
```bash
curl -X GET "http://localhost:8000/api/v1/qa/conversations?page=1&size=10"
```

**响应示例**:
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "items": [
      {
        "id": "conv-xxx",
        "last_message": "分析Q4财报中的营收图表",
        "message_count": 5,
        "created_at": "2024-01-01T12:00:00",
        "updated_at": "2024-01-01T12:30:00"
      }
    ],
    "total": 1,
    "page": 1,
    "size": 10
  }
}
```

### 3.3 获取对话详情

**请求**:
```bash
curl -X GET http://localhost:8000/api/v1/qa/conversations/<conversation-id>
```

**响应示例**:
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "conversation_id": "conv-xxx",
    "messages": [
      {
        "id": "msg-1",
        "role": "user",
        "content": "分析Q4财报中的营收图表",
        "created_at": "2024-01-01T12:00:00"
      },
      {
        "id": "msg-2",
        "role": "assistant",
        "content": "根据Q4财报图表分析...",
        "references": [...],
        "created_at": "2024-01-01T12:00:05"
      }
    ]
  }
}
```

### 3.4 删除对话

**请求**:
```bash
curl -X DELETE http://localhost:8000/api/v1/qa/conversations/<conversation-id>
```

**响应示例**:
```json
{
  "code": 200,
  "message": "success",
  "data": null
}
```

---

## 4. 图像服务接口

### 4.1 获取图像

**请求**:
```bash
curl -X GET http://localhost:8000/api/v1/images/<image-id> -o image.png
```

**响应**: 返回二进制图像数据

### 4.2 获取图像缩略图

**请求**:
```bash
curl -X GET "http://localhost:8000/api/v1/images/<image-id>/thumbnail?width=100&height=100" -o thumbnail.png
```

**响应**: 返回二进制缩略图数据

---

## 5. 向量库管理接口

### 5.1 获取向量库状态

**请求**:
```bash
curl -X GET http://localhost:8000/api/v1/vector/db/status
```

**响应示例**:
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "collection_count": 5,
    "vector_count": 10000,
    "disk_usage": "500MB",
    "status": "healthy"
  }
}
```

### 5.2 重建索引

**请求**:
```bash
curl -X POST http://localhost:8000/api/v1/vector/db/rebuild \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "doc-id-1"
  }'
```

**响应示例**:
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "status": "started",
    "task_id": "rebuild-doc-id-1"
  }
}
```

---

## 6. 模型服务接口

### 6.1 获取模型状态

**请求**:
```bash
curl -X GET http://localhost:8000/api/v1/models/status
```

**响应示例**:
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "clip": {
      "status": "healthy",
      "version": "1.0",
      "latency_ms": 50
    },
    "qwen-vl": {
      "status": "healthy",
      "version": "1.0",
      "latency_ms": 200
    },
    "llm": {
      "status": "healthy",
      "version": "1.0",
      "latency_ms": 1000
    }
  }
}
```

---

## 7. 健康检查

**请求**:
```bash
curl -X GET http://localhost:8000/health
```

**响应示例**:
```json
{
  "status": "healthy"
}
```

---

## 实用技巧

### 使用jq处理JSON响应

```bash
# 获取文档列表并格式化
curl -s -X GET "http://localhost:8000/api/v1/documents" | jq .

# 只获取文档名称
curl -s -X GET "http://localhost:8000/api/v1/documents" | jq '.data.items[].name'
```

### 批量上传文档

```bash
# 批量上传目录中的所有PDF文件
for file in /path/to/pdfs/*.pdf; do
  echo "Uploading $file..."
  curl -X POST http://localhost:8000/api/v1/documents/upload -F "file=@$file"
done
```

---

## 常见错误码

| 错误码 | 含义 | 说明 |
|--------|------|------|
| 200 | 成功 | 请求成功 |
| 400 | 请求参数错误 | 参数校验失败 |
| 404 | 资源不存在 | 请求的资源不存在 |
| 500 | 服务器内部错误 | 服务端异常 |

---

## 完整流程示例

```bash
# 1. 上传文档
curl -X POST http://localhost:8000/api/v1/documents/upload -F "file=@Q4财报.pdf"

# 2. 查看文档列表
curl -X GET http://localhost:8000/api/v1/documents | jq .

# 3. 发起检索
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Q4营收", "top_k": 3}' | jq .

# 4. 发起问答
curl -X POST http://localhost:8000/api/v1/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "分析Q4财报中的营收情况"}' | jq .
```