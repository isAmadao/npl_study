## 多模态RAG聊天框项目 - API接口设计文档

### 1. 接口规范

#### 1.1 基础信息

| 项目 | 值 |
| :--- | :--- |
| 协议 | HTTP/HTTPS |
| 主机 | localhost:8000 |
| 基础路径 | /api/v1 |
| 认证方式 | 无（Demo版本） |
| 数据格式 | JSON |

#### 1.2 通用响应格式

**成功响应**：
```json
{
    "code": 200,
    "message": "success",
    "data": {}
}
```

**失败响应**：
```json
{
    "code": 400,
    "message": "error message",
    "data": null
}
```

#### 1.3 错误码定义

| 错误码 | 含义 | 说明 |
| :--- | :--- | :--- |
| 200 | 成功 | 请求成功 |
| 400 | 请求参数错误 | 参数校验失败 |
| 404 | 资源不存在 | 请求的资源不存在 |
| 500 | 服务器内部错误 | 服务端异常 |

---

### 2. 文档管理接口

#### 2.1 上传文档

| 属性 | 值 |
| :--- | :--- |
| 路径 | `/documents/upload` |
| 方法 | POST |
| Content-Type | multipart/form-data |

**请求参数**：

| 参数名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| file | File | 是 | PDF文件 |

**成功响应**：
```json
{
    "code": 200,
    "message": "success",
    "data": {
        "id": "uuid-string",
        "name": "document.pdf",
        "status": "pending",
        "created_at": "2024-01-01T00:00:00"
    }
}
```

#### 2.2 获取文档列表

| 属性 | 值 |
| :--- | :--- |
| 路径 | `/documents` |
| 方法 | GET |

**请求参数**：

| 参数名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| page | int | 否 | 页码，默认1 |
| size | int | 否 | 每页数量，默认10 |
| status | string | 否 | 筛选状态 |

**成功响应**：
```json
{
    "code": 200,
    "message": "success",
    "data": {
        "items": [
            {
                "id": "uuid-string",
                "name": "document.pdf",
                "page_count": 10,
                "status": "completed",
                "created_at": "2024-01-01T00:00:00"
            }
        ],
        "total": 100,
        "page": 1,
        "size": 10
    }
}
```

#### 2.3 获取文档详情

| 属性 | 值 |
| :--- | :--- |
| 路径 | `/documents/{document_id}` |
| 方法 | GET |

**路径参数**：

| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| document_id | string | 文档ID |

**成功响应**：
```json
{
    "code": 200,
    "message": "success",
    "data": {
        "id": "uuid-string",
        "name": "document.pdf",
        "page_count": 10,
        "status": "completed",
        "file_path": "minio://bucket/document.pdf",
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:05:00"
    }
}
```

#### 2.4 删除文档

| 属性 | 值 |
| :--- | :--- |
| 路径 | `/documents/{document_id}` |
| 方法 | DELETE |

**路径参数**：

| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| document_id | string | 文档ID |

**成功响应**：
```json
{
    "code": 200,
    "message": "success",
    "data": null
}
```

---

### 3. 检索接口

#### 3.1 跨模态检索

| 属性 | 值 |
| :--- | :--- |
| 路径 | `/search` |
| 方法 | POST |

**请求体**：
```json
{
    "query": "分析Q4财报中的营收图表",
    "top_k": 5,
    "document_ids": ["doc1", "doc2"]
}
```

| 参数名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| query | string | 是 | 查询文本 |
| top_k | int | 否 | 返回数量，默认5 |
| document_ids | list | 否 | 指定文档ID列表，为空则搜索全部 |

**成功响应**：
```json
{
    "code": 200,
    "message": "success",
    "data": {
        "query": "分析Q4财报中的营收图表",
        "results": [
            {
                "id": "chunk-uuid",
                "document_id": "doc-uuid",
                "content_type": "image",
                "content": "Q4营收柱状图，显示各季度营收增长趋势...",
                "image_url": "/api/v1/images/image-uuid",
                "page_number": 5,
                "similarity": 0.85
            },
            {
                "id": "chunk-uuid",
                "document_id": "doc-uuid",
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

### 4. 问答接口

#### 4.1 发起问答

| 属性 | 值 |
| :--- | :--- |
| 路径 | `/qa/ask` |
| 方法 | POST |

**请求体**：
```json
{
    "question": "分析Q4财报中的营收图表，总结关键结论",
    "conversation_id": "conv-uuid",
    "document_ids": ["doc1", "doc2"]
}
```

| 参数名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| question | string | 是 | 用户问题 |
| conversation_id | string | 否 | 对话ID，不传则新建 |
| document_ids | list | 否 | 指定文档ID列表 |

**成功响应**：
```json
{
    "code": 200,
    "message": "success",
    "data": {
        "conversation_id": "conv-uuid",
        "answer": "根据Q4财报图表分析，营收达到100亿，同比增长25%...",
        "references": [
            {
                "chunk_id": "chunk-uuid",
                "document_id": "doc-uuid",
                "content_type": "image",
                "image_url": "/api/v1/images/image-uuid"
            }
        ],
        "created_at": "2024-01-01T00:00:00"
    }
}
```

#### 4.2 获取对话历史

| 属性 | 值 |
| :--- | :--- |
| 路径 | `/qa/conversations/{conversation_id}` |
| 方法 | GET |

**路径参数**：

| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| conversation_id | string | 对话ID |

**成功响应**：
```json
{
    "code": 200,
    "message": "success",
    "data": {
        "conversation_id": "conv-uuid",
        "messages": [
            {
                "id": "msg-uuid",
                "role": "user",
                "content": "分析Q4财报中的营收图表",
                "created_at": "2024-01-01T00:00:00"
            },
            {
                "id": "msg-uuid",
                "role": "assistant",
                "content": "根据Q4财报图表分析...",
                "references": [...],
                "created_at": "2024-01-01T00:00:05"
            }
        ]
    }
}
```

#### 4.3 获取对话列表

| 属性 | 值 |
| :--- | :--- |
| 路径 | `/qa/conversations` |
| 方法 | GET |

**请求参数**：

| 参数名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| page | int | 否 | 页码，默认1 |
| size | int | 否 | 每页数量，默认10 |

**成功响应**：
```json
{
    "code": 200,
    "message": "success",
    "data": {
        "items": [
            {
                "id": "conv-uuid",
                "last_message": "分析Q4财报中的营收图表",
                "message_count": 5,
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:05:00"
            }
        ],
        "total": 10,
        "page": 1,
        "size": 10
    }
}
```

#### 4.4 删除对话

| 属性 | 值 |
| :--- | :--- |
| 路径 | `/qa/conversations/{conversation_id}` |
| 方法 | DELETE |

**路径参数**：

| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| conversation_id | string | 对话ID |

**成功响应**：
```json
{
    "code": 200,
    "message": "success",
    "data": null
}
```

---

### 5. 图像服务接口

#### 5.1 获取图像

| 属性 | 值 |
| :--- | :--- |
| 路径 | `/images/{image_id}` |
| 方法 | GET |

**路径参数**：

| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| image_id | string | 图像ID |

**成功响应**：
- Content-Type: image/png | image/jpeg
- Body: 二进制图像数据

#### 5.2 获取图像缩略图

| 属性 | 值 |
| :--- | :--- |
| 路径 | `/images/{image_id}/thumbnail` |
| 方法 | GET |

**路径参数**：

| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| image_id | string | 图像ID |

**查询参数**：

| 参数名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| width | int | 否 | 宽度，默认100 |
| height | int | 否 | 高度，默认100 |

**成功响应**：
- Content-Type: image/png | image/jpeg
- Body: 二进制缩略图数据

---

### 6. 向量库管理接口

#### 6.1 获取向量库状态

| 属性 | 值 |
| :--- | :--- |
| 路径 | `/vector/db/status` |
| 方法 | GET |

**成功响应**：
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

#### 6.2 重建索引

| 属性 | 值 |
| :--- | :--- |
| 路径 | `/vector/db/rebuild` |
| 方法 | POST |

**请求体**：
```json
{
    "document_id": "doc-uuid"
}
```

| 参数名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| document_id | string | 否 | 文档ID，不传则重建全部 |

**成功响应**：
```json
{
    "code": 200,
    "message": "success",
    "data": {
        "status": "started",
        "task_id": "task-uuid"
    }
}
```

---

### 7. 模型服务接口

#### 7.1 获取模型状态

| 属性 | 值 |
| :--- | :--- |
| 路径 | `/models/status` |
| 方法 | GET |

**成功响应**：
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

### 8. 健康检查接口

#### 8.1 健康检查

| 属性 | 值 |
| :--- | :--- |
| 路径 | `/health` |
| 方法 | GET |

**成功响应**：
```json
{
    "status": "healthy"
}
```

---

### 9. API使用示例

#### 9.1 使用curl上传文档

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@document.pdf"
```

#### 9.2 使用curl发起问答

```bash
curl -X POST "http://localhost:8000/api/v1/qa/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "分析Q4财报中的营收图表",
    "document_ids": ["doc-uuid"]
  }'
```

#### 9.3 使用Python SDK

```python
import requests

class MultimodalRAGClient:
    def __init__(self, base_url):
        self.base_url = base_url
    
    def upload_document(self, file_path):
        with open(file_path, "rb") as f:
            response = requests.post(
                f"{self.base_url}/documents/upload",
                files={"file": f}
            )
        return response.json()
    
    def ask_question(self, question, document_ids=None):
        data = {"question": question}
        if document_ids:
            data["document_ids"] = document_ids
        response = requests.post(
            f"{self.base_url}/qa/ask",
            json=data
        )
        return response.json()
```