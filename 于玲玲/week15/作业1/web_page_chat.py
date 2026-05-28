#!/usr/bin/env python3
"""web_page_chat — 多模态问答服务（FastAPI）"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

import config
from models import get_db, _now, init_db
from utils.embedding import TextEmbedder, ImageEmbedder
from utils.milvus_client import MilvusStore


# ---------- 请求/响应模型 ----------
class CreateKBRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    description: str = ""


class ChatRequest(BaseModel):
    kb_id: int = Field(..., gt=0)
    question: str = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(default=config.DEFAULT_TOP_K, ge=1, le=20)


class SourceInfo(BaseModel):
    doc_id: int
    filename: str
    page_num: int
    chunk_type: str
    content_snippet: str = ""


class ChatResponse(BaseModel):
    code: int
    message: str
    data: Optional[dict] = None


# ---------- 全局 ----------
text_embedder: TextEmbedder | None = None
image_embedder: ImageEmbedder | None = None
milvus: MilvusStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global text_embedder, image_embedder, milvus
    init_db()
    text_embedder = TextEmbedder()
    image_embedder = ImageEmbedder()
    milvus = MilvusStore()
    milvus.ensure_collections()
    yield


app = FastAPI(
    title="多模态RAG - 问答服务",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------- 知识库管理 ----------
@app.post("/knowledge-base")
async def create_knowledge_base(req: CreateKBRequest):
    """创建知识库。"""
    now = _now()
    with get_db() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO knowledge_bases (name, description, created_at) VALUES (?, ?, ?)",
                (req.name, req.description, now),
            )
        except Exception:
            raise HTTPException(status_code=409, detail=f"知识库 '{req.name}' 已存在")
        kb_id = cur.lastrowid

    return {"code": 0, "message": "知识库创建成功", "data": {"kb_id": kb_id, "name": req.name}}


@app.get("/knowledge-bases")
async def list_knowledge_bases():
    """列出所有知识库。"""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, name, description, created_at FROM knowledge_bases ORDER BY id DESC"
        ).fetchall()

    return {
        "code": 0,
        "data": [
            {"id": r["id"], "name": r["name"], "description": r["description"], "created_at": r["created_at"]}
            for r in rows
        ],
    }


@app.get("/knowledge-base/{kb_id}/documents")
async def list_documents(kb_id: int):
    """列出知识库下的文档。"""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, filename, status, page_count, error_msg, created_at, updated_at "
            "FROM documents WHERE kb_id=? ORDER BY id DESC",
            (kb_id,),
        ).fetchall()

    return {
        "code": 0,
        "data": [
            {
                "doc_id": r["id"], "filename": r["filename"], "status": r["status"],
                "page_count": r["page_count"], "error_msg": r["error_msg"],
                "created_at": r["created_at"], "updated_at": r["updated_at"],
            }
            for r in rows
        ],
    }


# ---------- 多模态问答 ----------
@app.post("/chat")
async def chat(req: ChatRequest):
    """多模态问答：检索文本+图片 → 图文排版 → Qwen-VL 生成答案。"""
    # 1. 验证知识库
    with get_db() as conn:
        kb = conn.execute("SELECT id FROM knowledge_bases WHERE id=?", (req.kb_id,)).fetchone()
        if kb is None:
            raise HTTPException(status_code=404, detail=f"知识库 {req.kb_id} 不存在")

    # 2. 将用户问题编码为文本向量
    question_embedding = text_embedder.encode([req.question])[0]

    # 3. 检索文本 chunks
    text_results = milvus.search_text(
        kb_id=req.kb_id,
        query_embedding=question_embedding,
        top_k=req.top_k,
    )

    # 4. 将用户问题编码为 CLIP 向量，检索相关图片
    image_results = milvus.search_images(
        kb_id=req.kb_id,
        query_embedding=question_embedding,
        top_k=req.top_k,
    )

    # 5. 去重并组织检索结果
    seen_chunk_ids = set()
    text_items = []
    for r in text_results:
        cid = r.get("chunk_id", r.get("id"))
        if cid not in seen_chunk_ids:
            seen_chunk_ids.add(cid)
            text_items.append(r)

    image_items = []
    for r in image_results:
        cid = r.get("chunk_id", r.get("id"))
        if cid not in seen_chunk_ids:
            seen_chunk_ids.add(cid)
            image_items.append(r)

    # 如果没有检索到任何内容
    if not text_items and not image_items:
        return {
            "code": 0,
            "message": "知识库中未找到相关内容",
            "data": {"answer": "抱歉，在知识库中未找到与您问题相关的内容。", "sources": []},
        }

    # 6. 收集来源信息
    sources = _collect_sources(text_items, image_items)

    # 7. 构建图文上下文，调用 Qwen-VL 生成答案
    answer = _call_qwen_vl(req.question, text_items, image_items)

    return {
        "code": 0,
        "message": "success",
        "data": {
            "answer": answer,
            "sources": sources,
        },
    }


def _collect_sources(text_results: list, image_results: list) -> list[dict]:
    """从检索结果收集来源信息（文档名、页码等）。"""
    doc_ids = set()
    for r in text_results + image_results:
        doc_ids.add(r.get("doc_id", 0))

    if not doc_ids:
        return []

    with get_db() as conn:
        rows = conn.execute(
            f"SELECT id, filename FROM documents WHERE id IN ({','.join('?' * len(doc_ids))})",
            list(doc_ids),
        ).fetchall()
    doc_map = {r["id"]: r["filename"] for r in rows}

    sources = []
    for r in text_results:
        sources.append({
            "doc_id": r.get("doc_id"),
            "filename": doc_map.get(r.get("doc_id"), "未知"),
            "page_num": r.get("page_num", 0),
            "chunk_type": "text",
            "content_snippet": (r.get("content", "") or "")[:200],
        })
    for r in image_results:
        sources.append({
            "doc_id": r.get("doc_id"),
            "filename": doc_map.get(r.get("doc_id"), "未知"),
            "page_num": r.get("page_num", 0),
            "chunk_type": "image",
            "content_snippet": r.get("file_path", ""),
        })

    return sources


def _call_qwen_vl(question: str, text_items: list, image_items: list) -> str:
    """调用 Qwen-VL 进行多模态问答。"""

    # 构建文本上下文
    context_parts = []
    for i, item in enumerate(text_items):
        content = item.get("content", "") or ""
        page = item.get("page_num", 0)
        doc_id = item.get("doc_id", 0)
        context_parts.append(f"[文本片段{i+1}] (doc_id={doc_id}, 第{page}页)\n{content}")

    text_context = "\n\n".join(context_parts) if context_parts else "无相关文本"

    # 构建提示词
    system_prompt = (
        "你是一个多模态文档问答助手。请根据提供的文本内容和图片信息回答用户的问题。"
        "回答要求：1) 准确、简洁；2) 引用信息来源（文档ID和页码）；3) 如果信息不足，请如实说明。"
    )

    user_message = f"""用户问题：{question}

检索到的相关文本：
{text_context}

检索到的相关图片数量：{len(image_items)}

请综合以上文本和图片信息回答用户问题。回答时请注明信息来源。"""

    # 对于 Qwen-VL API (兼容 OpenAI 格式)，构建多模态消息
    api_url = f"{config.QWEN_VL_API_URL}/chat/completions"

    # 构建 content 数组：先文本，再图片
    content_array = [{"type": "text", "text": user_message}]

    # 添加图片 (最多 5 张，避免 context 过长)
    for img_item in image_items[:5]:
        img_path = img_item.get("file_path", "")
        if img_path and Path(img_path).exists():
            try:
                with open(img_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
                content_array.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                })
            except Exception:
                continue

    payload = {
        "model": config.QWEN_VL_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_array},
        ],
        "max_tokens": 1024,
        "temperature": 0.3,
    }

    try:
        resp = requests.post(api_url, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        answer = result["choices"][0]["message"]["content"]
        return answer
    except requests.exceptions.ConnectionError:
        # Qwen-VL 不可用时的降级回答
        return _fallback_answer(question, text_items, image_items)
    except Exception as e:
        return f"问答服务暂时不可用: {str(e)}"


def _fallback_answer(question: str, text_items: list, image_items: list) -> str:
    """当 Qwen-VL 不可用时，基于检索到的文本片段给出降级回答。"""
    if not text_items:
        return "无法连接到视觉问答模型，且未检索到相关文本内容，暂时无法回答您的问题。"

    snippets = []
    for item in text_items[:3]:
        content = (item.get("content", "") or "")[:300]
        page = item.get("page_num", 0)
        snippets.append(f"（第{page}页）{content}")

    return (
        f"注意：多模态问答模型暂不可用，以下是基于文本检索结果的参考信息：\n\n"
        + "\n\n".join(snippets)
        + f"\n\n共检索到 {len(text_items)} 个文本片段和 {len(image_items)} 张相关图片，"
        "完整的多模态推理需要 Qwen-VL 服务正常运行时才能提供。"
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "web_page_chat:app",
        host="0.0.0.0",
        port=config.CHAT_SERVICE_PORT,
    )