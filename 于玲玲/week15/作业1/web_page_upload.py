#!/usr/bin/env python3
"""web_page_upload — 文件上传服务（FastAPI + Kafka 生产者）"""
from __future__ import annotations

import uuid
import shutil
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
import uvicorn
import json

import config
from models import get_db, _now
from utils.storage import ensure_kb_dirs, save_uploaded_pdf


# ---------- Kafka producer ----------
producer: KafkaProducer | None = None


def get_producer() -> KafkaProducer:
    global producer
    if producer is None:
        try:
            producer = KafkaProducer(
                bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                max_block_ms=5000,
                acks=1,
            )
        except NoBrokersAvailable:
            raise HTTPException(status_code=503, detail="消息队列不可用，请稍后重试")
    return producer


# ---------- App ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    from models import init_db
    init_db()
    yield
    if producer:
        producer.flush()
        producer.close()


app = FastAPI(
    title="多模态RAG - 文档上传服务",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------- 上传 ----------
@app.post("/upload/document")
async def upload_document(
    file: UploadFile = File(...),
    kb_id: int = Form(...),
):
    """上传 PDF 文档到指定知识库，存储文件并将解析任务入队。"""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    with get_db() as conn:
        kb = conn.execute(
            "SELECT id FROM knowledge_bases WHERE id = ?", (kb_id,)
        ).fetchone()
        if kb is None:
            raise HTTPException(status_code=404, detail=f"知识库 {kb_id} 不存在")

    # 保存文件到本地
    kb_dirs = ensure_kb_dirs(kb_id)
    filepath = save_uploaded_pdf(kb_id, file)

    # 写入元数据
    now = _now()
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO documents (kb_id, filename, filepath, status, created_at, updated_at)
               VALUES (?, ?, ?, 'pending', ?, ?)""",
            (kb_id, file.filename, str(filepath), now, now),
        )
        doc_id = cur.lastrowid

    # 发送到 Kafka 解析队列
    message = {
        "doc_id": doc_id,
        "kb_id": kb_id,
        "filename": file.filename,
        "filepath": str(filepath),
    }
    try:
        kp = get_producer()
        kp.send(config.KAFKA_TOPIC_DOCUMENT_PARSE, message)
    except Exception as e:
        with get_db() as conn:
            conn.execute(
                "UPDATE documents SET status='failed', error_msg=?, updated_at=? WHERE id=?",
                (str(e), _now(), doc_id),
            )
        raise HTTPException(status_code=503, detail=f"消息入队失败: {e}")

    return {
        "code": 0,
        "message": "上传成功，文档已加入解析队列",
        "data": {
            "doc_id": doc_id,
            "kb_id": kb_id,
            "filename": file.filename,
            "status": "pending",
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "web_page_upload:app",
        host="0.0.0.0",
        port=config.UPLOAD_SERVICE_PORT,
        workers=config.UPLOAD_WORKERS,
    )