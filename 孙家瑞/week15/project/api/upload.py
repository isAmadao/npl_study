"""POST /upload/document —— 上传 PDF 并触发后台解析。"""

import os
import uuid

from fastapi import APIRouter, BackgroundTasks, UploadFile

from ..core.config import UPLOAD_DIR
from ..core.milvus_client import COLLECTION_NAME
from ..models.orm import File, SessionLocal
from ..services.encoder_service import encode_document
from ..services.parser_service import parse_document, update_file_state

router = APIRouter(prefix="/upload", tags=["upload"])


def _process_document(file: File, milvus_client):
    """后台任务：解析 → 编码 → 入库。"""
    update_file_state(file.id, "parsing")
    markdown_path = parse_document(file)
    if markdown_path is None:
        return

    records = encode_document(markdown_path, file.id, file.original_name, file.filepath)
    if records:
        milvus_client.insert(collection_name=COLLECTION_NAME, data=records)
    update_file_state(file.id, "completed")


@router.post("/document")
async def upload_document(file: UploadFile, background_tasks: BackgroundTasks):
    if not file.filename:
        return {"error": "filename is required"}

    ext = os.path.splitext(file.filename)[1]
    save_name = f"{uuid.uuid4()}{ext}"
    save_path = os.path.join(str(UPLOAD_DIR), save_name)

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    with SessionLocal() as session:
        record = File(
            original_name=file.filename,
            filename=save_name,
            filepath=save_path,
            filestate="uploaded",
        )
        session.add(record)
        session.commit()
        file_id = record.id

    from .. import core
    background_tasks.add_task(_process_document, record, core._milvus_client)

    return {"id": file_id, "filename": file.filename, "state": "uploaded"}
