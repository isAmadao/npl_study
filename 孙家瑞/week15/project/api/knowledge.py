"""知识库管理接口 —— 文档列表和删除。"""

import os

from fastapi import APIRouter
from pydantic import BaseModel

from ..core.milvus_client import COLLECTION_NAME
from ..models.orm import File, SessionLocal

router = APIRouter(prefix="/documents", tags=["knowledge"])


class DocumentOut(BaseModel):
    id: int
    original_name: str
    filestate: str
    error_message: str | None = None
    created_at: str | None = None

    model_config = {"from_attributes": True}


@router.get("", response_model=list[DocumentOut])
async def list_documents():
    with SessionLocal() as session:
        files = session.query(File).order_by(File.id.desc()).all()
        return [
            DocumentOut(
                id=f.id,
                original_name=f.original_name,
                filestate=f.filestate,
                error_message=f.error_message,
                created_at=str(f.created_at) if f.created_at else None,
            )
            for f in files
        ]


@router.delete("/{file_id}")
async def delete_document(file_id: int):
    from .. import core

    with SessionLocal() as session:
        f = session.query(File).filter(File.id == file_id).first()
        if not f:
            return {"error": "file not found"}
        filepath = f.filepath
        session.delete(f)
        session.commit()

    if os.path.exists(filepath):
        os.remove(filepath)

    core._milvus_client.delete(collection_name=COLLECTION_NAME, filter=f"file_id == {file_id}")

    return {"deleted": file_id}
