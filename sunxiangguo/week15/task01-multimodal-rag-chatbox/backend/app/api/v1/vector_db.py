from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from app.clients.milvus_client import MilvusClient

router = APIRouter()

milvus_client = MilvusClient()

class RebuildRequest(BaseModel):
    document_id: Optional[str] = None

@router.get("/status")
async def get_status():
    status = await milvus_client.get_status()
    return {
        "code": 200,
        "message": "success",
        "data": status
    }

@router.post("/rebuild")
async def rebuild_index(request: RebuildRequest = None):
    document_id = request.document_id if request else None
    result = await milvus_client.rebuild_index(document_id)
    return {
        "code": 200,
        "message": "success",
        "data": result
    }