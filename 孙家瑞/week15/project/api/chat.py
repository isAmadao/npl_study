"""POST /chat —— 多模态 RAG 问答。"""

from fastapi import APIRouter
from pydantic import BaseModel

from ..core.milvus_client import COLLECTION_NAME
from ..services.chat_service import chat

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    context: str


@router.post("", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    from .. import core
    result = chat(req.question, core._milvus_client)
    return ChatResponse(**result)
