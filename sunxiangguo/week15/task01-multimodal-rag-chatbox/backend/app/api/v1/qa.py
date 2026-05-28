from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from app.services.qa_service import QAService

router = APIRouter()

qa_service = QAService()

class QARequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    document_ids: Optional[List[str]] = None

@router.post("/ask")
async def ask_question(request: QARequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    result = await qa_service.ask_question(request.question, request.conversation_id, request.document_ids)
    return {
        "code": 200,
        "message": "success",
        "data": result
    }

@router.get("/conversations")
async def get_conversation_list(page: int = 1, size: int = 10):
    result = await qa_service.get_conversation_list(page, size)
    return {
        "code": 200,
        "message": "success",
        "data": result
    }

@router.get("/conversations/{conversation_id}")
async def get_conversation_detail(conversation_id: str):
    result = await qa_service.get_conversation_detail(conversation_id)
    if not result:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "code": 200,
        "message": "success",
        "data": result
    }

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    await qa_service.delete_conversation(conversation_id)
    return {
        "code": 200,
        "message": "success",
        "data": None
    }