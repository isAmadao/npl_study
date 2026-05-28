from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import Optional

from app.services.document_service import DocumentService

router = APIRouter()

document_service = DocumentService()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    result = await document_service.upload_document(file)
    return {
        "code": 200,
        "message": "success",
        "data": result
    }

@router.get("/")
async def get_document_list(page: int = 1, size: int = 10, status: Optional[str] = None):
    result = await document_service.get_document_list(page, size, status)
    return {
        "code": 200,
        "message": "success",
        "data": result
    }

@router.get("/{document_id}")
async def get_document_detail(document_id: str):
    result = await document_service.get_document_detail(document_id)
    if not result:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "code": 200,
        "message": "success",
        "data": result
    }

@router.delete("/{document_id}")
async def delete_document(document_id: str):
    await document_service.delete_document(document_id)
    return {
        "code": 200,
        "message": "success",
        "data": None
    }