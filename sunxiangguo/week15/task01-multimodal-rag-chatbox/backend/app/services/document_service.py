import uuid
from datetime import datetime
from fastapi import UploadFile

from app.repositories.document_repo import DocumentRepo

class DocumentService:
    def __init__(self):
        self.document_repo = DocumentRepo()
    
    async def upload_document(self, file: UploadFile):
        document_id = str(uuid.uuid4())
        file_path = f"documents/{document_id}/{file.filename}"
        
        document = {
            "id": document_id,
            "name": file.filename,
            "file_path": file_path,
            "page_count": 0,
            "status": "pending",
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        
        await self.document_repo.create(document)
        
        return {
            "id": document_id,
            "name": file.filename,
            "status": "pending",
            "created_at": document["created_at"].isoformat()
        }
    
    async def get_document_list(self, page: int, size: int, status: str = None):
        documents = await self.document_repo.get_list(page, size, status)
        total = await self.document_repo.get_count(status)
        
        return {
            "items": documents,
            "total": total,
            "page": page,
            "size": size
        }
    
    async def get_document_detail(self, document_id: str):
        return await self.document_repo.get_by_id(document_id)
    
    async def delete_document(self, document_id: str):
        await self.document_repo.delete(document_id)
