from typing import List, Optional

class DocumentRepo:
    def __init__(self):
        self.documents = {}
    
    async def create(self, document: dict):
        self.documents[document["id"]] = document
    
    async def get_by_id(self, document_id: str) -> Optional[dict]:
        return self.documents.get(document_id)
    
    async def get_list(self, page: int, size: int, status: str = None) -> List[dict]:
        docs = list(self.documents.values())
        
        if status:
            docs = [d for d in docs if d["status"] == status]
        
        start = (page - 1) * size
        end = start + size
        return docs[start:end]
    
    async def get_count(self, status: str = None) -> int:
        docs = list(self.documents.values())
        if status:
            return sum(1 for d in docs if d["status"] == status)
        return len(docs)
    
    async def update(self, document_id: str, updates: dict):
        if document_id in self.documents:
            self.documents[document_id].update(updates)
    
    async def delete(self, document_id: str):
        if document_id in self.documents:
            del self.documents[document_id]