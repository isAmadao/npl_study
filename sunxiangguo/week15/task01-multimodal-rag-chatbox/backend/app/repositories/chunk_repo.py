from typing import List, Optional

class ChunkRepo:
    def __init__(self):
        self.chunks = {}
        self.document_chunks = {}
    
    async def create(self, chunk: dict):
        self.chunks[chunk["id"]] = chunk
        
        if chunk["document_id"] not in self.document_chunks:
            self.document_chunks[chunk["document_id"]] = []
        self.document_chunks[chunk["document_id"]].append(chunk["id"])
    
    async def get_by_id(self, chunk_id: str) -> Optional[dict]:
        return self.chunks.get(chunk_id)
    
    async def get_by_ids(self, chunk_ids: List[str]) -> List[dict]:
        return [self.chunks.get(cid) for cid in chunk_ids if cid in self.chunks]
    
    async def get_by_document_id(self, document_id: str) -> List[dict]:
        chunk_ids = self.document_chunks.get(document_id, [])
        return [self.chunks.get(cid) for cid in chunk_ids if cid in self.chunks]
    
    async def delete_by_document_id(self, document_id: str):
        chunk_ids = self.document_chunks.get(document_id, [])
        for cid in chunk_ids:
            if cid in self.chunks:
                del self.chunks[cid]
        if document_id in self.document_chunks:
            del self.document_chunks[document_id]
    
    async def get_all(self) -> List[dict]:
        return list(self.chunks.values())