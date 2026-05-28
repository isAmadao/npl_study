from typing import List, Optional

from app.repositories.chunk_repo import ChunkRepo

class SearchService:
    def __init__(self):
        self.chunk_repo = ChunkRepo()
    
    async def search(self, query: str, top_k: int = 5, document_ids: Optional[List[str]] = None) -> List[dict]:
        all_chunks = await self.chunk_repo.get_all()
        
        if document_ids:
            all_chunks = [chunk for chunk in all_chunks if chunk["document_id"] in document_ids]
        
        search_results = []
        for chunk in all_chunks[:top_k]:
            search_results.append({
                "id": chunk["id"],
                "document_id": chunk["document_id"],
                "content_type": chunk["content_type"],
                "content": chunk["content"],
                "image_url": chunk.get("image_url"),
                "page_number": chunk["page_number"],
                "similarity": 0.9
            })
        
        return search_results
