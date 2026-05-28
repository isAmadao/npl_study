import uuid
from datetime import datetime
from typing import List

from app.repositories.chunk_repo import ChunkRepo

class VectorService:
    def __init__(self):
        self.chunk_repo = ChunkRepo()
    
    async def encode_text(self, text: str) -> List[float]:
        return [0.0] * 100
    
    async def encode_image(self, image_path: str) -> List[float]:
        return [0.0] * 100
    
    async def generate_description(self, image_path: str) -> str:
        return "这是一张图片的描述"
    
    async def process_chunk(self, document_id: str, content_type: str, content: str, 
                           page_number: int, image_path: str = None):
        chunk_id = str(uuid.uuid4())
        
        chunk = {
            "id": chunk_id,
            "document_id": document_id,
            "content_type": content_type,
            "content": content,
            "page_number": page_number,
            "image_url": image_path,
            "created_at": datetime.now()
        }
        
        await self.chunk_repo.create(chunk)
        
        return chunk_id
