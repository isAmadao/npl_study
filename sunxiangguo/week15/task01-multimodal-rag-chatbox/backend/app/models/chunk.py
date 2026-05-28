from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict

class Chunk(BaseModel):
    id: str
    document_id: str
    content_type: str
    content: str
    page_number: int
    position: Optional[Dict] = None
    image_url: Optional[str] = None
    created_at: datetime

class ChunkCreate(BaseModel):
    document_id: str
    content_type: str
    content: str
    page_number: int
    position: Optional[Dict] = None
    image_url: Optional[str] = None