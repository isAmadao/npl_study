from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class Document(BaseModel):
    id: str
    name: str
    file_path: str
    page_count: int
    status: str
    created_at: datetime
    updated_at: datetime

class DocumentCreate(BaseModel):
    name: str
    file_path: str

class DocumentUpdate(BaseModel):
    status: Optional[str] = None
    page_count: Optional[int] = None