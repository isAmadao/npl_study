from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Dict

class Message(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    references: Optional[List[Dict]] = None
    created_at: datetime

class MessageCreate(BaseModel):
    conversation_id: Optional[str] = None
    role: str
    content: str
    references: Optional[List[Dict]] = None