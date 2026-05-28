from pydantic import BaseModel
from datetime import datetime

class Conversation(BaseModel):
    id: str
    created_at: datetime
    updated_at: datetime

class ConversationCreate(BaseModel):
    pass
