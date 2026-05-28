from typing import List

class ConversationRepo:
    def __init__(self):
        self.conversations = {}
    
    async def create(self, conversation: dict):
        self.conversations[conversation["id"]] = conversation
    
    async def get_by_id(self, conversation_id: str):
        return self.conversations.get(conversation_id)
    
    async def get_list(self, page: int, size: int) -> List[dict]:
        convs = list(self.conversations.values())
        convs.sort(key=lambda x: x["updated_at"], reverse=True)
        
        start = (page - 1) * size
        end = start + size
        return convs[start:end]
    
    async def get_count(self) -> int:
        return len(self.conversations)
    
    async def update(self, conversation_id: str, updates: dict):
        if conversation_id in self.conversations:
            self.conversations[conversation_id].update(updates)
    
    async def delete(self, conversation_id: str):
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]