from typing import List

class MessageRepo:
    def __init__(self):
        self.messages = {}
        self.conversation_messages = {}
    
    async def create(self, message: dict):
        self.messages[message["id"]] = message
        
        if message["conversation_id"] not in self.conversation_messages:
            self.conversation_messages[message["conversation_id"]] = []
        self.conversation_messages[message["conversation_id"]].append(message["id"])
    
    async def get_by_conversation_id(self, conversation_id: str) -> List[dict]:
        message_ids = self.conversation_messages.get(conversation_id, [])
        messages = [self.messages.get(mid) for mid in message_ids if mid in self.messages]
        messages.sort(key=lambda x: x["created_at"])
        return messages
    
    async def delete_by_conversation_id(self, conversation_id: str):
        message_ids = self.conversation_messages.get(conversation_id, [])
        for mid in message_ids:
            if mid in self.messages:
                del self.messages[mid]
        if conversation_id in self.conversation_messages:
            del self.conversation_messages[conversation_id]