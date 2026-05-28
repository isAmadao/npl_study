import uuid
from datetime import datetime
from typing import List, Optional

from app.services.search_service import SearchService
from app.clients.llm_client import LLMClient
from app.repositories.conversation_repo import ConversationRepo
from app.repositories.message_repo import MessageRepo

class QAService:
    def __init__(self):
        self.search_service = SearchService()
        self.llm_client = LLMClient()
        self.conversation_repo = ConversationRepo()
        self.message_repo = MessageRepo()
    
    async def ask_question(self, question: str, conversation_id: str = None, 
                          document_ids: Optional[List[str]] = None):
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            await self.conversation_repo.create({
                "id": conversation_id,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            })
        
        search_results = await self.search_service.search(question, top_k=5, document_ids=document_ids)
        
        context = "\n\n".join([
            f"来源{idx+1} ({result['content_type']}): {result['content']}"
            for idx, result in enumerate(search_results)
        ])
        
        if context:
            prompt = f"""你是一个多模态知识问答助手。请根据提供的参考资料回答用户问题。

参考资料：
{context}

用户问题：
{question}

请直接给出详细的回答。"""
        else:
            prompt = f"请回答以下问题：{question}"
        
        answer = await self.llm_client.generate_answer(prompt)
        
        references = [{
            "chunk_id": result["id"],
            "document_id": result["document_id"],
            "content_type": result["content_type"],
            "image_url": result.get("image_url")
        } for result in search_results]
        
        user_message_id = str(uuid.uuid4())
        await self.message_repo.create({
            "id": user_message_id,
            "conversation_id": conversation_id,
            "role": "user",
            "content": question,
            "created_at": datetime.now()
        })
        
        assistant_message_id = str(uuid.uuid4())
        await self.message_repo.create({
            "id": assistant_message_id,
            "conversation_id": conversation_id,
            "role": "assistant",
            "content": answer,
            "references": references,
            "created_at": datetime.now()
        })
        
        await self.conversation_repo.update(conversation_id, {"updated_at": datetime.now()})
        
        return {
            "conversation_id": conversation_id,
            "answer": answer,
            "references": references,
            "created_at": datetime.now().isoformat()
        }
    
    async def get_conversation_list(self, page: int, size: int):
        conversations = await self.conversation_repo.get_list(page, size)
        total = await self.conversation_repo.get_count()
        
        return {
            "items": conversations,
            "total": total,
            "page": page,
            "size": size
        }
    
    async def get_conversation_detail(self, conversation_id: str):
        messages = await self.message_repo.get_by_conversation_id(conversation_id)
        
        if not messages:
            return None
        
        return {
            "conversation_id": conversation_id,
            "messages": messages
        }
    
    async def delete_conversation(self, conversation_id: str):
        await self.message_repo.delete_by_conversation_id(conversation_id)
        await self.conversation_repo.delete(conversation_id)
