from typing import List, Optional

class MilvusClient:
    async def get_status(self):
        return {
            "collection_count": 0,
            "vector_count": 0,
            "disk_usage": "0MB",
            "status": "healthy"
        }
    
    async def rebuild_index(self, document_id: Optional[str] = None):
        return {
            "status": "started",
            "task_id": "demo-task-id"
        }
    
    async def search_vectors(self, query_vector: List[float], top_k: int = 5, 
                            document_ids: Optional[List[str]] = None) -> List[dict]:
        return []
    
    async def insert_vector(self, vector_data: dict):
        pass
