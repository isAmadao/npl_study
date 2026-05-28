from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from app.services.search_service import SearchService

router = APIRouter()

search_service = SearchService()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    document_ids: Optional[List[str]] = None

@router.post("/")
async def search(request: SearchRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    results = await search_service.search(request.query, request.top_k, request.document_ids)
    return {
        "code": 200,
        "message": "success",
        "data": {
            "query": request.query,
            "results": results
        }
    }