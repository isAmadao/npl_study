from fastapi import APIRouter

from app.clients.clip_client import ClipClient
from app.clients.qwen_vl_client import QwenVLClient
from app.clients.llm_client import LLMClient

router = APIRouter()

clip_client = ClipClient()
qwen_vl_client = QwenVLClient()
llm_client = LLMClient()

@router.get("/status")
async def get_status():
    clip_status = await clip_client.get_status()
    qwen_vl_status = await qwen_vl_client.get_status()
    llm_status = await llm_client.get_status()
    
    return {
        "code": 200,
        "message": "success",
        "data": {
            "clip": clip_status,
            "qwen-vl": qwen_vl_status,
            "llm": llm_status
        }
    }