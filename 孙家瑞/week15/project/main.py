"""多模态 RAG Chatbot —— FastAPI 应用入口。"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import core
from .api import chat, knowledge, upload
from .core.encoders import BgeEncoder, ClipEncoder, QwenChatModel  # noqa: F401  注册默认实现
from .core.milvus_client import create_client, ensure_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型、连接 Milvus；关闭时释放资源。"""
    milvus = create_client()
    ensure_collection(milvus)
    core._milvus_client = milvus
    yield
    milvus.close()


app = FastAPI(
    title="多模态 RAG Chatbot",
    description="基于 Mineru + BGE + CLIP + Milvus + Qwen 的本地优先多模态文档问答系统",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(knowledge.router)


@app.get("/health")
async def health():
    return {"status": "ok", "milvus": core._milvus_client is not None}
