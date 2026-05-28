from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.v1 import documents, search, qa, images, vector_db, models
from app.core.config import settings
from app.core.logging import setup_logging

setup_logging()

app = FastAPI(
    title="多模态RAG聊天框",
    description="基于CLIP/Qwen-VL的多模态检索问答系统 - Demo版本",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/api/v1/documents", tags=["文档管理"])
app.include_router(search.router, prefix="/api/v1/search", tags=["检索"])
app.include_router(qa.router, prefix="/api/v1/qa", tags=["问答"])
app.include_router(images.router, prefix="/api/v1/images", tags=["图像服务"])
app.include_router(vector_db.router, prefix="/api/v1/vector/db", tags=["向量库管理"])
app.include_router(models.router, prefix="/api/v1/models", tags=["模型服务"])

Instrumentator().instrument(app).expose(app)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug
    )