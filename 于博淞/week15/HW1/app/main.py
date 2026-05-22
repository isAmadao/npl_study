from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import chat, kb, upload
from app.core.database import Base, engine
from app.core.kafka import close_producer


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    try:
        from app.db.milvus import init_collections
        init_collections()
    except Exception:
        # Milvus may not be available in local dev — worker initialises collections on startup
        pass
    yield
    close_producer()


app = FastAPI(title="Multimodal RAG API", version="0.1.0", lifespan=lifespan)

app.include_router(kb.router, prefix="/api/v1")
app.include_router(upload.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
