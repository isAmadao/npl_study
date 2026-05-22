from pydantic import BaseModel, Field


class Source(BaseModel):
    filename: str
    page_num: int
    chunk_type: str  # "text" | "image"
    score: float


class ChatRequest(BaseModel):
    kb_id: int
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    doc_count: int  # number of indexed docs searched
