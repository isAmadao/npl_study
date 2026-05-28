from pydantic import BaseModel


class UploadResponse(BaseModel):
    doc_id: int
    filename: str
    status: str
    message: str


class DocumentStatusResponse(BaseModel):
    doc_id: int
    status: str


class DocumentListResponse(BaseModel):
    doc_id: int
    kb_id: int
    filename: str
    status: str
    created_at: str
