import os
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.kafka import get_producer
from app.db.session import get_db
from app.models.document import Document, KnowledgeBase
from app.schemas.upload import DocumentListResponse, DocumentStatusResponse, UploadResponse

router = APIRouter(prefix="/upload", tags=["upload"])

_ALLOWED_CONTENT_TYPES = {"application/pdf"}
_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


def _validate_pdf(file: UploadFile) -> None:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    if file.content_type and file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type '{file.content_type}'. Expected application/pdf",
        )


@router.post("/document", response_model=UploadResponse, status_code=202)
async def upload_document(
    kb_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    _validate_pdf(file)

    if not db.get(KnowledgeBase, kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    raw = await file.read()
    if len(raw) > _MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 100 MB limit")

    os.makedirs(settings.pdf_storage_path, exist_ok=True)
    saved_name = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(settings.pdf_storage_path, saved_name)
    with open(file_path, "wb") as f:
        f.write(raw)

    doc = Document(kb_id=kb_id, filename=saved_name, original_name=file.filename)
    db.add(doc)
    db.commit()
    db.refresh(doc)

    producer = get_producer()
    future = producer.send(
        settings.kafka_topic_parse,
        {"doc_id": doc.id, "file_path": file_path},
    )
    # Block until the broker confirms receipt (acks="all" already set on producer)
    future.get(timeout=10)

    return UploadResponse(
        doc_id=doc.id,
        filename=file.filename,
        status="pending",
        message="Document queued for processing",
    )


@router.get("/document/{doc_id}/status", response_model=DocumentStatusResponse)
def get_document_status(doc_id: int, db: Session = Depends(get_db)):
    doc = db.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentStatusResponse(doc_id=doc.id, status=doc.status)


@router.get("/documents", response_model=list[DocumentListResponse])
def list_documents(kb_id: int | None = None, db: Session = Depends(get_db)):
    q = db.query(Document)
    if kb_id is not None:
        q = q.filter(Document.kb_id == kb_id)
    return [
        DocumentListResponse(
            doc_id=d.id,
            kb_id=d.kb_id,
            filename=d.original_name,
            status=d.status,
            created_at=d.created_at.isoformat(),
        )
        for d in q.order_by(Document.created_at.desc()).all()
    ]
