from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.document import Document, KnowledgeBase
from app.schemas.chat import ChatRequest, ChatResponse, Source
from app.services.qa import generate_answer, stream_answer
from app.services.retriever import retrieve

router = APIRouter(prefix="/chat", tags=["chat"])


def _get_done_doc_ids(kb_id: int, db: Session) -> list[int]:
    """Return doc_ids that have finished indexing for the given knowledge base."""
    rows = (
        db.query(Document.id)
        .filter(Document.kb_id == kb_id, Document.status == "done")
        .all()
    )
    return [r.id for r in rows]


def _build_sources(results: list[dict]) -> list[Source]:
    return [
        Source(
            filename=r["filename"],
            page_num=r["page_num"],
            chunk_type=r["chunk_type"],
            score=round(r["score"], 4),
        )
        for r in results
    ]


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest, db: Session = Depends(get_db)):
    if not db.get(KnowledgeBase, req.kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    doc_ids = _get_done_doc_ids(req.kb_id, db)
    if not doc_ids:
        raise HTTPException(
            status_code=422,
            detail="No indexed documents in this knowledge base yet. "
                   "Upload a PDF and wait for the worker to finish processing.",
        )

    results = retrieve(query=req.query, doc_ids=doc_ids, top_k=req.top_k)
    answer = await generate_answer(query=req.query, context=results)

    return ChatResponse(
        answer=answer,
        sources=_build_sources(results),
        doc_count=len(doc_ids),
    )


@router.post("/stream")
async def chat_stream(req: ChatRequest, db: Session = Depends(get_db)):
    """
    SSE streaming endpoint. Client receives chunks as:
        data: {"text": "..."}\n\n
        data: [DONE]\n\n
    """
    if not db.get(KnowledgeBase, req.kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    doc_ids = _get_done_doc_ids(req.kb_id, db)
    if not doc_ids:
        raise HTTPException(
            status_code=422,
            detail="No indexed documents in this knowledge base yet.",
        )

    results = retrieve(query=req.query, doc_ids=doc_ids, top_k=req.top_k)

    return StreamingResponse(
        stream_answer(query=req.query, context=results),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )
