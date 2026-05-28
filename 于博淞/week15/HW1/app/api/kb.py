from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.document import KnowledgeBase
from app.schemas.kb import KBCreate, KBResponse

router = APIRouter(prefix="/kb", tags=["knowledge-base"])


@router.post("", response_model=KBResponse, status_code=201)
def create_kb(body: KBCreate, db: Session = Depends(get_db)):
    if db.query(KnowledgeBase).filter_by(name=body.name).first():
        raise HTTPException(status_code=409, detail="Knowledge base name already exists")
    kb = KnowledgeBase(name=body.name)
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return kb


@router.get("", response_model=list[KBResponse])
def list_kbs(db: Session = Depends(get_db)):
    return db.query(KnowledgeBase).all()
