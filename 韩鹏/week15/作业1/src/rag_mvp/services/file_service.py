from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select

from rag_mvp.config import settings
from rag_mvp.db import SessionLocal
from rag_mvp.models import FileRecord, FileStatus
from rag_mvp.services.queue_service import TaskQueue
from rag_mvp.services.vector_store import HybridVectorStore
from rag_mvp.types import FileTask


class FileService:
    def __init__(self, session_factory=SessionLocal, queue=None, vector_store=None) -> None:
        self.session_factory = session_factory
        self.queue = TaskQueue()
        if queue is not None:
            self.queue = queue
        self.vector_store = vector_store or HybridVectorStore()

    def list_files(self) -> list[FileRecord]:
        with self.session_factory() as session:
            stmt = select(FileRecord).order_by(FileRecord.created_at.desc())
            return list(session.scalars(stmt).all())

    def save_uploaded_file(self, uploaded_file):
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix not in {".pdf", ".docx"}:
            raise ValueError("仅支持 PDF 或 DOCX 文件。")

        settings.ensure_directories()
        stored_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid4().hex}{suffix}"
        file_path = settings.upload_dir / stored_name
        file_path.write_bytes(uploaded_file.getbuffer())

        with self.session_factory() as session:
            record = FileRecord(
                original_name=uploaded_file.name,
                stored_name=stored_name,
                relative_path=str(file_path.relative_to(settings.root_dir)),
                document_type=suffix.lstrip("."),
                size=file_path.stat().st_size,
                status=FileStatus.PENDING,
                status_detail="文件已保存，等待投递处理任务。",
            )
            session.add(record)
            session.commit()
            session.refresh(record)

            dispatch = self.queue.publish(
                FileTask(
                    file_id=record.id,
                    file_path=str(file_path),
                    original_name=record.original_name,
                    suffix=suffix,
                )
            )
            record.status = FileStatus.QUEUED
            record.status_detail = dispatch.message
            session.commit()
            session.refresh(record)
            return record, dispatch

    def delete_file(self, file_id: int) -> None:
        with self.session_factory() as session:
            record = session.get(FileRecord, file_id)
            if record is None:
                raise ValueError(f"文件记录不存在：{file_id}")

            file_path = settings.root_dir / record.relative_path
            if file_path.exists():
                file_path.unlink()

            derived_dir = settings.derived_dir / Path(record.stored_name).stem
            if derived_dir.exists():
                shutil.rmtree(derived_dir)

            self.vector_store.delete_by_file_id(file_id)
            session.delete(record)
            session.commit()
