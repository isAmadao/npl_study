from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from rag_mvp.config import settings
from rag_mvp.db import Base
from rag_mvp.models import FileRecord, FileStatus
from rag_mvp.services.ingest_service import IngestionService
from rag_mvp.services.vector_store import LocalVectorStore
from rag_mvp.types import ParsedDocument, ParsedImage


@contextmanager
def override_settings(**overrides):
    original_values = {key: getattr(settings, key) for key in overrides}
    try:
        for key, value in overrides.items():
            object.__setattr__(settings, key, value)
        yield
    finally:
        for key, value in original_values.items():
            object.__setattr__(settings, key, value)


class FakeParser:
    def __init__(self, image_path: str) -> None:
        self.image_path = image_path

    def parse(self, file_path: Path, derived_dir: Path) -> ParsedDocument:
        del file_path
        derived_dir.mkdir(parents=True, exist_ok=True)
        return ParsedDocument(
            text_pages=[(1, "这是一个用于测试切分和入库的段落。")],
            images=[
                ParsedImage(
                    image_path=self.image_path,
                    page_no=1,
                    description="测试图片",
                )
            ],
        )


class FakeTextEmbedder:
    def encode(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeMultimodalEmbedder:
    def encode_images(self, image_paths, fallback_texts=None):
        del fallback_texts
        return [[0.0, 1.0] for _ in image_paths]


class IngestServiceTestCase(unittest.TestCase):
    def test_process_file_updates_status_and_writes_vectors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            upload_dir = root / "uploads"
            derived_dir = upload_dir / "_derived"
            data_dir = root / "data"
            queue_dir = data_dir / "local_queue"
            upload_dir.mkdir(parents=True, exist_ok=True)
            derived_dir.mkdir(parents=True, exist_ok=True)
            queue_dir.mkdir(parents=True, exist_ok=True)

            db_path = root / "db.db"
            engine = create_engine(
                f"sqlite:///{db_path.as_posix()}",
                connect_args={"check_same_thread": False},
                future=True,
            )
            session_factory = sessionmaker(
                bind=engine,
                autoflush=False,
                autocommit=False,
                expire_on_commit=False,
                future=True,
            )
            Base.metadata.create_all(bind=engine)

            document_path = upload_dir / "sample.docx"
            document_path.write_text("placeholder", encoding="utf-8")
            image_path = str(root / "sample.png")
            Path(image_path).write_bytes(b"img")

            with session_factory() as session:
                record = FileRecord(
                    original_name="sample.docx",
                    stored_name="sample.docx",
                    relative_path=str(document_path.relative_to(root)),
                    document_type="docx",
                    size=document_path.stat().st_size,
                    status=FileStatus.QUEUED,
                    status_detail="waiting",
                )
                session.add(record)
                session.commit()
                session.refresh(record)
                file_id = record.id

            vector_store = LocalVectorStore(data_dir / "store.json")

            try:
                with override_settings(
                    root_dir=root,
                    upload_dir=upload_dir,
                    derived_dir=derived_dir,
                    data_dir=data_dir,
                    local_queue_dir=queue_dir,
                    local_vector_store_path=data_dir / "store.json",
                    db_path=db_path,
                ):
                    service = IngestionService(
                        parser=FakeParser(image_path=image_path),
                        text_embedder=FakeTextEmbedder(),
                        multimodal_embedder=FakeMultimodalEmbedder(),
                        vector_store=vector_store,
                        session_factory=session_factory,
                    )
                    service.process_file(file_id)

                with session_factory() as session:
                    updated = session.get(FileRecord, file_id)
                    self.assertEqual(updated.status, FileStatus.COMPLETED)
                    self.assertIn("处理完成", updated.status_detail)

                search_results = vector_store.search([1.0] + [0.0] * 1023, chunk_type="text", top_k=5)
                self.assertTrue(search_results)
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
