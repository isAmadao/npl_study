from __future__ import annotations

from datetime import datetime
from pathlib import Path

from rag_mvp.config import settings
from rag_mvp.db import SessionLocal
from rag_mvp.models import FileRecord, FileStatus
from rag_mvp.services.document_parser import DocumentParser
from rag_mvp.services.embedding_service import MultimodalEmbeddingService, TextEmbeddingService
from rag_mvp.services.vector_store import HybridVectorStore
from rag_mvp.types import ParsedDocument, VectorRecord
from rag_mvp.utils import chunk_text, pad_embedding


class IngestionService:
    def __init__(
        self,
        parser: DocumentParser | None = None,
        text_embedder: TextEmbeddingService | None = None,
        multimodal_embedder: MultimodalEmbeddingService | None = None,
        vector_store: HybridVectorStore | None = None,
        session_factory=SessionLocal,
    ) -> None:
        self.parser = parser or DocumentParser()
        self.text_embedder = text_embedder or TextEmbeddingService()
        self.multimodal_embedder = multimodal_embedder or MultimodalEmbeddingService(
            text_fallback=self.text_embedder
        )
        self.vector_store = vector_store or HybridVectorStore()
        self.session_factory = session_factory

    def process_file(self, file_id: int) -> None:
        with self.session_factory() as session:
            record = session.get(FileRecord, file_id)
            if record is None:
                raise ValueError(f"找不到文件记录：{file_id}")
            file_path = settings.root_dir / record.relative_path
            stored_name = record.stored_name
            record.status = FileStatus.PROCESSING
            record.status_detail = "正在解析文档并写入向量库。"
            session.commit()

        try:
            parsed = self.parser.parse(
                file_path=file_path,
                derived_dir=settings.derived_dir / Path(stored_name).stem,
            )
            vector_records = self._build_vector_records(record, file_path, parsed)
            if not vector_records:
                raise ValueError("文档中未提取到可写入索引的文本或图片内容。")
            self.vector_store.delete_by_file_id(record.id)
            self.vector_store.upsert_records(vector_records)

            with self.session_factory() as session:
                db_record = session.get(FileRecord, file_id)
                if db_record is not None:
                    db_record.status = FileStatus.COMPLETED
                    db_record.status_detail = f"处理完成，共写入 {len(vector_records)} 条向量记录。"
                    db_record.processed_at = datetime.utcnow()
                    session.commit()
        except Exception as exc:
            with self.session_factory() as session:
                db_record = session.get(FileRecord, file_id)
                if db_record is not None:
                    db_record.status = FileStatus.FAILED
                    db_record.status_detail = f"处理失败：{exc}"
                    session.commit()
            raise

    def _build_vector_records(
        self,
        record: FileRecord,
        file_path: Path,
        parsed: ParsedDocument,
    ) -> list[VectorRecord]:
        vector_records: list[VectorRecord] = []

        text_items: list[tuple[int, str]] = []
        for page_no, page_text in parsed.text_pages:
            for chunk in chunk_text(
                page_text,
                max_chars=settings.text_chunk_size,
                overlap=settings.text_chunk_overlap,
            ):
                text_items.append((page_no, chunk))

        if text_items:
            text_embeddings = self.text_embedder.encode([item[1] for item in text_items])
            for index, ((page_no, chunk), embedding) in enumerate(zip(text_items, text_embeddings), start=1):
                vector_records.append(
                    VectorRecord(
                        record_id=f"text-{record.id}-{index}",
                        file_id=record.id,
                        file_name=record.original_name,
                        source_path=str(file_path),
                        chunk_type="text",
                        page_no=page_no,
                        content=chunk,
                        image_path=None,
                        embedding=pad_embedding(list(embedding), settings.target_vector_dim),
                    )
                )

        if parsed.images:
            image_paths = [image.image_path for image in parsed.images]
            image_embeddings = self.multimodal_embedder.encode_images(
                image_paths,
                fallback_texts=[image.description for image in parsed.images],
            )
            for index, (image, embedding) in enumerate(zip(parsed.images, image_embeddings), start=1):
                vector_records.append(
                    VectorRecord(
                        record_id=f"image-{record.id}-{index}",
                        file_id=record.id,
                        file_name=record.original_name,
                        source_path=str(file_path),
                        chunk_type="image",
                        page_no=image.page_no,
                        content=image.description,
                        image_path=image.image_path,
                        embedding=pad_embedding(list(embedding), settings.target_vector_dim),
                    )
                )

        return vector_records
