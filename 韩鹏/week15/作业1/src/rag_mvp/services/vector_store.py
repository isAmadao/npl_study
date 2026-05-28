from __future__ import annotations

import json
from pathlib import Path

from rag_mvp.config import settings
from rag_mvp.types import SearchResult, VectorRecord
from rag_mvp.utils import cosine_similarity


class LocalVectorStore:
    def __init__(self, store_path: Path | None = None) -> None:
        self.store_path = Path(store_path or settings.local_vector_store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def _read_all(self) -> list[VectorRecord]:
        if not self.store_path.exists():
            return []
        content = self.store_path.read_text(encoding="utf-8").strip()
        if not content:
            return []
        payload = json.loads(content)
        return [VectorRecord.from_dict(item) for item in payload]

    def _write_all(self, records: list[VectorRecord]) -> None:
        payload = [record.to_dict() for record in records]
        self.store_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def upsert_records(self, records: list[VectorRecord]) -> None:
        existing = {record.record_id: record for record in self._read_all()}
        for record in records:
            existing[record.record_id] = record
        self._write_all(list(existing.values()))

    def delete_by_file_id(self, file_id: int) -> None:
        records = [record for record in self._read_all() if record.file_id != file_id]
        self._write_all(records)

    def has_chunk_type(self, chunk_type: str) -> bool:
        return any(record.chunk_type == chunk_type for record in self._read_all())

    def search(self, query_vector: list[float], chunk_type: str, top_k: int = 4) -> list[SearchResult]:
        results: list[SearchResult] = []
        for record in self._read_all():
            if record.chunk_type != chunk_type:
                continue
            score = cosine_similarity(query_vector, record.embedding)
            results.append(
                SearchResult(
                    record_id=record.record_id,
                    file_id=record.file_id,
                    file_name=record.file_name,
                    source_path=record.source_path,
                    chunk_type=record.chunk_type,
                    page_no=record.page_no,
                    content=record.content,
                    image_path=record.image_path,
                    score=score,
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]


class MilvusVectorStore:
    def __init__(self) -> None:
        self.uri = settings.milvus_uri
        self.token = settings.milvus_token
        self.collection_name = settings.milvus_collection_name
        self.alias = "default"
        self._collection = None

    def _get_collection(self):
        if self._collection is not None:
            return self._collection

        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

        connections.connect(alias=self.alias, uri=self.uri, token=self.token)
        if not utility.has_collection(self.collection_name, using=self.alias):
            fields = [
                FieldSchema(name="record_id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
                FieldSchema(name="file_id", dtype=DataType.INT64),
                FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="source_path", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=32),
                FieldSchema(name="page_no", dtype=DataType.INT64),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.target_vector_dim),
            ]
            schema = CollectionSchema(fields=fields, description="RAG MVP knowledge base")
            collection = Collection(name=self.collection_name, schema=schema, using=self.alias)
            collection.create_index(
                field_name="embedding",
                index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE", "params": {}},
            )
        self._collection = Collection(name=self.collection_name, using=self.alias)
        self._collection.load()
        return self._collection

    def upsert_records(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        collection = self._get_collection()
        payload = [
            {
                "record_id": record.record_id,
                "file_id": record.file_id,
                "file_name": record.file_name,
                "source_path": record.source_path,
                "chunk_type": record.chunk_type,
                "page_no": record.page_no,
                "content": record.content[:8192],
                "image_path": record.image_path or "",
                "embedding": record.embedding,
            }
            for record in records
        ]
        collection.insert(payload)
        collection.flush()

    def delete_by_file_id(self, file_id: int) -> None:
        collection = self._get_collection()
        collection.delete(expr=f"file_id == {file_id}")
        collection.flush()

    def search(self, query_vector: list[float], chunk_type: str, top_k: int = 4) -> list[SearchResult]:
        collection = self._get_collection()
        hits = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {}},
            limit=top_k,
            expr=f"chunk_type == '{chunk_type}'",
            output_fields=["record_id", "file_id", "file_name", "source_path", "chunk_type", "page_no", "content", "image_path"],
        )
        results: list[SearchResult] = []
        for hit in hits[0]:
            entity = hit.entity
            results.append(
                SearchResult(
                    record_id=str(entity.get("record_id")),
                    file_id=int(entity.get("file_id")),
                    file_name=str(entity.get("file_name")),
                    source_path=str(entity.get("source_path")),
                    chunk_type=str(entity.get("chunk_type")),
                    page_no=int(entity.get("page_no")),
                    content=str(entity.get("content")),
                    image_path=(entity.get("image_path") or None),
                    score=float(hit.score),
                )
            )
        return results


class HybridVectorStore:
    def __init__(self) -> None:
        self.local_store = LocalVectorStore()
        self.milvus_store = MilvusVectorStore()

    def upsert_records(self, records: list[VectorRecord]) -> None:
        self.local_store.upsert_records(records)
        try:
            self.milvus_store.upsert_records(records)
        except Exception:
            pass

    def delete_by_file_id(self, file_id: int) -> None:
        self.local_store.delete_by_file_id(file_id)
        try:
            self.milvus_store.delete_by_file_id(file_id)
        except Exception:
            pass

    def has_chunk_type(self, chunk_type: str) -> bool:
        return self.local_store.has_chunk_type(chunk_type)

    def search(self, query_vector: list[float], chunk_type: str, top_k: int = 4) -> list[SearchResult]:
        try:
            results = self.milvus_store.search(query_vector, chunk_type, top_k)
            if results:
                return results
        except Exception:
            pass
        return self.local_store.search(query_vector, chunk_type, top_k)
