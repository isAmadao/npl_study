"""
Milvus 向量数据库封装：管理 text_chunks 和 image_chunks 两个 Collection，
支持插入和检索操作。兼容 milvus-lite (嵌入式) 和独立 Milvus 服务。
"""
from __future__ import annotations

import numpy as np
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)

import config


class MilvusStore:
    TEXT_COLLECTION = "text_chunks"
    IMAGE_COLLECTION = "image_chunks"

    def __init__(self):
        if config.MILVUS_USE_LITE:
            from pymilvus import MilvusClient
            connections.connect(
                alias="default",
                uri=config.MILVUS_LITE_PATH,
            )
        else:
            connections.connect(
                alias="default",
                host=config.MILVUS_HOST,
                port=config.MILVUS_PORT,
            )

    # ---------- Collection 管理 ----------
    def ensure_collections(self):
        """确保两个 Collection 已创建。"""
        self._ensure_text_collection()
        self._ensure_image_collection()

    def _ensure_text_collection(self):
        if utility.has_collection(self.TEXT_COLLECTION):
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="kb_id", dtype=DataType.INT64),
            FieldSchema(name="doc_id", dtype=DataType.INT64),
            FieldSchema(name="chunk_id", dtype=DataType.INT64),
            FieldSchema(name="page_num", dtype=DataType.INT64),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=config.TEXT_EMBEDDING_DIM),
        ]
        schema = CollectionSchema(fields, description="文本 chunks 向量集合")
        collection = Collection(name=self.TEXT_COLLECTION, schema=schema)

        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()

    def _ensure_image_collection(self):
        if utility.has_collection(self.IMAGE_COLLECTION):
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="kb_id", dtype=DataType.INT64),
            FieldSchema(name="doc_id", dtype=DataType.INT64),
            FieldSchema(name="chunk_id", dtype=DataType.INT64),
            FieldSchema(name="page_num", dtype=DataType.INT64),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=config.IMAGE_EMBEDDING_DIM),
        ]
        schema = CollectionSchema(fields, description="图片 chunks 向量集合")
        collection = Collection(name=self.IMAGE_COLLECTION, schema=schema)

        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()

    # ---------- 插入 ----------
    def insert_text_chunks(
        self,
        kb_id: int,
        doc_id: int,
        chunks: list[str],
        embeddings: np.ndarray,
        page_nums: list[int] | None = None,
    ) -> list[int]:
        """插入文本 chunks，返回 Milvus 内部 ID 列表。"""
        collection = Collection(name=self.TEXT_COLLECTION)
        if page_nums is None:
            page_nums = [0] * len(chunks)

        n = len(chunks)
        data = [
            [kb_id] * n,
            [doc_id] * n,
            list(range(n)),
            page_nums,
            [c[:4096] for c in chunks],
            embeddings.tolist(),
        ]

        mr = collection.insert(data)
        collection.flush()
        return mr.primary_keys

    def insert_image_chunks(
        self,
        kb_id: int,
        doc_id: int,
        image_paths: list[str],
        embeddings: np.ndarray,
        page_nums: list[int] | None = None,
    ) -> list[int]:
        """插入图片 chunks，返回 Milvus 内部 ID 列表。"""
        collection = Collection(name=self.IMAGE_COLLECTION)
        if page_nums is None:
            page_nums = [0] * len(image_paths)

        n = len(image_paths)
        data = [
            [kb_id] * n,
            [doc_id] * n,
            list(range(n)),
            page_nums,
            image_paths,
            embeddings.tolist(),
        ]

        mr = collection.insert(data)
        collection.flush()
        return mr.primary_keys

    # ---------- 检索 ----------
    def search_text(
        self,
        kb_id: int,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:
        """基于文本 embedding 检索最相似的文本 chunks。"""
        collection = Collection(name=self.TEXT_COLLECTION)
        collection.load()

        search_params = {"metric_type": "IP", "params": {"nprobe": 16}}
        results = collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=f"kb_id == {kb_id}",
            output_fields=["doc_id", "kb_id", "chunk_id", "page_num", "content"],
        )

        items = []
        for hits in results:
            for hit in hits:
                items.append({
                    "milvus_id": hit.id,
                    "doc_id": hit.entity.get("doc_id"),
                    "kb_id": hit.entity.get("kb_id"),
                    "chunk_id": hit.entity.get("chunk_id"),
                    "page_num": hit.entity.get("page_num"),
                    "content": hit.entity.get("content"),
                    "score": float(hit.distance),
                })
        return items

    def search_images(
        self,
        kb_id: int,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:
        """基于 embedding 检索最相似的图片 chunks。"""
        collection = Collection(name=self.IMAGE_COLLECTION)
        collection.load()

        search_params = {"metric_type": "IP", "params": {"nprobe": 16}}
        results = collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=f"kb_id == {kb_id}",
            output_fields=["doc_id", "kb_id", "chunk_id", "page_num", "file_path"],
        )

        items = []
        for hits in results:
            for hit in hits:
                items.append({
                    "milvus_id": hit.id,
                    "doc_id": hit.entity.get("doc_id"),
                    "kb_id": hit.entity.get("kb_id"),
                    "chunk_id": hit.entity.get("chunk_id"),
                    "page_num": hit.entity.get("page_num"),
                    "file_path": hit.entity.get("file_path"),
                    "score": float(hit.distance),
                })
        return items

    def search_images_by_clip_text(
        self,
        kb_id: int,
        clip_text_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:
        """使用 CLIP 文本向量跨模态检索图片。"""
        collection = Collection(name=self.IMAGE_COLLECTION)
        collection.load()

        search_params = {"metric_type": "IP", "params": {"nprobe": 16}}
        results = collection.search(
            data=[clip_text_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=f"kb_id == {kb_id}",
            output_fields=["doc_id", "kb_id", "chunk_id", "page_num", "file_path"],
        )

        items = []
        for hits in results:
            for hit in hits:
                items.append({
                    "milvus_id": hit.id,
                    "doc_id": hit.entity.get("doc_id"),
                    "kb_id": hit.entity.get("kb_id"),
                    "chunk_id": hit.entity.get("chunk_id"),
                    "page_num": hit.entity.get("page_num"),
                    "file_path": hit.entity.get("file_path"),
                    "score": float(hit.distance),
                })
        return items