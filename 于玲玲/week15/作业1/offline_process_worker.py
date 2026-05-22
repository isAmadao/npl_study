#!/usr/bin/env python3
"""offline_process_worker — 离线解析 Worker（Kafka 消费者）"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from kafka import KafkaConsumer, KafkaError

import config
from models import get_db, _now
from utils.mineru_client import parse_pdf_with_mineru
from utils.embedding import TextEmbedder, ImageEmbedder
from utils.milvus_client import MilvusStore
from utils.storage import ensure_kb_dirs, get_chunk_images_dir


class OfflineProcessWorker:
    def __init__(self):
        self.text_embedder = TextEmbedder()
        self.image_embedder = ImageEmbedder()
        self.milvus = MilvusStore()

    def run(self):
        self.milvus.ensure_collections()

        consumer = KafkaConsumer(
            config.KAFKA_TOPIC_DOCUMENT_PARSE,
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            group_id=config.KAFKA_GROUP_ID,
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            max_poll_interval_ms=600000,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )

        print(f"[Worker] 已连接 Kafka, 监听 topic: {config.KAFKA_TOPIC_DOCUMENT_PARSE}")

        for message in consumer:
            data = message.value
            doc_id = data["doc_id"]
            kb_id = data["kb_id"]
            filepath = data["filepath"]
            filename = data.get("filename", "")

            print(f"[Worker] 收到任务: doc_id={doc_id}, kb_id={kb_id}, file={filename}")

            try:
                # 1. 标记为 processing
                self._update_status(doc_id, "processing")

                # 2. 调用 MinerU 解析 PDF
                parse_result = parse_pdf_with_mineru(filepath, kb_id, doc_id)

                # 3. 处理文本 chunks
                text_chunks = parse_result.get("text_chunks", [])
                image_items = parse_result.get("images", [])
                page_count = parse_result.get("page_count", 0)

                total_chunks = 0

                # 4. 文本 embedding + 存入 Milvus
                if text_chunks:
                    text_embeddings = self.text_embedder.encode(text_chunks)
                    text_ids = self.milvus.insert_text_chunks(
                        kb_id=kb_id,
                        doc_id=doc_id,
                        chunks=text_chunks,
                        embeddings=text_embeddings,
                        page_nums=parse_result.get("text_page_nums", []),
                    )
                    # 写入 SQLite chunks 记录
                    now = _now()
                    with get_db() as conn:
                        for i, (chunk, m_id) in enumerate(zip(text_chunks, text_ids)):
                            conn.execute(
                                """INSERT INTO chunks (doc_id, kb_id, chunk_type, content, page_num, chunk_index, milvus_id, created_at)
                                   VALUES (?, ?, 'text', ?, ?, ?, ?, ?)""",
                                (doc_id, kb_id, chunk, parse_result.get("text_page_nums", [0]*len(text_chunks))[i] if i < len(parse_result.get("text_page_nums", [])) else 0, i, m_id, now),
                            )
                    total_chunks += len(text_chunks)
                    print(f"[Worker] doc_id={doc_id}: {len(text_chunks)} 个文本 chunks 已入库")

                # 5. 图片 embedding + 存入 Milvus
                if image_items:
                    image_paths = [img["path"] for img in image_items]
                    image_page_nums = [img.get("page_num", 0) for img in image_items]
                    image_embeddings = self.image_embedder.encode(image_paths)
                    image_ids = self.milvus.insert_image_chunks(
                        kb_id=kb_id,
                        doc_id=doc_id,
                        image_paths=image_paths,
                        embeddings=image_embeddings,
                        page_nums=image_page_nums,
                    )
                    now = _now()
                    with get_db() as conn:
                        for i, (img, m_id) in enumerate(zip(image_items, image_ids)):
                            conn.execute(
                                """INSERT INTO chunks (doc_id, kb_id, chunk_type, file_path, page_num, chunk_index, milvus_id, created_at)
                                   VALUES (?, ?, 'image', ?, ?, ?, ?, ?)""",
                                (doc_id, kb_id, img["path"], img.get("page_num", 0), i, m_id, now),
                            )
                    total_chunks += len(image_items)
                    print(f"[Worker] doc_id={doc_id}: {len(image_items)} 张图片已入库")

                # 6. 标记完成
                self._update_status(doc_id, "completed", page_count=page_count)
                consumer.commit()
                print(f"[Worker] doc_id={doc_id}: 处理完成，共 {total_chunks} 个 chunks，{page_count} 页")

            except Exception as e:
                traceback.print_exc()
                self._update_status(doc_id, "failed", error_msg=str(e))
                print(f"[Worker] doc_id={doc_id}: 处理失败 → {e}")
                try:
                    consumer.commit()
                except Exception:
                    pass

    def _update_status(self, doc_id: int, status: str, page_count: int = 0, error_msg: str = ""):
        now = _now()
        with get_db() as conn:
            conn.execute(
                "UPDATE documents SET status=?, page_count=?, error_msg=?, updated_at=? WHERE id=?",
                (status, page_count, error_msg, now, doc_id),
            )


if __name__ == "__main__":
    from models import init_db
    init_db()
    worker = OfflineProcessWorker()
    worker.run()