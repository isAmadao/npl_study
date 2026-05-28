"""Kafka consumer — offline PDF parse → chunk → embed → Milvus index pipeline."""
import json
import logging
import os
import re

from kafka import KafkaConsumer
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.milvus import get_image_collection, get_text_collection, init_collections
from app.db.session import SessionLocal
from app.models.document import Chunk, Document
from app.services.chunker import chunk_markdown
from app.services.embedder import embed_images, embed_texts
from app.services.parser import parse_pdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
EMBED_BATCH_SIZE = 32   # max items per embedding call


# ── helpers ───────────────────────────────────────────────────────────────────

def _batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _extract_page_from_image_name(path: str) -> int:
    """MinerU names images like page_1_img_0.png — extract page number."""
    m = re.search(r"page[_-](\d+)", os.path.basename(path), re.IGNORECASE)
    return int(m.group(1)) if m else 1


def _cleanup_existing(doc_id: int, db: Session) -> None:
    """
    Remove any previous index data for this doc so re-processing is idempotent.
    Deletes from Milvus first (by doc_id expr), then from SQLite.
    """
    existing_chunks = db.query(Chunk).filter(Chunk.doc_id == doc_id).all()
    if not existing_chunks:
        return

    log.info("[doc=%d] Cleaning up %d existing chunks before re-index", doc_id, len(existing_chunks))

    try:
        col = get_text_collection()
        col.delete(f"doc_id == {doc_id}")
        col.flush()
    except Exception:
        log.warning("[doc=%d] Could not delete text vectors (Milvus may be empty)", doc_id, exc_info=True)

    try:
        col = get_image_collection()
        col.delete(f"doc_id == {doc_id}")
        col.flush()
    except Exception:
        log.warning("[doc=%d] Could not delete image vectors", doc_id, exc_info=True)

    db.query(Chunk).filter(Chunk.doc_id == doc_id).delete()
    db.commit()


# ── text indexing ─────────────────────────────────────────────────────────────

def _index_text(doc_id: int, original_name: str, md_path: str, db: Session) -> int:
    chunks = chunk_markdown(md_path)
    if not chunks:
        log.info("[doc=%d] No text chunks produced", doc_id)
        return 0

    log.info("[doc=%d] Indexing %d text chunks (batch=%d)", doc_id, len(chunks), EMBED_BATCH_SIZE)
    col = get_text_collection()
    total = 0

    for batch in _batched(chunks, EMBED_BATCH_SIZE):
        embeddings = embed_texts([c["content"] for c in batch])

        result = col.insert({
            "doc_id":      [doc_id] * len(batch),
            "filename":    [original_name] * len(batch),
            "page_num":    [c["page_num"] for c in batch],
            "chunk_index": [c["chunk_index"] for c in batch],
            "content":     [c["content"] for c in batch],
            "embedding":   embeddings,
        })
        col.flush()

        for chunk, milvus_id in zip(batch, result.primary_keys):
            db.add(Chunk(
                doc_id=doc_id,
                chunk_type="text",
                page_num=chunk["page_num"],
                chunk_index=chunk["chunk_index"],
                content=chunk["content"],
                milvus_id=str(milvus_id),
            ))
        total += len(batch)

    log.info("[doc=%d] Text indexing complete (%d chunks)", doc_id, total)
    return total


# ── image indexing ────────────────────────────────────────────────────────────

def _index_images(doc_id: int, original_name: str, img_dir: str, db: Session) -> int:
    if not os.path.isdir(img_dir):
        return 0

    image_paths = sorted(
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in _IMAGE_EXTS
    )
    if not image_paths:
        return 0

    log.info("[doc=%d] Indexing %d images (batch=%d)", doc_id, len(image_paths), EMBED_BATCH_SIZE)
    col = get_image_collection()
    total = 0

    for batch_paths in _batched(image_paths, EMBED_BATCH_SIZE):
        try:
            embeddings = embed_images(batch_paths)
        except Exception:
            log.warning("[doc=%d] Image embedding failed for batch, skipping", doc_id, exc_info=True)
            continue

        page_nums = [_extract_page_from_image_name(p) for p in batch_paths]
        result = col.insert({
            "doc_id":     [doc_id] * len(batch_paths),
            "filename":   [original_name] * len(batch_paths),
            "page_num":   page_nums,
            "image_path": batch_paths,
            "embedding":  embeddings,
        })
        col.flush()

        for img_path, page_num, milvus_id in zip(batch_paths, page_nums, result.primary_keys):
            db.add(Chunk(
                doc_id=doc_id,
                chunk_type="image",
                page_num=page_num,
                chunk_index=0,
                file_path=img_path,
                milvus_id=str(milvus_id),
            ))
        total += len(batch_paths)

    log.info("[doc=%d] Image indexing complete (%d images)", doc_id, total)
    return total


# ── main processing entry ─────────────────────────────────────────────────────

def process_document(doc_id: int, file_path: str) -> None:
    db: Session = SessionLocal()
    try:
        doc = db.get(Document, doc_id)
        if not doc:
            log.error("[doc=%d] Not found in DB — skipping", doc_id)
            return

        # Idempotency: wipe any previous index data for this doc
        _cleanup_existing(doc_id, db)

        doc.status = "processing"
        db.commit()

        # 1. Parse PDF → markdown + images
        parsed = parse_pdf(doc_id, file_path)

        # 2 & 3. Embed and index (batched)
        n_text = _index_text(doc_id, doc.original_name, parsed["markdown_path"], db)
        n_img = _index_images(doc_id, doc.original_name, parsed["image_dir"], db)

        doc.status = "done"
        db.commit()
        log.info("[doc=%d] Done — %d text chunks, %d images", doc_id, n_text, n_img)

    except Exception:
        log.exception("[doc=%d] Processing failed", doc_id)
        db.rollback()
        doc = db.get(Document, doc_id)
        if doc:
            doc.status = "error"
            db.commit()
    finally:
        db.close()


# ── Kafka consumer loop ───────────────────────────────────────────────────────

def main() -> None:
    init_collections()
    consumer = KafkaConsumer(
        settings.kafka_topic_parse,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_deserializer=lambda v: json.loads(v.decode()),
        auto_offset_reset="earliest",
        group_id="document_worker",
        # Don't auto-commit offset until processing is complete
        enable_auto_commit=False,
    )
    log.info("Worker started, listening on topic '%s'", settings.kafka_topic_parse)

    for message in consumer:
        payload = message.value
        log.info("Received: %s", payload)
        process_document(payload["doc_id"], payload["file_path"])
        # Commit offset only after successful (or gracefully failed) processing
        consumer.commit()


if __name__ == "__main__":
    main()
