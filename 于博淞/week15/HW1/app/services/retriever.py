"""Fused text + image retrieval from Milvus, scoped to a set of doc_ids."""
from app.db.milvus import get_image_collection, get_text_collection
from app.services.embedder import embed_query_for_images, embed_texts

_SEARCH_PARAMS = {"metric_type": "COSINE", "params": {"nprobe": 16}}


def retrieve(query: str, doc_ids: list[int], top_k: int = 5) -> list[dict]:
    """
    Embed the query with BGE (text) and CLIP (image→text), search both Milvus
    collections filtered to doc_ids, merge by score, and return up to top_k chunks.

    doc_ids must be the caller's responsibility — chat endpoint fetches the list
    of done doc_ids for the requested knowledge base from SQLite before calling here.
    """
    if not doc_ids:
        return []

    expr = f"doc_id in [{', '.join(map(str, doc_ids))}]"

    text_hits = _search_text(query, expr, top_k)
    image_hits = _search_images(query, expr, top_k)

    seen: set[tuple] = set()
    merged: list[dict] = []
    for hit in sorted(text_hits + image_hits, key=lambda h: h["score"], reverse=True):
        key = (hit["filename"], hit["page_num"], hit["chunk_type"])
        if key not in seen:
            seen.add(key)
            merged.append(hit)
        if len(merged) >= top_k:
            break

    return merged


def _search_text(query: str, expr: str, top_k: int) -> list[dict]:
    col = get_text_collection()
    embedding = embed_texts([query])[0]
    hits = col.search(
        [embedding],
        "embedding",
        _SEARCH_PARAMS,
        limit=top_k,
        expr=expr,
        output_fields=["doc_id", "filename", "page_num", "chunk_index", "content"],
    )
    return [
        {
            "chunk_type": "text",
            "doc_id": h.entity.get("doc_id"),
            "filename": h.entity.get("filename"),
            "page_num": h.entity.get("page_num"),
            "content": h.entity.get("content"),
            "score": h.score,
        }
        for h in hits[0]
    ]


def _search_images(query: str, expr: str, top_k: int) -> list[dict]:
    col = get_image_collection()
    embedding = embed_query_for_images(query)
    hits = col.search(
        [embedding],
        "embedding",
        _SEARCH_PARAMS,
        limit=top_k,
        expr=expr,
        output_fields=["doc_id", "filename", "page_num", "image_path"],
    )
    return [
        {
            "chunk_type": "image",
            "doc_id": h.entity.get("doc_id"),
            "filename": h.entity.get("filename"),
            "page_num": h.entity.get("page_num"),
            "image_path": h.entity.get("image_path"),
            "score": h.score,
        }
        for h in hits[0]
    ]
