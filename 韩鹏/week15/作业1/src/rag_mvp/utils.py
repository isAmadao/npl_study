from __future__ import annotations

import math


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split()).strip()


def chunk_text(text: str, max_chars: int = 500, overlap: int = 80) -> list[str]:
    clean_text = normalize_whitespace(text)
    if not clean_text:
        return []
    if max_chars <= 0:
        raise ValueError("max_chars 必须大于 0。")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap 必须在 [0, max_chars) 范围内。")

    chunks: list[str] = []
    start = 0
    while start < len(clean_text):
        end = min(len(clean_text), start + max_chars)
        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(clean_text):
            break
        start = end - overlap
    return chunks


def pad_embedding(vector: list[float], target_dim: int = 1024) -> list[float]:
    if len(vector) == target_dim:
        return vector
    if len(vector) > target_dim:
        return vector[:target_dim]
    return vector + [0.0] * (target_dim - len(vector))


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(vec1, vec2))
    denom1 = math.sqrt(sum(a * a for a in vec1))
    denom2 = math.sqrt(sum(b * b for b in vec2))
    if denom1 == 0 or denom2 == 0:
        return 0.0
    return numerator / (denom1 * denom2)

