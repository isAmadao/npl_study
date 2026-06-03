"""Embeddings cache implementation backed by Redis."""

from __future__ import annotations

import hashlib
from typing import Iterable

from redis import Redis

from app.config import get_settings


class EmbeddingsCache:
    """Exact hash cache for text embeddings."""

    def __init__(self, redis_client: Redis, name: str = "embeddings", ttl: int | None = None) -> None:
        self.redis_client = redis_client
        self.name = name
        self.ttl = ttl
        self.model_name = get_settings().hf_model_name

    def _key_for_text(self, text: str) -> str:
        raw = f"{text}|{self.model_name}".encode("utf-8")
        md5_hash = hashlib.md5(raw).hexdigest()
        return f"embedcache:{self.name}:{md5_hash}"

    @staticmethod
    def _serialize_vector(vector: list[float]) -> str:
        return ",".join(format(float(value), ".17g") for value in vector)

    @staticmethod
    def _deserialize_vector(raw_value: str | None) -> list[float] | None:
        if raw_value is None:
            return None
        if raw_value == "":
            return []
        return [float(item) for item in raw_value.split(",")]

    def set(self, texts: Iterable[str], embeddings: Iterable[list[float]]) -> None:
        text_list = list(texts)
        embedding_list = list(embeddings)
        if len(text_list) != len(embedding_list):
            raise ValueError("texts and embeddings must have the same length.")

        with self.redis_client.pipeline(transaction=False) as pipeline:
            for text, vector in zip(text_list, embedding_list):
                key = self._key_for_text(text)
                serialized = self._serialize_vector(vector)
                if self.ttl is None:
                    pipeline.set(key, serialized)
                else:
                    pipeline.set(key, serialized, ex=self.ttl)
            pipeline.execute()

    def get(self, texts: Iterable[str]) -> dict[str, list[float] | None]:
        text_list = list(texts)
        keys = [self._key_for_text(text) for text in text_list]
        raw_values = self.redis_client.mget(keys)
        return {
            text: self._deserialize_vector(raw_value)
            for text, raw_value in zip(text_list, raw_values)
        }

    def exists(self, text: str) -> bool:
        return bool(self.redis_client.exists(self._key_for_text(text)))
