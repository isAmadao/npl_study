"""Semantic message history backed by Redis + RediSearch."""

from __future__ import annotations

import time
import uuid
from typing import Any

import numpy as np
from redis import Redis
from redis.exceptions import ResponseError

from app.config import get_settings


class SemanticMessageHistory:
    """Store and retrieve conversation turns per session."""

    def __init__(
        self,
        redis_client: Redis,
        vectorizer: Any,
        name: str = "msghist",
        session_tag: str = "default",
        distance_threshold: float = 0.5,
    ) -> None:
        self.redis_client = redis_client
        self.vectorizer = vectorizer
        self.name = name
        self.session_tag = session_tag or "default"
        self.distance_threshold = distance_threshold
        self.embedding_dim = get_settings().embedding_dim
        self.index_name = f"idx:msghist:{self.name}:{self.session_tag}"
        self.key_prefix = f"msghist:{self.name}:{self.session_tag}:"
        self._ensure_index()

    @staticmethod
    def _to_vector_blob(vector: list[float]) -> bytes:
        return np.asarray(vector, dtype=np.float32).tobytes()

    @staticmethod
    def _decode(value: Any) -> Any:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return value

    def _ensure_index(self) -> None:
        try:
            self.redis_client.execute_command("FT.INFO", self.index_name)
            return
        except ResponseError as exc:
            message = str(exc).lower()
            if "unknown index" not in message:
                raise

        self.redis_client.execute_command(
            "FT.CREATE",
            self.index_name,
            "ON",
            "HASH",
            "PREFIX",
            "1",
            self.key_prefix,
            "SCHEMA",
            "prompt",
            "TEXT",
            "response",
            "TEXT",
            "prompt_vector",
            "VECTOR",
            "FLAT",
            "6",
            "TYPE",
            "FLOAT32",
            "DIM",
            str(self.embedding_dim),
            "DISTANCE_METRIC",
            "COSINE",
            "created_at",
            "NUMERIC",
            "SORTABLE",
        )

    def store(self, prompt: str, response: str, vector: list[float] | None = None) -> str:
        vector = vector if vector is not None else self.vectorizer.embed(prompt)
        vector_blob = self._to_vector_blob(vector)
        key = f"{self.key_prefix}{uuid.uuid4().hex}"

        self.redis_client.hset(
            key,
            mapping={
                "prompt": prompt,
                "response": response,
                "prompt_vector": vector_blob,
                "prompt_vector_hex": vector_blob.hex(),
                "created_at": str(time.time()),
            },
        )
        return key

    def get_recent(self, top_k: int = 5) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []

        result = self.redis_client.execute_command(
            "FT.SEARCH",
            self.index_name,
            "*",
            "SORTBY",
            "created_at",
            "DESC",
            "LIMIT",
            "0",
            str(top_k),
            "RETURN",
            "3",
            "prompt",
            "response",
            "created_at",
            "DIALECT",
            "2",
        )
        return self._parse_rows(result, with_distance=False)

    def get_relevant(
        self,
        query: str,
        top_k: int = 5,
        query_vector: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []

        vector = query_vector if query_vector is not None else self.vectorizer.embed(query)
        vector_blob = self._to_vector_blob(vector)
        result = self.redis_client.execute_command(
            "FT.SEARCH",
            self.index_name,
            f"*=>[KNN {top_k} @prompt_vector $vec AS dist]",
            "SORTBY",
            "dist",
            "RETURN",
            "4",
            "prompt",
            "response",
            "created_at",
            "dist",
            "LIMIT",
            "0",
            str(top_k),
            "PARAMS",
            "2",
            "vec",
            vector_blob,
            "DIALECT",
            "2",
        )

        rows = self._parse_rows(result, with_distance=True)
        return [row for row in rows if float(row["distance"]) <= self.distance_threshold]

    def clear(self) -> int:
        keys = list(self.redis_client.scan_iter(match=f"{self.key_prefix}*"))
        if not keys:
            return 0
        return int(self.redis_client.delete(*keys))

    def _parse_rows(self, result: Any, *, with_distance: bool) -> list[dict[str, Any]]:
        if not result:
            return []

        rows: list[dict[str, Any]] = []
        if isinstance(result, dict):
            for item in result.get("results") or []:
                attrs = item.get("extra_attributes") or item.get("attributes") or {}
                row = {
                    "prompt": self._decode(attrs.get("prompt", "")),
                    "response": self._decode(attrs.get("response", "")),
                    "created_at": str(self._decode(attrs.get("created_at", ""))),
                }
                if with_distance:
                    row["distance"] = float(attrs.get("dist", "inf"))
                rows.append(row)
            return rows

        if isinstance(result, list):
            for offset in range(1, len(result), 2):
                fields = self._pair_list_to_dict(result[offset + 1])
                row = {
                    "prompt": self._decode(fields.get("prompt", "")),
                    "response": self._decode(fields.get("response", "")),
                    "created_at": str(self._decode(fields.get("created_at", ""))),
                }
                if with_distance:
                    row["distance"] = float(fields.get("dist", "inf"))
                rows.append(row)
            return rows

        return []

    @staticmethod
    def _pair_list_to_dict(values: list[Any]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        iterator = iter(values)
        for key in iterator:
            value = next(iterator, None)
            if isinstance(key, bytes):
                key = key.decode("utf-8", errors="ignore")
            output[str(key)] = value
        return output
