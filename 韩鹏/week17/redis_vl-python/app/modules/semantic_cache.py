"""Semantic cache backed by Redis + RediSearch vector search."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import numpy as np
from redis import Redis
from redis.exceptions import ResponseError

from app.config import get_settings


class SemanticCache:
    """Cache prompt-response pairs using vector similarity search."""

    def __init__(
        self,
        redis_client: Redis,
        vectorizer: Any,
        name: str = "semcache",
        distance_threshold: float = 0.2,
    ) -> None:
        self.redis_client = redis_client
        self.vectorizer = vectorizer
        self.name = name
        self.distance_threshold = distance_threshold
        self.embedding_dim = get_settings().embedding_dim
        self.index_name = f"idx:semcache:{self.name}"
        self.key_prefix = f"semcache:{self.name}:"
        self._ensure_index()

    @staticmethod
    def _to_vector_blob(vector: list[float]) -> bytes:
        return np.asarray(vector, dtype=np.float32).tobytes()

    def _ensure_index(self) -> None:
        try:
            self.redis_client.execute_command("FT.INFO", self.index_name)
            return
        except ResponseError as exc:
            if "unknown index name" not in str(exc).lower():
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

    def store(
        self,
        prompt: str,
        response: str,
        metadata: dict[str, Any] | None = None,
        vector: list[float] | None = None,
    ) -> str:
        vector = vector if vector is not None else self.vectorizer.embed(prompt)
        vector_blob = self._to_vector_blob(vector)
        key = f"{self.key_prefix}{uuid.uuid4().hex}"

        payload: dict[str, Any] = {
            "prompt": prompt,
            "response": response,
            "prompt_vector": vector_blob,
            "prompt_vector_hex": vector_blob.hex(),
            "created_at": str(time.time()),
        }
        if metadata is not None:
            payload["metadata"] = json.dumps(metadata, ensure_ascii=False)

        self.redis_client.hset(key, mapping=payload)
        return key

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

    def check(self, prompt: str, query_vector: list[float] | None = None) -> dict[str, Any] | None:
        query_vector = query_vector if query_vector is not None else self.vectorizer.embed(prompt)
        query_blob = self._to_vector_blob(query_vector)

        result = self.redis_client.execute_command(
            "FT.SEARCH",
            self.index_name,
            "*=>[KNN 1 @prompt_vector $vec AS dist]",
            "SORTBY",
            "dist",
            "RETURN",
            "3",
            "prompt",
            "response",
            "dist",
            "LIMIT",
            "0",
            "1",
            "PARAMS",
            "2",
            "vec",
            query_blob,
            "DIALECT",
            "2",
        )

        parsed = self._parse_search_result(result)
        if parsed is None:
            return None

        distance = float(parsed.get("distance", float("inf")))
        if distance > self.distance_threshold:
            return None

        cached_prompt = parsed.get("cached_prompt", "")
        cached_response = parsed.get("response", "")
        if isinstance(cached_prompt, bytes):
            cached_prompt = cached_prompt.decode("utf-8", errors="ignore")
        if isinstance(cached_response, bytes):
            cached_response = cached_response.decode("utf-8", errors="ignore")

        return {
            "key": parsed.get("key", ""),
            "cached_prompt": str(cached_prompt),
            "response": str(cached_response),
            "distance": distance,
        }

    def _parse_search_result(self, result: Any) -> dict[str, Any] | None:
        if not result:
            return None

        # RESP3 format (dict) in newer redis-py/Redis Stack combos.
        if isinstance(result, dict):
            total = int(result.get("total_results", 0))
            if total == 0:
                return None
            first = (result.get("results") or [None])[0]
            if not first:
                return None
            attrs = first.get("extra_attributes") or {}
            return {
                "key": first.get("id", ""),
                "cached_prompt": attrs.get("prompt", ""),
                "response": attrs.get("response", ""),
                "distance": attrs.get("dist", "inf"),
            }

        # RESP2 format (list) fallback.
        if isinstance(result, list):
            if int(result[0]) == 0:
                return None
            fields = self._pair_list_to_dict(result[2])
            return {
                "key": result[1],
                "cached_prompt": fields.get("prompt", ""),
                "response": fields.get("response", ""),
                "distance": fields.get("dist", "inf"),
            }

        return None

    def clear(self) -> int:
        keys = list(self.redis_client.scan_iter(match=f"{self.key_prefix}*"))
        if not keys:
            return 0
        return int(self.redis_client.delete(*keys))
