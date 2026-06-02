"""Semantic router backed by Redis + RediSearch."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

import numpy as np
from redis import Redis
from redis.exceptions import ResponseError

from app.config import get_settings


class SemanticRouter:
    """Route prompts to pre-defined branches by semantic similarity."""

    def __init__(self, redis_client: Redis, vectorizer: Any, name: str = "router") -> None:
        self.redis_client = redis_client
        self.vectorizer = vectorizer
        self.name = name
        self.embedding_dim = get_settings().embedding_dim
        self.index_name = f"idx:router:{self.name}"
        self.key_prefix = f"router:{self.name}:"
        self.meta_key = f"routermeta:{self.name}:version"
        self._ensure_index()

    @staticmethod
    def _to_vector_blob(vector: list[float]) -> bytes:
        return np.asarray(vector, dtype=np.float32).tobytes()

    @staticmethod
    def _decode(value: Any) -> Any:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return value

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

    def _ensure_index(self) -> None:
        try:
            self.redis_client.execute_command("FT.INFO", self.index_name)
            return
        except ResponseError as exc:
            message = str(exc).lower()
            if "unknown index" not in message and "unknown index name" not in message:
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
            "route_name",
            "TAG",
            "reference",
            "TEXT",
            "vector",
            "VECTOR",
            "FLAT",
            "6",
            "TYPE",
            "FLOAT32",
            "DIM",
            str(self.embedding_dim),
            "DISTANCE_METRIC",
            "COSINE",
            "threshold",
            "NUMERIC",
            "SORTABLE",
        )

    def has_routes(self) -> bool:
        return next(self.redis_client.scan_iter(match=f"{self.key_prefix}*"), None) is not None

    def add_routes(self, routes: Iterable[dict[str, Any]]) -> int:
        route_list = list(routes)
        existing_keys = list(self.redis_client.scan_iter(match=f"{self.key_prefix}*"))
        if existing_keys:
            self.redis_client.delete(*existing_keys)

        if not route_list:
            return 0

        inserted = 0
        with self.redis_client.pipeline(transaction=False) as pipeline:
            for route in route_list:
                route_name = str(route.get("name", "")).strip()
                threshold = float(route.get("distance_threshold", 0.3))
                references_raw = route.get("references") or []
                references = [str(reference).strip() for reference in references_raw if str(reference).strip()]

                if not route_name:
                    raise ValueError("Route name must not be empty.")
                if not references:
                    raise ValueError(f"Route '{route_name}' must contain at least one reference.")

                for ref_index, reference in enumerate(references):
                    vector = self.vectorizer.embed(reference)
                    vector_blob = self._to_vector_blob(vector)
                    key = f"{self.key_prefix}{route_name}:{ref_index}"
                    pipeline.hset(
                        key,
                        mapping={
                            "route_name": route_name,
                            "reference": reference,
                            "vector": vector_blob,
                            "vector_hex": vector_blob.hex(),
                            "threshold": str(threshold),
                        },
                    )
                    inserted += 1

            pipeline.execute()

        return inserted

    def sync_routes(self, routes: Iterable[dict[str, Any]]) -> bool:
        route_list = list(routes)
        payload = json.dumps(route_list, ensure_ascii=False, sort_keys=True)
        digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
        stored_digest = self.redis_client.get(self.meta_key)
        if stored_digest == digest and self.has_routes():
            return False

        self.add_routes(route_list)
        self.redis_client.set(self.meta_key, digest)
        return True

    def route(self, query: str, query_vector: list[float] | None = None) -> dict[str, Any] | None:
        if not isinstance(query, str) or not query.strip():
            return None

        vector = query_vector if query_vector is not None else self.vectorizer.embed(query)
        vector_blob = self._to_vector_blob(vector)
        result = self.redis_client.execute_command(
            "FT.SEARCH",
            self.index_name,
            "*=>[KNN 1 @vector $vec AS dist]",
            "SORTBY",
            "dist",
            "RETURN",
            "4",
            "route_name",
            "reference",
            "threshold",
            "dist",
            "LIMIT",
            "0",
            "1",
            "PARAMS",
            "2",
            "vec",
            vector_blob,
            "DIALECT",
            "2",
        )

        parsed = self._parse_search_result(result)
        if parsed is None:
            return None

        distance = float(parsed.get("distance", float("inf")))
        threshold = float(parsed.get("threshold", 0.0))
        if distance > threshold:
            return None

        return {
            "matched": True,
            "route_name": str(parsed.get("route_name", "")),
            "distance": distance,
            "matched_reference": str(parsed.get("reference", "")),
            "threshold": threshold,
        }

    def _parse_search_result(self, result: Any) -> dict[str, Any] | None:
        if not result:
            return None

        if isinstance(result, dict):
            total = int(result.get("total_results", 0))
            if total == 0:
                return None

            first = (result.get("results") or [None])[0]
            if not first:
                return None

            attrs = first.get("extra_attributes") or first.get("attributes") or {}
            return {
                "key": first.get("id", ""),
                "route_name": self._decode(attrs.get("route_name", "")),
                "reference": self._decode(attrs.get("reference", "")),
                "threshold": attrs.get("threshold", "0"),
                "distance": attrs.get("dist", "inf"),
            }

        if isinstance(result, list):
            if int(result[0]) == 0:
                return None

            fields = self._pair_list_to_dict(result[2])
            return {
                "key": result[1],
                "route_name": self._decode(fields.get("route_name", "")),
                "reference": self._decode(fields.get("reference", "")),
                "threshold": fields.get("threshold", "0"),
                "distance": fields.get("dist", "inf"),
            }

        return None
