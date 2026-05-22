from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from rag_mvp.config import settings
from rag_mvp.services.local_jina_clip import LocalJinaClipModel
from rag_mvp.services.model_resolver import ModelResolver


def _normalize(vectors) -> list[list[float]]:
    array = np.asarray(vectors, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (array / norms).tolist()


class TextEmbeddingService:
    def __init__(
        self,
        model_name_or_path: str | Path | None = None,
        resolver: ModelResolver | None = None,
    ) -> None:
        self.model_name_or_path = (
            model_name_or_path or settings.text_model_name_or_path
        )
        self.resolver = resolver or ModelResolver()
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            resolved_model_dir = self.resolver.resolve_text_model(
                self.model_name_or_path
            )
            self._model = SentenceTransformer(str(resolved_model_dir))
        return self._model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load_model()
        vectors = model.encode(list(texts), normalize_embeddings=True)
        return _normalize(vectors)


class MultimodalEmbeddingService:
    def __init__(
        self,
        model_name_or_path: str | Path | None = None,
        text_fallback: TextEmbeddingService | None = None,
        resolver: ModelResolver | None = None,
    ) -> None:
        self.model_name_or_path = (
            model_name_or_path or settings.multimodal_model_name_or_path
        )
        self.resolver = resolver or ModelResolver()
        self.text_fallback = text_fallback or TextEmbeddingService(
            resolver=self.resolver
        )
        self._model = None
        self._load_attempted = False
        self._load_error: RuntimeError | None = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        if self._load_attempted:
            if self._load_error is not None:
                raise self._load_error
            raise RuntimeError("Multimodal model is unavailable.")

        self._load_attempted = True
        try:
            self._model = LocalJinaClipModel(
                model_name_or_path=self.model_name_or_path,
                resolver=self.resolver,
            )
            return self._model
        except Exception as exc:
            self._load_error = RuntimeError(
                "jina-clip-v2 failed to load in the current environment; fallback text embeddings will be used."
            )
            self._load_error.__cause__ = exc
            raise self._load_error

    def encode_text(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            model = self._load_model()
            vectors = model.encode_text(list(texts))
            return _normalize(vectors)
        except Exception:
            return self.text_fallback.encode(list(texts))

    def encode_images(
        self,
        image_paths: Sequence[str],
        fallback_texts: Sequence[str] | None = None,
    ) -> list[list[float]]:
        if not image_paths:
            return []
        try:
            model = self._load_model()
            vectors = model.encode_images(list(image_paths))
            return _normalize(vectors)
        except Exception:
            descriptions = list(fallback_texts or [])
            if len(descriptions) != len(image_paths):
                descriptions = [Path(path).stem for path in image_paths]
            return self.text_fallback.encode(descriptions)
