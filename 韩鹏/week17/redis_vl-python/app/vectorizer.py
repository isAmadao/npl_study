"""SentenceTransformer wrapper with local-cache-first loading."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from app.config import get_settings


class Vectorizer:
    """Vectorizer wrapper that avoids runtime model downloads."""

    def __init__(self, model_name: str | None = None, cache_dir: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.hf_model_name
        self.cache_dir = cache_dir or settings.hf_cache_dir
        self._model = None

    def available_cached_models(self) -> list[str]:
        """Return discoverable model names from local HuggingFace cache."""
        cache_path = Path(self.cache_dir)
        if not cache_path.exists():
            return []

        models: list[str] = []
        for item in cache_path.iterdir():
            if not item.is_dir():
                continue

            # HuggingFace cache format: models--org--name
            if item.name.startswith("models--"):
                models.append(item.name.replace("models--", "", 1).replace("--", "/"))

        return sorted(set(models))

    def _load_model(self):
        if self._model is not None:
            return self._model

        cache_path = Path(self.cache_dir)
        if not cache_path.exists():
            raise RuntimeError(
                f"HuggingFace cache directory does not exist: {self.cache_dir}. "
                "Please set HF_CACHE_DIR correctly."
            )

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - import error depends on runtime.
            raise RuntimeError(
                "Failed to import sentence-transformers. Please install requirements first."
            ) from exc

        try:
            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir,
                local_files_only=True,
            )
        except Exception as exc:
            model_list = self.available_cached_models()
            if model_list:
                available_message = ", ".join(model_list)
            else:
                available_message = "No cached models discovered."
            raise RuntimeError(
                f"Unable to load model '{self.model_name}' from local cache '{self.cache_dir}'. "
                "Automatic download is disabled. "
                f"Available cached models: {available_message}"
            ) from exc

        return self._model

    def embed(self, text: str) -> list[float]:
        """Embed a single text and return a float list."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string.")

        model = self._load_model()
        vector = model.encode(text, convert_to_numpy=True)
        return [float(value) for value in vector.tolist()]

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        text_list = list(texts)
        if not text_list:
            return []

        if any((not isinstance(text, str) or not text.strip()) for text in text_list):
            raise ValueError("All texts in batch must be non-empty strings.")

        model = self._load_model()
        vectors = model.encode(text_list, convert_to_numpy=True)
        return [[float(value) for value in vector.tolist()] for vector in vectors]

