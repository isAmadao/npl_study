from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_mvp.services.embedding_service import (
    MultimodalEmbeddingService,
    TextEmbeddingService,
)


class FakeTextFallback:
    def encode(self, texts):
        return [[float(index + 1), 0.0] for index, _ in enumerate(texts)]


class MultimodalEmbeddingServiceTestCase(unittest.TestCase):
    def test_text_service_resolves_model_before_loading(self):
        resolver = Mock()
        resolver.resolve_text_model.return_value = ROOT_DIR / "cached-text-model"

        class FakeSentenceTransformer:
            def __init__(self, model_path: str):
                self.model_path = model_path

            def encode(self, texts, normalize_embeddings: bool = True):
                del normalize_embeddings
                return [[1.0, 0.0] for _ in texts]

        fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
        service = TextEmbeddingService(
            model_name_or_path="BAAI/bge-small-zh-v1.5",
            resolver=resolver,
        )

        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            vectors = service.encode(["娴嬭瘯"])

        self.assertEqual(vectors, [[1.0, 0.0]])
        resolver.resolve_text_model.assert_called_once_with("BAAI/bge-small-zh-v1.5")

    def test_encode_text_falls_back_when_model_load_fails(self):
        service = MultimodalEmbeddingService(text_fallback=FakeTextFallback())

        def fail_loader():
            raise RuntimeError("boom")

        service._load_model = fail_loader  # type: ignore[method-assign]
        vectors = service.encode_text(["你好", "世界"])
        self.assertEqual(vectors, [[1.0, 0.0], [2.0, 0.0]])

    def test_encode_images_falls_back_to_descriptions(self):
        service = MultimodalEmbeddingService(text_fallback=FakeTextFallback())

        def fail_loader():
            raise RuntimeError("boom")

        service._load_model = fail_loader  # type: ignore[method-assign]
        vectors = service.encode_images(
            ["a.png", "b.png"],
            fallback_texts=["图片A", "图片B"],
        )
        self.assertEqual(vectors, [[1.0, 0.0], [2.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
