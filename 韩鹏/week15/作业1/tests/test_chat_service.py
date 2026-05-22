from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_mvp.services.chat_service import ChatService
from rag_mvp.types import SearchResult


class FakeTextEmbedder:
    def encode(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FailIfCalledMultimodal:
    def encode_text(self, texts):
        raise AssertionError(f"multimodal encode_text should not be called: {texts}")


class FakeVectorStore:
    def has_chunk_type(self, chunk_type: str) -> bool:
        return chunk_type == "text"

    def search(self, query_vector, chunk_type: str, top_k: int = 4):
        del query_vector, top_k
        if chunk_type == "text":
            return [
                SearchResult(
                    record_id="text-1",
                    file_id=1,
                    file_name="doc.docx",
                    source_path="doc.docx",
                    chunk_type="text",
                    page_no=1,
                    content="测试片段",
                    image_path=None,
                    score=0.9,
                )
            ]
        return []


class FakeLLM:
    def ask(self, question: str, context: str, history: list[dict]) -> str:
        del question, context, history
        return "ok"


class ChatServiceTestCase(unittest.TestCase):
    def test_skip_multimodal_when_no_image_index_exists(self):
        service = ChatService()
        service.text_embedder = FakeTextEmbedder()
        service.multimodal_embedder = FailIfCalledMultimodal()
        service.vector_store = FakeVectorStore()
        service.llm = FakeLLM()

        reply = service.ask("你好", [])
        self.assertEqual(reply.answer, "ok")
        self.assertEqual(len(reply.references), 1)


if __name__ == "__main__":
    unittest.main()
