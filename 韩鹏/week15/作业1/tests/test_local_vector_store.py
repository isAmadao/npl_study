from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_mvp.services.vector_store import LocalVectorStore
from rag_mvp.types import VectorRecord


class LocalVectorStoreTestCase(unittest.TestCase):
    def test_upsert_search_and_delete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalVectorStore(Path(temp_dir) / "store.json")
            store.upsert_records(
                [
                    VectorRecord(
                        record_id="text-1",
                        file_id=1,
                        file_name="a.docx",
                        source_path="a.docx",
                        chunk_type="text",
                        page_no=1,
                        content="文档A",
                        image_path=None,
                        embedding=[1.0, 0.0, 0.0],
                    ),
                    VectorRecord(
                        record_id="text-2",
                        file_id=2,
                        file_name="b.docx",
                        source_path="b.docx",
                        chunk_type="text",
                        page_no=1,
                        content="文档B",
                        image_path=None,
                        embedding=[0.0, 1.0, 0.0],
                    ),
                ]
            )

            results = store.search([1.0, 0.0, 0.0], chunk_type="text", top_k=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].file_id, 1)
            self.assertTrue(store.has_chunk_type("text"))
            self.assertFalse(store.has_chunk_type("image"))

            store.delete_by_file_id(1)
            remaining = store.search([1.0, 0.0, 0.0], chunk_type="text", top_k=5)
            self.assertTrue(all(item.file_id != 1 for item in remaining))


if __name__ == "__main__":
    unittest.main()
