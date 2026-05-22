from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_mvp.utils import chunk_text, pad_embedding


class UtilsTestCase(unittest.TestCase):
    def test_chunk_text_creates_overlap(self):
        text = "a" * 20
        chunks = chunk_text(text, max_chars=8, overlap=2)
        self.assertEqual(chunks, ["aaaaaaaa", "aaaaaaaa", "aaaaaaaa"])

    def test_pad_embedding_expands_to_target_dim(self):
        padded = pad_embedding([1.0, 2.0], target_dim=5)
        self.assertEqual(padded, [1.0, 2.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
