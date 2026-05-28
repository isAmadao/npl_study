from __future__ import annotations

import base64
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_mvp.services.document_parser import DocumentParser


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9WHH0m4AAAAASUVORK5CYII="
)


class DocumentParserTestCase(unittest.TestCase):
    def test_parse_docx_extracts_text_and_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docx_path = root / "sample.docx"
            derived_dir = root / "derived"

            with zipfile.ZipFile(docx_path, "w") as archive:
                archive.writestr(
                    "word/document.xml",
                    (
                        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                        "<w:body>"
                        "<w:p><w:r><w:t>第一段内容</w:t></w:r></w:p>"
                        "<w:p><w:r><w:t>第二段内容</w:t></w:r></w:p>"
                        "</w:body>"
                        "</w:document>"
                    ),
                )
                archive.writestr("word/media/image1.png", PNG_BYTES)

            parsed = DocumentParser().parse(docx_path, derived_dir)
            self.assertEqual(len(parsed.text_pages), 1)
            self.assertIn("第一段内容", parsed.text_pages[0][1])
            self.assertEqual(len(parsed.images), 1)
            self.assertTrue(Path(parsed.images[0].image_path).exists())


if __name__ == "__main__":
    unittest.main()
