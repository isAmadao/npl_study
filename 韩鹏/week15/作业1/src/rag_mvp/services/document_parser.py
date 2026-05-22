from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from rag_mvp.types import ParsedDocument, ParsedImage


class DocumentParser:
    def parse(self, file_path: Path, derived_dir: Path) -> ParsedDocument:
        suffix = file_path.suffix.lower()
        if suffix == ".docx":
            return self._parse_docx(file_path, derived_dir)
        if suffix == ".pdf":
            return self._parse_pdf(file_path, derived_dir)
        raise ValueError(f"暂不支持的文件类型：{suffix}")

    def _parse_docx(self, file_path: Path, derived_dir: Path) -> ParsedDocument:
        derived_dir.mkdir(parents=True, exist_ok=True)
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        text_pages: list[tuple[int, str]] = []
        images: list[ParsedImage] = []

        with zipfile.ZipFile(file_path) as archive:
            document_xml = archive.read("word/document.xml")
            root = ElementTree.fromstring(document_xml)
            paragraphs = []
            for paragraph in root.findall(".//w:p", namespace):
                text_nodes = [node.text for node in paragraph.findall(".//w:t", namespace) if node.text]
                line = "".join(text_nodes).strip()
                if line:
                    paragraphs.append(line)
            if paragraphs:
                text_pages.append((1, "\n".join(paragraphs)))

            for member_name in archive.namelist():
                if not member_name.startswith("word/media/"):
                    continue
                target_path = derived_dir / Path(member_name).name
                with archive.open(member_name) as source, target_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
                images.append(
                    ParsedImage(
                        image_path=str(target_path),
                        page_no=0,
                        description=f"图片资源，来源文件 {file_path.name}",
                    )
                )

        return ParsedDocument(text_pages=text_pages, images=images)

    def _parse_pdf(self, file_path: Path, derived_dir: Path) -> ParsedDocument:
        del derived_dir
        import pdfplumber

        text_pages: list[tuple[int, str]] = []
        with pdfplumber.open(file_path) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    text_pages.append((index, text))
        return ParsedDocument(text_pages=text_pages, images=[])

