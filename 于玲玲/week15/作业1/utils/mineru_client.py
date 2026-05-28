"""
MinerU PDF 解析客户端：通过 HTTP API 调用 MinerU 服务解析 PDF，
输出 markdown 文本和提取的图片文件。
支持 Mock 模式（无 MinerU 服务时用 PyMuPDF 做基础抽取）。
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import requests

import config
from utils.storage import ensure_kb_dirs


def parse_pdf_with_mineru(pdf_path: str, kb_id: int, doc_id: int) -> dict[str, Any]:
    """调用 MinerU 解析 PDF，返回 text_chunks、images、page_count 等。

    返回格式:
    {
        "text_chunks": ["chunk1 text", "chunk2 text", ...],
        "text_page_nums": [1, 1, 2, ...],
        "images": [{"path": "/abs/path/to/img.png", "page_num": 1}, ...],
        "page_count": 10,
    }
    """
    kb_dirs = ensure_kb_dirs(kb_id)

    try:
        return _call_mineru_api(pdf_path, kb_dirs)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        print(f"[MinerU] 服务不可用，降级为 PyMuPDF 基础解析: {pdf_path}")
        return _fallback_pymupdf_parse(pdf_path, kb_dirs)


def _call_mineru_api(pdf_path: str, kb_dirs: dict) -> dict[str, Any]:
    """通过 MinerU API 解析 PDF。"""
    url = f"{config.MINERU_API_URL}/parse"

    with open(pdf_path, "rb") as f:
        resp = requests.post(
            url,
            files={"file": (Path(pdf_path).name, f, "application/pdf")},
            timeout=300,  # MinerU 解析可能需要几分钟
        )
    resp.raise_for_status()
    result = resp.json()

    # MinerU 返回结构（示例）:
    # {
    #   "content": "markdown 全文",
    #   "images": [{"path": "/tmp/xxx/img_001.png", "page_num": 1}, ...],
    #   "pages": 10
    # }

    full_markdown = result.get("content", "")
    raw_images = result.get("images", [])
    page_count = result.get("pages", 0)

    # 将图片复制到知识库目录
    images = []
    markdown_dir = kb_dirs["markdown"]
    images_dir = kb_dirs["images"]

    for img_info in raw_images:
        src = img_info.get("path", "")
        if not src or not Path(src).exists():
            continue
        dst_name = f"doc_{Path(pdf_path).stem}_p{img_info.get('page_num', 0)}_{Path(src).name}"
        dst = images_dir / dst_name
        shutil.copy2(src, dst)
        images.append({"path": str(dst), "page_num": img_info.get("page_num", 0)})

    # 保存 markdown
    md_path = markdown_dir / f"{Path(pdf_path).stem}.md"
    md_path.write_text(full_markdown, encoding="utf-8")

    # 对 markdown 按段落分 chunk
    text_chunks = _split_markdown_into_chunks(full_markdown)
    # 简单的页面估算：将 chunks 均匀分配到各页面
    text_page_nums = _estimate_page_nums(text_chunks, page_count)

    return {
        "text_chunks": text_chunks,
        "text_page_nums": text_page_nums,
        "images": images,
        "page_count": page_count,
    }


def _fallback_pymupdf_parse(pdf_path: str, kb_dirs: dict) -> dict[str, Any]:
    """当 MinerU 不可用时，用 PyMuPDF 做基础文本抽取和图片提取。"""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    page_count = len(doc)
    text_chunks = []
    text_page_nums = []
    images = []

    markdown_dir = kb_dirs["markdown"]
    images_dir = kb_dirs["images"]
    stem = Path(pdf_path).stem

    md_lines = []

    for page_idx in range(page_count):
        page = doc[page_idx]
        page_num = page_idx + 1

        # 抽取文本
        text = page.get_text("text")
        if text.strip():
            text_chunks.append(text.strip())
            text_page_nums.append(page_num)
            md_lines.append(f"## 第 {page_num} 页\n\n{text.strip()}")

        # 抽取图片
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            ext = base_image["ext"]
            img_name = f"{stem}_p{page_num}_img{img_idx}.{ext}"
            img_path = images_dir / img_name
            img_path.write_bytes(img_bytes)
            images.append({"path": str(img_path), "page_num": page_num})

    # 保存 markdown
    md_path = markdown_dir / f"{stem}.md"
    md_path.write_text("\n\n".join(md_lines), encoding="utf-8")

    doc.close()
    return {
        "text_chunks": text_chunks,
        "text_page_nums": text_page_nums,
        "images": images,
        "page_count": page_count,
    }


def _split_markdown_into_chunks(markdown_text: str, chunk_size: int = 500) -> list[str]:
    """将 markdown 文本按段落切分为 chunks。"""
    paragraphs = [p.strip() for p in markdown_text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) < chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            current = para

    if current:
        chunks.append(current)

    return chunks if chunks else [markdown_text[:chunk_size]]


def _estimate_page_nums(chunks: list[str], page_count: int) -> list[int]:
    """粗略估算每个 chunk 属于第几页（按 chunk 数均匀分配）。"""
    if page_count <= 0:
        return [0] * len(chunks)
    n = len(chunks)
    page_nums = []
    for i in range(n):
        page_nums.append(int(i * page_count / n) + 1)
    return page_nums