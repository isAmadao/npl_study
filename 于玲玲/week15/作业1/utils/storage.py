"""本地文件存储：PDF 保存、目录管理。"""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import UploadFile

import config


def ensure_kb_dirs(kb_id: int) -> dict[str, Path]:
    """为指定知识库创建存储目录，返回子目录路径字典。"""
    kb_root = config.STORAGE_DIR / str(kb_id)
    dirs = {
        "pdfs": kb_root / "pdfs",
        "markdown": kb_root / "markdown",
        "images": kb_root / "images",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def save_uploaded_pdf(kb_id: int, file: UploadFile) -> Path:
    """保存上传的 PDF 文件，返回存储路径。"""
    kb_dirs = ensure_kb_dirs(kb_id)
    # 使用 uuid 避免文件名冲突
    stem = Path(file.filename).stem
    safe_name = f"{stem}_{uuid.uuid4().hex[:8]}.pdf"
    filepath = kb_dirs["pdfs"] / safe_name

    with open(filepath, "wb") as f:
        while chunk := file.file.read(1024 * 1024):
            f.write(chunk)

    return filepath


def get_chunk_images_dir(kb_id: int) -> Path:
    """获取知识库的图片存储目录。"""
    return config.STORAGE_DIR / str(kb_id) / "images"