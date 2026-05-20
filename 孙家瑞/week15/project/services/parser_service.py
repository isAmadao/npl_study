"""文档解析服务 —— 调用 mineru API 将 PDF 转为 Markdown + 图片。"""

import glob
import os
import traceback
from pathlib import Path

import requests

from ..core.config import MINERU_API_URL, PROCESSED_DIR
from ..models.orm import File, SessionLocal


def parse_document(file: File) -> str | None:
    """
    调用 mineru API 解析 PDF，输出到 processed/ 目录。
    返回解析后的 markdown 文件路径，失败返回 None。
    """
    output_dir = str(PROCESSED_DIR)
    url = f"{MINERU_API_URL}/api/parse"

    try:
        resp = requests.post(
            url,
            json={"file_path": file.filepath, "output_dir": output_dir},
            timeout=600,
        )
        resp.raise_for_status()
    except Exception:
        update_file_state(file.id, "failed", traceback.format_exc())
        return None

    base = os.path.basename(file.filepath).split(".")[0]
    candidates = glob.glob(os.path.join(output_dir, base) + "/**/*.md", recursive=True)
    if not candidates:
        update_file_state(file.id, "failed", "no markdown output from mineru")
        return None

    update_file_state(file.id, "completed")
    return candidates[0]


def update_file_state(file_id: int, state: str, error: str | None = None) -> None:
    with SessionLocal() as session:
        f = session.query(File).filter(File.id == file_id).first()
        if f:
            f.filestate = state
            f.error_message = error
            session.commit()
