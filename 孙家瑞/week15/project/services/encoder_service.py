"""编码服务 —— 将文本 chunk 和图像编码为向量。"""

import traceback
from pathlib import Path

import numpy as np

from ..core.model_loader import get_encoder


def split_text_to_chunks(lines: list[str], chunk_size: int = 256) -> list[str]:
    chunks: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line == "# References":
            continue
        if len(line) > 2 and line[0] == "[" and line[1].isdigit():
            continue

        if not chunks or len(chunks[-1]) > chunk_size:
            chunks.append(line)
        else:
            chunks[-1] += "\n" + line
    return chunks


def encode_chunk(text: str, markdown_dir: str) -> dict:
    """
    编码单个 chunk，返回:
      {text_vector, clip_text_vector, clip_image_vector, text, ...}
    """
    bge = get_encoder("bge")
    clip = get_encoder("clip")

    # 分离纯文本行和图片行
    lines = text.split("\n")
    text_lines = [l for l in lines if not l.startswith("![")]
    image_lines = [l for l in lines if l.startswith("![")]

    text_only = "\n".join(text_lines)

    try:
        text_vec = bge.encode([text_only], normalize_embeddings=True)
        text_vec = list(text_vec[0])
    except Exception:
        traceback.print_exc()
        text_vec = [0.0] * 512

    try:
        clip_text_vec = clip.encode([text_only], normalize_embeddings=True)
        clip_text_vec = list(clip_text_vec[0])
    except Exception:
        traceback.print_exc()
        clip_text_vec = [0.0] * 1024

    try:
        if image_lines:
            img_rel = image_lines[0].split("](")[1].rstrip(")")
            img_path = Path(markdown_dir) / img_rel.split("/")[-1]
            clip_img_vec = clip.encode([str(img_path)], normalize_embeddings=True)
            clip_img_vec = list(clip_img_vec[0])
        else:
            clip_img_vec = [0.0] * 1024
    except Exception:
        traceback.print_exc()
        clip_img_vec = [0.0] * 1024

    return {
        "text_vector": text_vec,
        "clip_text_vector": clip_text_vec,
        "clip_image_vector": clip_img_vec,
        "text": text,
    }


def encode_document(markdown_path: str, file_id: int, file_name: str, file_path: str) -> list[dict]:
    """解析 markdown 文件，切分 chunk 并编码，返回数据列表。"""
    with open(markdown_path, encoding="utf-8") as f:
        lines = f.readlines()

    chunks = split_text_to_chunks(lines)
    markdown_dir = str(Path(markdown_path).parent)

    records = []
    for chunk in chunks:
        data = encode_chunk(chunk, markdown_dir)
        data["file_id"] = file_id
        data["file_name"] = file_name
        data["file_path"] = file_path
        records.append(data)
    return records
