"""检索服务 —— Milvus 向量搜索。"""

import os
import re

from pymilvus import MilvusClient as _MilvusClient

from ..core.config import COLLECTION_NAME, TOP_K
from ..core.model_loader import get_encoder
from .encoder_service import encode_chunk


def retrieve(question: str, milvus_client: _MilvusClient, top_k: int | None = None) -> str:
    """
    检索与问题最相关的 chunk 文本，拼接为上下文字符串返回。
    图文链接中的 images/ 会被替换为实际本地路径。
    """
    k = top_k or TOP_K
    bge = get_encoder("bge")
    question_vec = bge.encode([question], normalize_embeddings=True)
    question_vec = list(question_vec[0])

    results = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[question_vec],
        limit=k,
        anns_field="text_vector",
        output_fields=["text", "file_id", "file_name", "file_path"],
    )

    related_parts: list[str] = []
    for hit in results[0]:
        entity = hit["entity"]
        text = entity["text"]
        # 将 images/ 相对路径替换为实际的 processed 子目录路径
        file_dir = os.path.basename(entity["file_path"]).split(".")[0]
        text = re.sub(r"images/", f"./processed/{file_dir}/vlm/images/", text)
        related_parts.append(text)

    return "\n\n".join(related_parts)
