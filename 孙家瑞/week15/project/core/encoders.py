"""默认编码器和聊天模型实现，模块导入时自动注册。"""

import os

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from .config import BGE_MODEL_PATH, CLIP_MODEL_PATH, HF_ENDPOINT, QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL
from .model_loader import register_encoder, register_chat_model


class BgeEncoder:
    """BGE 文本编码器（512 维）。"""

    def __init__(self, model_path: str | None = None, **kwargs):
        os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
        self._model = SentenceTransformer(model_path or BGE_MODEL_PATH)

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=normalize_embeddings)  # type: ignore[return-value]


class ClipEncoder:
    """Jina-CLIP 跨模态编码器（1024 维），同时编码文本和图像。"""

    def __init__(self, model_path: str | None = None, **kwargs):
        os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
        self._model = SentenceTransformer(
            model_path or CLIP_MODEL_PATH,
            trust_remote_code=True,
            truncate_dim=1024,
        )

    def encode(self, texts_or_images: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        return self._model.encode(texts_or_images, normalize_embeddings=normalize_embeddings)  # type: ignore[return-value]


class QwenChatModel:
    """Qwen 云端聊天模型，通过 DashScope API 调用。"""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None, **kwargs):
        self._client = OpenAI(
            api_key=api_key or QWEN_API_KEY,
            base_url=base_url or QWEN_BASE_URL,
        )
        self._model = model or QWEN_MODEL

    def chat(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


# 自动注册默认实现
register_encoder("bge", BgeEncoder)
register_encoder("clip", ClipEncoder)
register_chat_model("qwen", QwenChatModel)
