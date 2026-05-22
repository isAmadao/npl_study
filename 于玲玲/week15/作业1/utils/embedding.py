"""
文本编码（BGE）与图像编码（CLIP）的封装。
首次加载会从 HuggingFace 下载模型，之后缓存到本地。
"""
from __future__ import annotations

import torch
import numpy as np
from PIL import Image
from pathlib import Path

import config


class TextEmbedder:
    """基于 BGE 的文本编码器。"""

    def __init__(self, model_name: str | None = None):
        from transformers import AutoModel, AutoTokenizer

        self._model_name = model_name or config.BGE_MODEL_NAME
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name)
        self._model.eval()

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    def encode(self, texts: list[str]) -> np.ndarray:
        """将文本列表编码为归一化向量数组，shape=(len(texts), dim)。"""
        if not texts:
            return np.array([])

        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**encoded)
            # BGE 使用最后一层 [CLS] token 的输出
            embeddings = outputs.last_hidden_state[:, 0, :]

        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy().astype(np.float32)


class ImageEmbedder:
    """基于 CLIP 的图像编码器。"""

    def __init__(self, model_name: str | None = None):
        from transformers import CLIPProcessor, CLIPModel

        self._model_name = model_name or config.CLIP_MODEL_NAME
        self._model = CLIPModel.from_pretrained(self._model_name)
        self._processor = CLIPProcessor.from_pretrained(self._model_name)
        self._model.eval()

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    def encode(self, image_paths: list[str]) -> np.ndarray:
        """将图片路径列表编码为归一化向量数组。"""
        if not image_paths:
            return np.array([])

        images = []
        valid_indices = []
        for i, p in enumerate(image_paths):
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                valid_indices.append(i)
            except Exception as e:
                print(f"[ImageEmbedder] 跳过无法读取的图片 {p}: {e}")
                continue

        if not images:
            return np.array([])

        inputs = self._processor(images=images, return_tensors="pt", padding=True).to(self._device)

        with torch.no_grad():
            image_features = self._model.get_image_features(**inputs)

        image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
        embeddings = image_features.cpu().numpy().astype(np.float32)

        # 如果部分图片加载失败，返回的 embedding 数组可能比输入短
        if len(valid_indices) < len(image_paths):
            full = np.zeros((len(image_paths), embeddings.shape[1]), dtype=np.float32)
            for j, idx in enumerate(valid_indices):
                full[idx] = embeddings[j]
            return full

        return embeddings

    def encode_text(self, texts: list[str]) -> np.ndarray:
        """使用 CLIP 的文本编码器对文本编码（用于跨模态检索）。"""
        if not texts:
            return np.array([])

        inputs = self._processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self._device)

        with torch.no_grad():
            text_features = self._model.get_text_features(**inputs)

        text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)
        return text_features.cpu().numpy().astype(np.float32)