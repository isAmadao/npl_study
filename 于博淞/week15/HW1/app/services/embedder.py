"""BGE (text) and CLIP (image) embedding wrappers. Models are loaded lazily on first call."""
from __future__ import annotations

import torch
from PIL import Image

_bge_tokenizer = None
_bge_model = None
_clip_model = None
_clip_processor = None


def _load_bge() -> None:
    global _bge_tokenizer, _bge_model
    if _bge_model is not None:
        return
    from transformers import AutoModel, AutoTokenizer
    _bge_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-zh-v1.5")
    _bge_model = AutoModel.from_pretrained("BAAI/bge-base-zh-v1.5")
    _bge_model.eval()


def _load_clip() -> None:
    global _clip_model, _clip_processor
    if _clip_model is not None:
        return
    from transformers import CLIPModel, CLIPProcessor
    _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _clip_model.eval()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed text chunks with BGE. Returns L2-normalized vectors (dim=768)."""
    _load_bge()
    inputs = _bge_tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = _bge_model(**inputs)
    vecs = outputs.last_hidden_state[:, 0]  # CLS token
    vecs = torch.nn.functional.normalize(vecs, dim=-1)
    return vecs.tolist()


def embed_images(image_paths: list[str]) -> list[list[float]]:
    """Embed images with CLIP. Returns L2-normalized vectors (dim=512)."""
    _load_clip()
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = _clip_processor(images=images, return_tensors="pt")
    with torch.no_grad():
        vecs = _clip_model.get_image_features(**inputs)
    vecs = torch.nn.functional.normalize(vecs, dim=-1)
    return vecs.tolist()


def embed_query_for_images(text: str) -> list[float]:
    """Embed a text query into CLIP space for image retrieval (dim=512)."""
    _load_clip()
    inputs = _clip_processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        vec = _clip_model.get_text_features(**inputs)
    vec = torch.nn.functional.normalize(vec, dim=-1)
    return vec[0].tolist()
