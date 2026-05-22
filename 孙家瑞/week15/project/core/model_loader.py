"""编码器和聊天模型的可插拔注册表。

注册新实现以替换默认模型:

    from core.model_loader import register_encoder, register_chat_model, get_encoder

    class MyCustomEncoder(TextEncoder):
        def encode(self, text): ...

    register_encoder("bge", MyCustomEncoder)
    # 然后在 config 中将 BGE_ENCODER_NAME 设为 "custom"
"""

from typing import Protocol

import numpy as np


class TextEncoder(Protocol):
    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray: ...


class ImageEncoder(Protocol):
    def encode(self, images: list[str], normalize_embeddings: bool = True) -> np.ndarray: ...


class ChatModel(Protocol):
    def chat(self, prompt: str, context: str) -> str: ...


# 注册表
_encoder_registry: dict[str, type] = {}
_chat_model_registry: dict[str, type] = {}

# 实例缓存
_encoder_instance: dict[str, object] = {}
_chat_model_instance: dict[str, object] = {}


def register_encoder(name: str, cls: type) -> None:
    _encoder_registry[name] = cls
    _encoder_instance.pop(name, None)


def get_encoder(name: str, **kwargs) -> object:
    if name not in _encoder_instance:
        cls = _encoder_registry[name]
        _encoder_instance[name] = cls(**kwargs)
    return _encoder_instance[name]


def register_chat_model(name: str, cls: type) -> None:
    _chat_model_registry[name] = cls
    _chat_model_instance.pop(name, None)


def get_chat_model(name: str, **kwargs) -> object:
    if name not in _chat_model_instance:
        cls = _chat_model_registry[name]
        _chat_model_instance[name] = cls(**kwargs)
    return _chat_model_instance[name]
