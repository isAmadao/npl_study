"""Centralized application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _get_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value in (None, ""):
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc


@dataclass(frozen=True)
class Settings:
    redis_host: str
    redis_port: int
    redis_password: str
    hf_cache_dir: str
    hf_model_name: str
    embedding_dim: int
    app_host: str
    app_port: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings object loaded from environment variables."""
    return Settings(
        redis_host=os.getenv("REDIS_HOST", "localhost"),
        redis_port=_get_int_env("REDIS_PORT", 6379),
        redis_password=os.getenv("REDIS_PASSWORD", ""),
        hf_cache_dir=os.getenv("HF_CACHE_DIR", r"C:\Users\peng\.cache\huggingface\hub"),
        hf_model_name=os.getenv("HF_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
        embedding_dim=_get_int_env("EMBEDDING_DIM", 384),
        app_host=os.getenv("APP_HOST", "0.0.0.0"),
        app_port=_get_int_env("APP_PORT", 8000),
    )
