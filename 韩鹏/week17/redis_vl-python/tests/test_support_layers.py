"""Tests for configuration, Redis helpers, and vectorizer support layers."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from redis.exceptions import RedisError

from app import config as config_module
from app import redis_client as redis_client_module
from app.vectorizer import Vectorizer


def test_get_int_env_default_and_invalid(monkeypatch) -> None:
    monkeypatch.delenv("REDIS_PORT", raising=False)
    assert config_module._get_int_env("REDIS_PORT", 6379) == 6379

    monkeypatch.setenv("REDIS_PORT", "invalid")
    with pytest.raises(ValueError, match="must be an integer"):
        config_module._get_int_env("REDIS_PORT", 6379)


def test_get_settings_reads_env(monkeypatch) -> None:
    config_module.get_settings.cache_clear()
    monkeypatch.setenv("REDIS_HOST", "127.0.0.1")
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("REDIS_PASSWORD", "secret")
    monkeypatch.setenv("HF_CACHE_DIR", "C:\\cache")
    monkeypatch.setenv("HF_MODEL_NAME", "demo/model")
    monkeypatch.setenv("EMBEDDING_DIM", "256")
    monkeypatch.setenv("APP_HOST", "127.0.0.1")
    monkeypatch.setenv("APP_PORT", "9000")

    settings = config_module.get_settings()

    assert settings.redis_host == "127.0.0.1"
    assert settings.redis_port == 6380
    assert settings.redis_password == "secret"
    assert settings.hf_cache_dir == "C:\\cache"
    assert settings.hf_model_name == "demo/model"
    assert settings.embedding_dim == 256
    assert settings.app_host == "127.0.0.1"
    assert settings.app_port == 9000
    config_module.get_settings.cache_clear()


def test_build_client_and_getters(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_redis(**kwargs):
        captured.update(kwargs)
        return {"client": "ok", "db": kwargs["db"]}

    monkeypatch.setattr(redis_client_module.redis, "Redis", fake_redis)
    monkeypatch.setattr(
        redis_client_module,
        "get_settings",
        lambda: config_module.Settings(
            redis_host="localhost",
            redis_port=16379,
            redis_password="",
            hf_cache_dir="cache",
            hf_model_name="model",
            embedding_dim=512,
            app_host="0.0.0.0",
            app_port=18000,
        ),
    )

    client = redis_client_module._build_client(db=3)
    assert client == {"client": "ok", "db": 3}
    assert captured["host"] == "localhost"
    assert captured["port"] == 16379
    assert captured["password"] is None
    assert captured["db"] == 3
    assert captured["decode_responses"] is True


def test_ensure_connection_wraps_redis_error(monkeypatch) -> None:
    class FakeClient:
        def ping(self) -> None:
            raise RedisError("down")

    monkeypatch.setattr(
        redis_client_module,
        "get_settings",
        lambda: config_module.Settings(
            redis_host="localhost",
            redis_port=6379,
            redis_password="",
            hf_cache_dir="cache",
            hf_model_name="model",
            embedding_dim=384,
            app_host="0.0.0.0",
            app_port=8000,
        ),
    )

    with pytest.raises(RuntimeError, match="Unable to connect to Redis"):
        redis_client_module._ensure_connection(FakeClient())


def test_get_redis_client_calls_connection_check(monkeypatch) -> None:
    fake_client = object()
    monkeypatch.setattr(redis_client_module, "_build_client", lambda db=0: fake_client)

    called = {"value": False}

    def fake_ensure(client) -> None:
        called["value"] = client is fake_client

    monkeypatch.setattr(redis_client_module, "_ensure_connection", fake_ensure)

    result = redis_client_module.get_redis_client()
    assert result is fake_client
    assert called["value"] is True
    assert redis_client_module.get_test_client() is fake_client


def test_vectorizer_available_models_and_missing_cache(tmp_path) -> None:
    (tmp_path / "models--org--model-a").mkdir()
    (tmp_path / "models--org--model-b").mkdir()
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    vectorizer = Vectorizer(cache_dir=str(tmp_path))
    assert vectorizer.available_cached_models() == ["org/model-a", "org/model-b"]

    missing = Vectorizer(cache_dir=str(tmp_path / "not-found"))
    assert missing.available_cached_models() == []

    with pytest.raises(RuntimeError, match="does not exist"):
        missing._load_model()


def test_vectorizer_embed_and_batch_with_fake_sentence_transformer(monkeypatch, tmp_path) -> None:
    fake_module = types.ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, cache_folder: str, local_files_only: bool) -> None:
            self.model_name = model_name
            self.cache_folder = cache_folder
            self.local_files_only = local_files_only

        def encode(self, texts, convert_to_numpy: bool = True):
            if isinstance(texts, str):
                return np.array([1.0, 2.0, 3.0], dtype=np.float32)
            return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    vectorizer = Vectorizer(model_name="fake/model", cache_dir=str(tmp_path))

    assert vectorizer.embed("hello") == [1.0, 2.0, 3.0]
    assert vectorizer.embed_batch(["a", "b"]) == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    with pytest.raises(ValueError, match="non-empty string"):
        vectorizer.embed("   ")

    with pytest.raises(ValueError, match="non-empty strings"):
        vectorizer.embed_batch(["ok", ""])


def test_vectorizer_load_failure_reports_available_models(monkeypatch, tmp_path) -> None:
    (tmp_path / "models--demo--cached").mkdir()
    fake_module = types.ModuleType("sentence_transformers")

    class FailingSentenceTransformer:
        def __init__(self, model_name: str, cache_folder: str, local_files_only: bool) -> None:
            raise RuntimeError("load failed")

    fake_module.SentenceTransformer = FailingSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    vectorizer = Vectorizer(model_name="demo/model", cache_dir=str(tmp_path))

    with pytest.raises(RuntimeError, match="Available cached models: demo/cached"):
        vectorizer._load_model()
