from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_mvp.services.model_resolver import ModelResolver


def _touch(path: Path, content: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class ModelResolverTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(
            tempfile.mkdtemp(prefix="resolver_test_", dir=str(ROOT_DIR / "data"))
        )
        self.cache_dir = self.temp_dir / "hf_cache"
        self.bundled_text_dir = self.temp_dir / "bundled_text"
        self.bundled_multimodal_dir = self.temp_dir / "bundled_jina"

        _touch(self.bundled_text_dir / "config.json")
        _touch(self.bundled_text_dir / "model.safetensors", "weights")

        for file_name in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "preprocessor_config.json",
        ):
            _touch(self.bundled_multimodal_dir / file_name)
        _touch(self.bundled_multimodal_dir / "model.safetensors", "weights")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_resolve_text_model_prefers_cached_snapshot(self):
        cached_dir = self.temp_dir / "cached_text_snapshot"
        cached_dir.mkdir(parents=True)
        _touch(cached_dir / "config.json")
        _touch(cached_dir / "model.safetensors", "weights")
        resolver = ModelResolver(
            cache_dir=self.cache_dir,
            bundled_text_dir=self.bundled_text_dir,
            bundled_multimodal_dir=self.bundled_multimodal_dir,
        )

        with patch(
            "rag_mvp.services.model_resolver.snapshot_download",
            return_value=str(cached_dir),
        ) as mocked_download:
            resolved = resolver.resolve_text_model("BAAI/bge-small-zh-v1.5")

        self.assertEqual(resolved, cached_dir)
        mocked_download.assert_called_once()
        _, kwargs = mocked_download.call_args
        self.assertEqual(kwargs["repo_id"], "BAAI/bge-small-zh-v1.5")
        self.assertEqual(kwargs["cache_dir"], str(self.cache_dir))
        self.assertTrue(kwargs["local_files_only"])

    def test_resolve_text_model_downloads_when_cache_misses(self):
        downloaded_dir = self.temp_dir / "downloaded_text_snapshot"
        downloaded_dir.mkdir(parents=True)
        _touch(downloaded_dir / "config.json")
        _touch(downloaded_dir / "model.safetensors", "weights")
        resolver = ModelResolver(
            cache_dir=self.cache_dir,
            bundled_text_dir=self.bundled_text_dir,
            bundled_multimodal_dir=self.bundled_multimodal_dir,
        )
        calls: list[bool] = []

        def fake_snapshot_download(*, local_files_only: bool = False, **kwargs):
            del kwargs
            calls.append(local_files_only)
            if local_files_only:
                raise RuntimeError("cache miss")
            return str(downloaded_dir)

        with patch(
            "rag_mvp.services.model_resolver.snapshot_download",
            side_effect=fake_snapshot_download,
        ):
            resolved = resolver.resolve_text_model("BAAI/bge-small-zh-v1.5")

        self.assertEqual(resolved, downloaded_dir)
        self.assertEqual(calls, [True, False])

    def test_resolve_multimodal_assets_falls_back_to_bundled_dir(self):
        resolver = ModelResolver(
            cache_dir=self.cache_dir,
            auto_download=False,
            bundled_text_dir=self.bundled_text_dir,
            bundled_multimodal_dir=self.bundled_multimodal_dir,
        )

        with patch(
            "rag_mvp.services.model_resolver.snapshot_download",
            side_effect=RuntimeError("unavailable"),
        ):
            resolved = resolver.resolve_multimodal_assets("jinaai/jina-clip-v2")

        self.assertEqual(resolved, self.bundled_multimodal_dir)

    def test_resolve_multimodal_assets_downloads_when_cached_snapshot_is_incomplete(self):
        cached_dir = self.temp_dir / "cached_jina_snapshot"
        downloaded_dir = self.temp_dir / "downloaded_jina_snapshot"
        cached_dir.mkdir(parents=True)
        downloaded_dir.mkdir(parents=True)

        _touch(cached_dir / "config.json")
        _touch(cached_dir / "model.safetensors", "weights")

        for file_name in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "preprocessor_config.json",
        ):
            _touch(downloaded_dir / file_name)
        _touch(downloaded_dir / "model.safetensors", "weights")

        resolver = ModelResolver(
            cache_dir=self.cache_dir,
            bundled_text_dir=self.bundled_text_dir,
            bundled_multimodal_dir=self.bundled_multimodal_dir,
        )
        calls: list[bool] = []

        def fake_snapshot_download(*, local_files_only: bool = False, **kwargs):
            del kwargs
            calls.append(local_files_only)
            if local_files_only:
                return str(cached_dir)
            return str(downloaded_dir)

        with patch(
            "rag_mvp.services.model_resolver.snapshot_download",
            side_effect=fake_snapshot_download,
        ):
            resolved = resolver.resolve_multimodal_assets("jinaai/jina-clip-v2")

        self.assertEqual(resolved, downloaded_dir)
        self.assertEqual(calls, [True, False])


if __name__ == "__main__":
    unittest.main()
