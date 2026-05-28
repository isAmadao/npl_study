from __future__ import annotations

from pathlib import Path
from typing import Iterable

from huggingface_hub import snapshot_download

from rag_mvp.config import settings


TEXT_MODEL_PATTERNS = (
    "*.json",
    "*.safetensors",
    "*.bin",
)

JINA_CLIP_PATTERNS = (
    "config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
)


def _as_existing_path(value: str | Path) -> Path | None:
    candidate = Path(value).expanduser()
    if candidate.exists():
        return candidate.resolve()
    return None


def _build_download_error(
    repo_id: str,
    cache_dir: Path,
    auto_download: bool,
) -> RuntimeError:
    if auto_download:
        return RuntimeError(
            f"Unable to resolve model '{repo_id}' from Hugging Face cache or download it automatically."
        )
    return RuntimeError(
        f"Model '{repo_id}' was not found in Hugging Face cache '{cache_dir}', and AUTO_DOWNLOAD_MODELS is disabled."
    )


class ModelResolver:
    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        prefer_hf_cache: bool | None = None,
        auto_download: bool | None = None,
        bundled_text_dir: Path | None = None,
        bundled_multimodal_dir: Path | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir or settings.huggingface_hub_cache_dir)
        self.prefer_hf_cache = (
            settings.prefer_hf_cache
            if prefer_hf_cache is None
            else prefer_hf_cache
        )
        self.auto_download = (
            settings.auto_download_models
            if auto_download is None
            else auto_download
        )
        self.bundled_text_dir = Path(bundled_text_dir or settings.bundled_text_model_dir)
        self.bundled_multimodal_dir = Path(
            bundled_multimodal_dir or settings.multimodal_patch_dir
        )

    def resolve_text_model(self, model_name_or_path: str | Path) -> Path:
        local_path = _as_existing_path(model_name_or_path)
        if local_path is not None:
            return local_path

        repo_id = str(model_name_or_path)
        try:
            return self._resolve_hf_snapshot(
                repo_id,
                TEXT_MODEL_PATTERNS,
                self._has_text_model_assets,
            )
        except Exception as exc:
            if self._has_text_model_assets(self.bundled_text_dir):
                return self.bundled_text_dir
            raise _build_download_error(
                repo_id=repo_id,
                cache_dir=self.cache_dir,
                auto_download=self.auto_download,
            ) from exc

    def resolve_multimodal_assets(self, model_name_or_path: str | Path) -> Path:
        local_path = _as_existing_path(model_name_or_path)
        if local_path is not None:
            return local_path

        repo_id = str(model_name_or_path)
        try:
            return self._resolve_hf_snapshot(
                repo_id,
                JINA_CLIP_PATTERNS,
                self._has_jina_clip_assets,
            )
        except Exception as exc:
            if self._has_jina_clip_assets(self.bundled_multimodal_dir):
                return self.bundled_multimodal_dir
            raise _build_download_error(
                repo_id=repo_id,
                cache_dir=self.cache_dir,
                auto_download=self.auto_download,
            ) from exc

    def _resolve_hf_snapshot(
        self,
        repo_id: str,
        allow_patterns: Iterable[str],
        validator,
    ) -> Path:
        patterns = list(allow_patterns)
        if self.prefer_hf_cache:
            try:
                cached_snapshot = Path(
                    self._download_snapshot(
                        repo_id=repo_id,
                        allow_patterns=patterns,
                        local_files_only=True,
                    )
                )
                if validator(cached_snapshot):
                    return cached_snapshot
            except Exception:
                pass

        if self.auto_download:
            downloaded_snapshot = Path(
                self._download_snapshot(
                    repo_id=repo_id,
                    allow_patterns=patterns,
                )
            )
            if validator(downloaded_snapshot):
                return downloaded_snapshot
            raise RuntimeError(
                f"Downloaded model snapshot for '{repo_id}' is incomplete under '{downloaded_snapshot}'."
            )

        raise _build_download_error(
            repo_id=repo_id,
            cache_dir=self.cache_dir,
            auto_download=self.auto_download,
        )

    def _download_snapshot(
        self,
        *,
        repo_id: str,
        allow_patterns: list[str],
        local_files_only: bool = False,
    ) -> str:
        return snapshot_download(
            repo_id=repo_id,
            cache_dir=str(self.cache_dir),
            local_files_only=local_files_only,
            allow_patterns=allow_patterns,
        )

    @staticmethod
    def _has_text_model_assets(model_dir: Path) -> bool:
        return (
            model_dir.exists()
            and (model_dir / "config.json").exists()
            and (
                (model_dir / "model.safetensors").exists()
                or (model_dir / "pytorch_model.bin").exists()
            )
        )

    @staticmethod
    def _has_jina_clip_assets(model_dir: Path) -> bool:
        required_files = (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "preprocessor_config.json",
        )
        has_weights = (
            (model_dir / "model.safetensors").exists()
            or (model_dir / "pytorch_model.bin").exists()
        )
        return model_dir.exists() and has_weights and all(
            (model_dir / name).exists() for name in required_files
        )
