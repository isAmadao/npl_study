from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

from safetensors.torch import load_file

from rag_mvp.config import settings
from rag_mvp.services.model_resolver import ModelResolver


class LocalJinaClipModel:
    def __init__(
        self,
        model_name_or_path: str | Path | None = None,
        resolver: ModelResolver | None = None,
        package_dir: Path | None = None,
    ) -> None:
        self.model_name_or_path = (
            model_name_or_path or settings.multimodal_model_name_or_path
        )
        self.resolver = resolver or ModelResolver()
        self.package_dir = Path(package_dir or settings.multimodal_patch_dir)
        self.asset_dir: Path | None = None
        self._model = None

    def _load_package(self):
        package_name = "rag_mvp_local_jina_clip_v2"
        if package_name in sys.modules:
            return package_name

        spec = importlib.util.spec_from_file_location(
            package_name,
            self.package_dir / "__init__.py",
            submodule_search_locations=[str(self.package_dir)],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load package from {self.package_dir}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        spec.loader.exec_module(module)
        return package_name

    def _load_model(self):
        if self._model is not None:
            return self._model

        self.asset_dir = self.resolver.resolve_multimodal_assets(
            self.model_name_or_path
        )
        package_name = self._load_package()
        config_mod = importlib.import_module(f"{package_name}.configuration_clip")
        model_mod = importlib.import_module(f"{package_name}.modeling_clip")

        config = config_mod.JinaCLIPConfig.from_pretrained(
            str(self.package_dir),
            local_files_only=True,
        )
        config._name_or_path = str(self.asset_dir)
        local_text_encoder_dir = self.package_dir / "text_encoder"
        if local_text_encoder_dir.exists():
            config.text_config.hf_model_name_or_path = str(local_text_encoder_dir)
        model = model_mod.JinaCLIPModel(config)
        model.config._name_or_path = str(self.asset_dir)

        safetensors_path = self.asset_dir / "model.safetensors"
        if safetensors_path.exists():
            state_dict = load_file(str(safetensors_path), device="cpu")
        else:
            state_dict = self._load_torch_state_dict()

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            raise RuntimeError(
                "Missing keys while loading jina-clip-v2: "
                + ", ".join(missing_keys[:10])
            )
        if unexpected_keys:
            raise RuntimeError(
                "Unexpected keys while loading jina-clip-v2: "
                + ", ".join(unexpected_keys[:10])
            )

        model.eval()
        self._model = model
        return self._model

    def _load_torch_state_dict(self):
        import torch

        if self.asset_dir is None:
            raise RuntimeError("Multimodal asset directory has not been resolved yet.")
        pytorch_path = self.asset_dir / "pytorch_model.bin"
        if not pytorch_path.exists():
            raise FileNotFoundError(
                f"No local weights found under {self.asset_dir}"
            )
        state_dict = torch.load(str(pytorch_path), map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            return state_dict["state_dict"]
        return state_dict

    def encode_text(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load_model()
        vectors = model.encode_text(
            texts,
            task="retrieval.query",
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        if hasattr(vectors, "ndim") and getattr(vectors, "ndim", 1) == 1:
            return [vectors.tolist()]
        return vectors.tolist()

    def encode_images(self, image_paths: list[str]) -> list[list[float]]:
        if not image_paths:
            return []
        model = self._load_model()
        vectors = model.encode_image(
            image_paths,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        if hasattr(vectors, "ndim") and getattr(vectors, "ndim", 1) == 1:
            return [vectors.tolist()]
        return vectors.tolist()
