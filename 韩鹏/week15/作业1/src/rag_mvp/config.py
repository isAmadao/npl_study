from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


def _env_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    root_dir: Path = ROOT_DIR
    upload_dir: Path = ROOT_DIR / "uploads"
    data_dir: Path = ROOT_DIR / "data"
    derived_dir: Path = ROOT_DIR / "uploads" / "_derived"
    local_queue_dir: Path = ROOT_DIR / "data" / "local_queue"
    local_vector_store_path: Path = ROOT_DIR / "data" / "local_vector_store.json"
    db_path: Path = ROOT_DIR / "db.db"

    text_model_name_or_path: str = os.getenv(
        "TEXT_MODEL_NAME_OR_PATH",
        "BAAI/bge-small-zh-v1.5",
    )
    multimodal_model_name_or_path: str = os.getenv(
        "MULTIMODAL_MODEL_NAME_OR_PATH",
        "jinaai/jina-clip-v2",
    )
    bundled_text_model_dir: Path = ROOT_DIR / "models" / "BAAI" / "bge-small-zh-v1.5"
    multimodal_patch_dir: Path = ROOT_DIR / "models" / "jinaai" / "jina-clip-v2"
    huggingface_hub_cache_dir: Path = Path(
        os.getenv(
            "HF_HUB_CACHE",
            str(Path.home() / ".cache" / "huggingface" / "hub"),
        )
    )
    prefer_hf_cache: bool = _env_flag("PREFER_HF_CACHE", True)
    auto_download_models: bool = _env_flag("AUTO_DOWNLOAD_MODELS", True)

    kafka_bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    kafka_topic: str = os.getenv("KAFKA_TOPIC", "rag_file_tasks")
    kafka_group_id: str = os.getenv("KAFKA_GROUP_ID", "rag_mvp_worker")

    milvus_uri: str = os.getenv(
        "MILVUS_URI",
        "https://in03-5cb3b56f3af9ebc.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn",
    )
    milvus_token: str = os.getenv(
        "MILVUS_TOKEN",
        "9027d285f74e5ce113bf24162fc5cabe04b67db3ee25055f4748ea23785f00d0fa9b8217c108a04dc77c4a703b5860a7d39d7a7b",
    )
    milvus_collection_name: str = os.getenv("MILVUS_COLLECTION_NAME", "rag_data_new")

    llm_model: str = os.getenv("LLM_MODEL", "qwen3.5-plus")
    llm_api_key: str = os.getenv("LLM_API_KEY", "API-KEY")
    llm_base_url: str = os.getenv(
        "LLM_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    text_chunk_size: int = 500
    text_chunk_overlap: int = 80
    query_top_k: int = 4
    target_vector_dim: int = 1024

    def ensure_directories(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.derived_dir.mkdir(parents=True, exist_ok=True)
        self.local_queue_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
