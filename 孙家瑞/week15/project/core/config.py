"""应用配置 —— 路径、模型名、连接参数，统一管理。"""

import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"
DB_PATH = BASE_DIR / "data.db"

for d in [UPLOAD_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Milvus 连接
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "rag_collection")

# 模型路径
BGE_MODEL_PATH = os.getenv("BGE_MODEL_PATH", "BAAI/bge-small-zh-v1.5")
CLIP_MODEL_PATH = os.getenv("CLIP_MODEL_PATH", "jinaai/jina-clip-v2")
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

# Qwen API
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")

# Mineru API
MINERU_API_URL = os.getenv("MINERU_API_URL", "http://127.0.0.1:30000")

# 检索参数
TOP_K = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "256"))
