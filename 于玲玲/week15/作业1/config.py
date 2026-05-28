"""
全局配置：所有服务的共用参数，通过环境变量或默认值注入。
"""
import os
from pathlib import Path

# ---------- 项目根目录 ----------
BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"

# ---------- 数据库 ----------
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", str(BASE_DIR / "metadata.db"))

# ---------- Kafka ----------
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_DOCUMENT_PARSE = os.getenv("KAFKA_TOPIC_DOCUMENT_PARSE", "document_parse")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "offline_worker")

# ---------- Milvus ----------
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_USE_LITE = os.getenv("MILVUS_USE_LITE", "true").lower() == "true"
MILVUS_LITE_PATH = os.getenv("MILVUS_LITE_PATH", str(BASE_DIR / "milvus_lite.db"))

# 向量维度: BGE-large-zh 1024, CLIP-ViT-B/32 512
TEXT_EMBEDDING_DIM = int(os.getenv("TEXT_EMBEDDING_DIM", "1024"))
IMAGE_EMBEDDING_DIM = int(os.getenv("IMAGE_EMBEDDING_DIM", "512"))

# ---------- 模型 ----------
# BGE 文本编码模型
BGE_MODEL_NAME = os.getenv("BGE_MODEL_NAME", "BAAI/bge-large-zh-v1.5")
# CLIP 多模态编码模型
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
# Qwen-VL 多模态问答模型 (API endpoint 或本地路径)
QWEN_VL_API_URL = os.getenv("QWEN_VL_API_URL", "http://localhost:8002/v1")
QWEN_VL_MODEL_NAME = os.getenv("QWEN_VL_MODEL_NAME", "Qwen/Qwen2-VL-7B-Instruct")
# MinerU 解析服务地址
MINERU_API_URL = os.getenv("MINERU_API_URL", "http://localhost:8003")

# ---------- 检索 ----------
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))

# ---------- 服务端口 ----------
UPLOAD_SERVICE_PORT = int(os.getenv("UPLOAD_SERVICE_PORT", "8001"))
CHAT_SERVICE_PORT = int(os.getenv("CHAT_SERVICE_PORT", "8000"))

# ---------- 并发 ----------
UPLOAD_WORKERS = int(os.getenv("UPLOAD_WORKERS", "10"))