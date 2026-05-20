from pymilvus import MilvusClient

# 由 main.py 的 lifespan 设置
_milvus_client: MilvusClient | None = None
