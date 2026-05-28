from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from app.core.config import settings

TEXT_COLLECTION = "text_chunks"
IMAGE_COLLECTION = "image_chunks"

TEXT_DIM = 768   # BAAI/bge-base-zh-v1.5
IMAGE_DIM = 512  # openai/clip-vit-base-patch32

_COSINE_INDEX = {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}}


def connect() -> None:
    connections.connect(host=settings.milvus_host, port=settings.milvus_port)


def init_collections() -> None:
    connect()
    _ensure_text_collection()
    _ensure_image_collection()


def _ensure_text_collection() -> None:
    if utility.has_collection(TEXT_COLLECTION):
        return
    fields = [
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema("doc_id", DataType.INT64),
        FieldSchema("filename", DataType.VARCHAR, max_length=512),
        FieldSchema("page_num", DataType.INT64),
        FieldSchema("chunk_index", DataType.INT64),
        FieldSchema("content", DataType.VARCHAR, max_length=4096),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=TEXT_DIM),
    ]
    col = Collection(TEXT_COLLECTION, CollectionSchema(fields))
    col.create_index("embedding", _COSINE_INDEX)


def _ensure_image_collection() -> None:
    if utility.has_collection(IMAGE_COLLECTION):
        return
    fields = [
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema("doc_id", DataType.INT64),
        FieldSchema("filename", DataType.VARCHAR, max_length=512),
        FieldSchema("page_num", DataType.INT64),
        FieldSchema("image_path", DataType.VARCHAR, max_length=1024),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=IMAGE_DIM),
    ]
    col = Collection(IMAGE_COLLECTION, CollectionSchema(fields))
    col.create_index("embedding", _COSINE_INDEX)


def get_text_collection() -> Collection:
    connect()
    col = Collection(TEXT_COLLECTION)
    col.load()
    return col


def get_image_collection() -> Collection:
    connect()
    col = Collection(IMAGE_COLLECTION)
    col.load()
    return col
