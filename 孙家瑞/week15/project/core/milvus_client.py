"""Milvus 连接管理和 collection 创建。"""

from pymilvus import MilvusClient, DataType

from .config import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME


def create_client() -> MilvusClient:
    return MilvusClient(uri=f"tcp://{MILVUS_HOST}:{MILVUS_PORT}")


def ensure_collection(client: MilvusClient) -> None:
    if client.has_collection(COLLECTION_NAME):
        return

    schema = client.create_schema(enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("text_vector", DataType.FLOAT_VECTOR, dim=512)
    schema.add_field("clip_text_vector", DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field("clip_image_vector", DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("file_id", DataType.INT64)
    schema.add_field("file_name", DataType.VARCHAR, max_length=1024)
    schema.add_field("file_path", DataType.VARCHAR, max_length=2048)

    index_params = client.prepare_index_params()
    index_params.add_index("text_vector", index_type="IVF_FLAT", metric_type="IP", params={"nlist": 128})
    index_params.add_index("clip_text_vector", index_type="IVF_FLAT", metric_type="IP", params={"nlist": 128})
    index_params.add_index("clip_image_vector", index_type="IVF_FLAT", metric_type="IP", params={"nlist": 128})

    client.create_collection(
        COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
