from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite:///./multimodal_rag.db"

    milvus_host: str = "localhost"
    milvus_port: int = 19530

    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_parse: str = "document_parse"

    qwen_vl_api_key: str = ""
    qwen_vl_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_vl_model: str = "qwen-vl-max"

    pdf_storage_path: str = "./data/pdfs"
    parsed_storage_path: str = "./data/parsed"

    class Config:
        env_file = ".env"


settings = Settings()
