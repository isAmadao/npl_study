"""
配置管理模块
=============
从 .env 文件加载配置项，提供类型安全的配置类。
使用 Pydantic Settings 进行配置校验。
"""

import os
from typing import Optional, Literal
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RedisConfig(BaseSettings):
    """Redis 连接配置（兼容低版本 Redis 6.x/7.x）。"""

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    url: str = Field("redis://localhost:6379/0", description="Redis 连接 URL")
    password: Optional[str] = Field(None, description="Redis 密码")
    socket_timeout: int = Field(5, description="Socket 超时时间（秒）")
    socket_connect_timeout: int = Field(5, description="连接超时时间（秒）")
    max_connections: int = Field(50, description="最大连接数")
    decode_responses: bool = Field(True, description="是否解码响应为字符串")


class MilvusConfig(BaseSettings):
    """Milvus 向量数据库连接配置。"""

    model_config = SettingsConfigDict(env_prefix="MILVUS_")

    uri: Optional[str] = Field(None, description="Milvus URI（milvus.db | http://host:port）")
    host: str = Field("localhost", description="Milvus 主机地址（URI 为空时使用）")
    port: int = Field(19530, description="Milvus 端口")
    collection: str = Field("cache_vectors", description="集合名称")
    metric_type: str = Field("IP", description="距离度量方式 (IP/COSINE)")


class EmbeddingConfig(BaseSettings):
    """Embedding 模型配置。"""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    provider: Literal["openai", "qwen", "sentence_transformers", "local"] = Field(
        "qwen", description="Embedding 提供方 (openai / qwen / sentence_transformers)"
    )
    model: str = Field("text-embedding-v2", description="模型名称")
    dim: int = Field(1536, description="向量维度")
    api_key: Optional[str] = Field(None, description="API Key")
    api_base: Optional[str] = Field(None, description="API 基础地址")


class CacheConfig(BaseSettings):
    """语义缓存配置。"""

    model_config = SettingsConfigDict(env_prefix="CACHE_")

    similarity_threshold: float = Field(0.85, ge=0.0, le=1.0, description="相似度阈值")
    default_ttl: int = Field(86400, ge=0, description="默认 TTL（秒）")
    max_size: int = Field(10000, ge=1, description="最大缓存条目数")
    eviction_policy: Literal["lru", "lfu", "ttl"] = Field("lru", description="淘汰策略")
    top_k: int = Field(5, ge=1, description="Top-K 搜索结果数")
    enable_stats: bool = Field(True, description="是否启用统计")


class RouterConfig(BaseSettings):
    """语义路由配置。"""

    model_config = SettingsConfigDict(env_prefix="ROUTER_")

    confidence_threshold: float = Field(0.75, ge=0.0, le=1.0, description="置信度阈值")
    default_route: str = Field("fallback", description="默认兜底路由")
    config_path: Optional[str] = Field(None, description="路由配置文件路径")


class LogConfig(BaseSettings):
    """日志配置。"""

    model_config = SettingsConfigDict(env_prefix="LOG_")

    level: str = Field("INFO", description="日志级别")
    format: Literal["text", "json"] = Field("text", description="日志格式")


class Settings(BaseSettings):
    """全局配置聚合。"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    redis: RedisConfig = RedisConfig()
    milvus: MilvusConfig = MilvusConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    cache: CacheConfig = CacheConfig()
    router: RouterConfig = RouterConfig()
    log: LogConfig = LogConfig()

    @model_validator(mode="after")
    def _validate_dimension_match(self) -> "Settings":
        """校验向量维度与索引类型兼容性。"""
        if self.embedding.dim <= 0:
            raise ValueError(
                f"向量维度必须为正数，当前: {self.embedding.dim}"
            )
        return self


# 全局单例配置
settings = Settings()
