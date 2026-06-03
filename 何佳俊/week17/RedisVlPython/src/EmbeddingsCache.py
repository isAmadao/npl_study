"""
EmbeddingsCache — 嵌入向量缓存引擎
=====================================
基于 Redis（低版本兼容）+ Milvus 的双后端向量存储引擎。

架构说明：
    采用双层存储架构：
    - Redis：存储向量元数据、字符串内容、缓存统计信息
            兼容 Redis 6.x/7.x，不依赖 Redis Stack 模块
    - Milvus：存储高维浮点向量，提供高效的近似最近邻（ANN）搜索

    当 Milvus 不可用时，自动降级为 Redis 纯内存模式，
    使用 Redis 的 Sorted Set 进行暴力搜索（适合小规模数据）。

关键设计：
    1. 向量序列化：使用 numpy 的二进制格式存入 Redis String
    2. 元数据关联：Redis Hash 存储 metadata，通过 key 与 Milvus ID 关联
    3. TTL 管理：Redis 原生过期机制，Milvus 端定期同步清理
    4. LRU 淘汰：Redis 的 allkeys-lru 策略（需在 redis.conf 中配置）
"""

import base64
import json
import logging
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import redis as redis_client
    from redis import Redis
except ImportError:
    redis_client = None
    Redis = None  # type: ignore

try:
    from pymilvus import MilvusClient, utility
except ImportError:
    MilvusClient = None  # type: ignore
    utility = None  # type: ignore

from .config import Settings

logger = logging.getLogger(__name__)


class EmbeddingsCacheError(Exception):
    """EmbeddingsCache 基础异常。"""


class ConnectionError(EmbeddingsCacheError):
    """Redis/Milvus 连接异常。"""


class SerializationError(EmbeddingsCacheError):
    """向量序列化/反序列化异常。"""


class NotFoundError(EmbeddingsCacheError):
    """键或向量不存在。"""


class EmbeddingsCache:
    """嵌入向量缓存引擎。

    统一管理 Redis 和 Milvus 的向量读写操作。
    支持向量的增删改查、批量操作、TTL 过期和统计监控。

    Attributes:
        dim: 向量维度。
        redis: Redis 客户端实例。
        milvus_available: Milvus 是否可用。
        milvus_client: MilvusClient 实例（可用时）。
        milvus_collection: Milvus 集合名称。
    """

    # Redis Key 前缀常量
    _KEY_PREFIX_VECTOR = "vec:"          # 原始向量数据
    _KEY_PREFIX_META = "meta:"           # 元数据 Hash
    _KEY_PREFIX_ID_MAP = "idmap:"        # key -> Milvus ID 映射
    _KEY_PREFIX_STATS = "stats:"         # 统计信息
    _KEY_CACHE_HITS = "stats:cache_hits"
    _KEY_CACHE_MISSES = "stats:cache_misses"

    def __init__(
        self,
        settings: Optional[Settings] = None,
        redis_client: Optional["Redis"] = None,
        milvus_uri: Optional[str] = None,
        milvus_host: Optional[str] = None,
        milvus_port: Optional[int] = None,
        embedding_dim: Optional[int] = None,
        skip_milvus: bool = False,
    ) -> None:
        """初始化 EmbeddingsCache。

        Args:
            settings: 配置对象，为 None 时使用默认配置。
            redis_client: 外部传入的 Redis 客户端实例。
                         为 None 时根据 settings 自动创建。
            milvus_uri: Milvus 连接 URI。
                        "milvus.db" → 嵌入式 Milvus Lite（推荐开发用）
                        "http://localhost:19530" → 远程 Milvus
                        为 None 时使用 milvus_host:port 构建。
            milvus_host: Milvus 主机地址（覆盖 settings，仅旧 API）。
            milvus_port: Milvus 端口（覆盖 settings）。
            embedding_dim: 向量维度（覆盖 settings）。
            skip_milvus: 跳过 Milvus 初始化（测试环境用）。

        Raises:
            ConnectionError: Redis 连接失败时抛出。
        """
        self._settings = settings or Settings()
        self.dim = embedding_dim or self._settings.embedding.dim

        # ----- Redis 初始化 -----
        self.redis = redis_client or self._create_redis_client()
        self._validate_redis_connection()

        # ----- Milvus 初始化（可选） -----
        self.milvus_available = False
        self.milvus_client: Optional["MilvusClient"] = None
        self.milvus_collection: str = self._settings.milvus.collection
        if not skip_milvus:
            self._init_milvus(milvus_uri, milvus_host, milvus_port)

        logger.info(
            "EmbeddingsCache 初始化完成 | dim=%d | redis=connected | milvus=%s",
            self.dim,
            "connected" if self.milvus_available else "unavailable (fallback to redis)",
        )

    # ==================== 公开 API ====================

    def get(self, key: str) -> Optional[np.ndarray]:
        """获取指定 key 的嵌入向量。

        Args:
            key: 向量的唯一标识键。

        Returns:
            如果存在返回 numpy 向量数组，否则返回 None。

        Raises:
            ConnectionError: Redis 连接异常。
            SerializationError: 向量反序列化失败。
        """
        try:
            data = self.redis.get(f"{self._KEY_PREFIX_VECTOR}{key}")
            if data is None:
                return None
            return self._deserialize_vector(data)
        except redis_client.RedisError as e:
            raise ConnectionError(f"Redis 读取失败: {e}") from e
        except Exception as e:
            raise SerializationError(f"向量反序列化失败: {e}") from e

    def search_similar(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
        key_set: Optional[set] = None,
    ) -> List[Tuple[float, str]]:
        """近似搜索与查询向量最相似的 key。

        优先使用 Milvus ANN 搜索，不可用时降级为 Redis 暴力搜索。

        Args:
            query_vector: 查询向量（应已归一化）。
            top_k: 返回的最相似结果数。
            threshold: 相似度下限。
            key_set: 可选，限制搜索范围（仅 Redis 模式有效）。

        Returns:
            [(similarity, key)] 列表，按相似度降序排列。

        Raises:
            ConnectionError: Redis/Milvus 连接异常。
        """
        # ── 方案 A: Milvus ANN 搜索 ──
        if self.milvus_available and self.milvus_client is not None:
            return self._search_milvus(query_vector, top_k, threshold)

        # ── 方案 B: Redis 暴力搜索 ──
        return self._search_redis_bruteforce(query_vector, top_k, threshold, key_set)

    # ────────── 内部搜索实现 ──────────

    def _search_milvus(
        self,
        query_vector: np.ndarray,
        top_k: int,
        threshold: float,
    ) -> List[Tuple[float, str]]:
        """通过 Milvus ANN 索引搜索。

        Args:
            query_vector: 查询向量（归一化，IP 距离 = 余弦相似度）。
            top_k: 返回数量。
            threshold: 相似度下限。

        Returns:
            [(相似度, key)] 列表。
        """
        if self.milvus_client is None:
            return []

        try:
            results = self.milvus_client.search(
                collection_name=self.milvus_collection,
                data=[query_vector.tolist()],
                limit=top_k,
                output_fields=["key"],
            )
        except Exception as e:
            logger.error("Milvus ANN 搜索失败，降级到 Redis: %s", e)
            return self._search_redis_bruteforce(query_vector, top_k, threshold, None)

        matched: List[Tuple[float, str]] = []
        if not results or not results[0]:
            return matched

        for hit in results[0]:
            distance = hit.get("distance", 0.0)
            # IP 度量下，归一化向量的内积 = 余弦相似度
            similarity = max(0.0, min(1.0, float(distance)))
            if similarity < threshold:
                continue
            entity = hit.get("entity", {})
            key = entity.get("key") if entity else None
            if key:
                matched.append((similarity, str(key)))

        return matched

    def _search_redis_bruteforce(
        self,
        query_vector: np.ndarray,
        top_k: int,
        threshold: float,
        key_set: Optional[set] = None,
    ) -> List[Tuple[float, str]]:
        """通过 Redis 全量加载 + Python 暴力搜索。

        Args:
            query_vector: 查询向量。
            top_k: 返回数量。
            threshold: 相似度下限。
            key_set: 限制搜索范围的 key 集合。

        Returns:
            [(相似度, key)] 列表。
        """
        try:
            if key_set is not None:
                keys = list(key_set)
            else:
                cursor = 0
                keys: List[str] = []
                while True:
                    cursor, batch = self.redis.scan(
                        cursor, match=f"{self._KEY_PREFIX_VECTOR}*", count=200
                    )
                    keys.extend(
                        k.decode() if isinstance(k, bytes) else k
                        for k in batch
                    )
                    if cursor == 0:
                        break
                # 去掉前缀
                raw_keys = [
                    k[len(self._KEY_PREFIX_VECTOR):]
                    for k in keys
                    if k.startswith(self._KEY_PREFIX_VECTOR)
                ]
                keys = raw_keys

            if not keys:
                return []

            vectors = self.batch_get(keys)
            results: List[Tuple[float, str]] = []
            for key, vec in zip(keys, vectors):
                if vec is None:
                    continue
                sim = self._cosine_similarity(query_vector, vec)
                if sim >= threshold:
                    results.append((sim, key))

            results.sort(key=lambda x: x[0], reverse=True)
            return results[:top_k]

        except redis_client.RedisError as e:
            raise ConnectionError(f"Redis 搜索失败: {e}") from e

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """计算两个归一化向量的余弦相似度。"""
        sim = float(np.dot(a, b))
        return max(0.0, min(1.0, sim))

    def set(
        self,
        key: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> bool:
        """存储嵌入向量及其元数据。

        Args:
            key: 向量的唯一标识键。
            vector: numpy 向量数组，维度必须与 self.dim 一致。
            metadata: 关联的元数据字典。
            ttl: 过期时间（秒），为 None 时使用配置的默认 TTL。

        Returns:
            是否成功写入。

        Raises:
            ConnectionError: Redis/Milvus 连接异常。
            ValueError: 向量维度不匹配。
            SerializationError: 向量序列化失败。
        """
        self._validate_vector(vector)

        ttl = ttl or self._settings.cache.default_ttl
        metadata = metadata or {}

        try:
            # 1. 序列化向量并写入 Redis
            serialized = self._serialize_vector(vector)
            pipe = self.redis.pipeline()
            pipe.set(f"{self._KEY_PREFIX_VECTOR}{key}", serialized)

            # 2. 存储元数据（逐字段写入，兼容低版本 Redis）
            if metadata:
                mk = f"{self._KEY_PREFIX_META}{key}"
                for mf, mv in metadata.items():
                    pipe.hset(mk, mf,
                              json.dumps(mv) if not isinstance(mv, str) else mv)
                if ttl > 0:
                    pipe.expire(mk, ttl)

            # 3. 设置 TTL
            if ttl > 0:
                pipe.expire(f"{self._KEY_PREFIX_VECTOR}{key}", ttl)

            pipe.execute()

            # 4. 写入 Milvus（可用时）
            if self.milvus_available and self.milvus_client is not None:
                self._insert_to_milvus(key, vector, metadata, ttl)

            return True

        except redis_client.RedisError as e:
            raise ConnectionError(f"Redis 写入失败: {e}") from e
        except Exception as e:
            raise SerializationError(f"向量序列化失败: {e}") from e

    def delete(self, key: str) -> bool:
        """删除指定 key 的向量和元数据。

        Args:
            key: 要删除的向量键。

        Returns:
            是否成功删除（key 不存在时返回 False）。
        """
        try:
            # 删除 Redis 数据
            pipe = self.redis.pipeline()
            pipe.delete(f"{self._KEY_PREFIX_VECTOR}{key}")
            pipe.delete(f"{self._KEY_PREFIX_META}{key}")
            results = pipe.execute()

            # 删除 Milvus 数据
            if self.milvus_available and self.milvus_client is not None:
                id_key = f"{self._KEY_PREFIX_ID_MAP}{key}"
                milvus_id = self.redis.get(id_key)
                if milvus_id is not None:
                    self.milvus_client.delete(
                        self.milvus_collection,
                        ids=[int(milvus_id)],
                    )
                    self.redis.delete(id_key)

            deleted = results[0] > 0 or results[1] > 0
            if deleted:
                logger.debug("删除 key: %s", key)
            return deleted

        except redis_client.RedisError as e:
            raise ConnectionError(f"Redis 删除失败: {e}") from e

    def exists(self, key: str) -> bool:
        """检查 key 是否存在。

        Args:
            key: 要检查的键。

        Returns:
            key 是否存在。
        """
        try:
            return bool(self.redis.exists(f"{self._KEY_PREFIX_VECTOR}{key}"))
        except redis_client.RedisError as e:
            raise ConnectionError(f"Redis 检查失败: {e}") from e

    def clear(self) -> bool:
        """清空所有缓存数据（慎用）。

        Returns:
            是否成功清空。
        """
        try:
            count = 0
            for prefix in [self._KEY_PREFIX_VECTOR, self._KEY_PREFIX_META,
                           self._KEY_PREFIX_ID_MAP]:
                cursor = 0
                while True:
                    cursor, keys = self.redis.scan(cursor, match=f"{prefix}*", count=100)
                    if keys:
                        count += self.redis.delete(*keys)
                    if cursor == 0:
                        break

            # 清空 Milvus 集合
            if self.milvus_available and self.milvus_client is not None:
                self.milvus_client.drop_collection(self.milvus_collection)

            logger.info("缓存已清空，共删除 %d 个 key", count)
            return True

        except redis_client.RedisError as e:
            raise ConnectionError(f"Redis 清空失败: {e}") from e

    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """获取指定 key 的元数据。

        Args:
            key: 向量键。

        Returns:
            元数字典，不存在时返回 None。
        """
        try:
            data = self.redis.hgetall(f"{self._KEY_PREFIX_META}{key}")
            if not data:
                return None
            return self._decode_metadata(data)
        except redis_client.RedisError as e:
            raise ConnectionError(f"Redis 读取元数据失败: {e}") from e

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息。

        Returns:
            统计信息字典，包含缓存大小、命中率等。
        """
        try:
            # info("keyspace") 在 fakeredis 中不可用，捕获异常降级
            try:
                self.redis.info("keyspace")
            except Exception:
                pass

            vector_keys = self.redis.keys(f"{self._KEY_PREFIX_VECTOR}*") or []
            meta_keys = self.redis.keys(f"{self._KEY_PREFIX_META}*") or []

            hits = int(self.redis.get(self._KEY_CACHE_HITS) or 0)
            misses = int(self.redis.get(self._KEY_CACHE_MISSES) or 0)
            total = hits + misses
            hit_rate = hits / total if total > 0 else 0.0

            return {
                "vector_count": len(vector_keys),
                "metadata_count": len(meta_keys),
                "cache_hits": hits,
                "cache_misses": misses,
                "hit_rate": round(hit_rate, 4),
                "total_requests": total,
                "milvus_available": self.milvus_available,
                "dimension": self.dim,
            }

        except redis_client.RedisError as e:
            raise ConnectionError(f"获取统计信息失败: {e}") from e

    def batch_get(self, keys: List[str]) -> List[Optional[np.ndarray]]:
        """批量获取多个向量。

        Args:
            keys: 键列表。

        Returns:
            与 keys 对应的向量列表，不存在的键对应 None。
        """
        try:
            redis_keys = [f"{self._KEY_PREFIX_VECTOR}{k}" for k in keys]
            data_list = self.redis.mget(redis_keys)
            results: List[Optional[np.ndarray]] = []
            for data in data_list:
                if data is None:
                    results.append(None)
                else:
                    results.append(self._deserialize_vector(data))
            return results
        except redis_client.RedisError as e:
            raise ConnectionError(f"Redis 批量读取失败: {e}") from e

    def batch_set(
        self,
        items: List[Tuple[str, np.ndarray, Optional[Dict[str, Any]]]],
        ttl: Optional[int] = None,
    ) -> bool:
        """批量写入多个向量。

        Args:
            items: (key, vector, metadata) 元组列表。
            ttl: 过期时间（秒）。

        Returns:
            是否全部写入成功。
        """
        ttl = ttl or self._settings.cache.default_ttl
        try:
            pipe = self.redis.pipeline()
            for key, vector, metadata in items:
                self._validate_vector(vector)
                serialized = self._serialize_vector(vector)
                pipe.set(f"{self._KEY_PREFIX_VECTOR}{key}", serialized)
                if metadata:
                    mk = f"{self._KEY_PREFIX_META}{key}"
                    for mf, mv in metadata.items():
                        pipe.hset(mk, mf,
                                  json.dumps(mv) if not isinstance(mv, str) else mv)
                if ttl > 0:
                    pipe.expire(f"{self._KEY_PREFIX_VECTOR}{key}", ttl)
            pipe.execute()

            # 批量写入 Milvus
            if self.milvus_available and self.milvus_client is not None:
                for key, vector, metadata in items:
                    self._insert_to_milvus(key, vector, metadata or {}, ttl)

            return True
        except redis_client.RedisError as e:
            raise ConnectionError(f"Redis 批量写入失败: {e}") from e

    # ==================== 内部方法 ====================

    def _create_redis_client(self) -> "Redis":
        """创建 Redis 客户端连接。

        Returns:
            Redis 客户端实例。

        Raises:
            ConnectionError: Redis 库未安装或连接失败。
        """
        if Redis is None:
            raise ConnectionError(
                "redis 库未安装，请执行: pip install redis>=4.5.0"
            )
        try:
            config = self._settings.redis
            client = Redis.from_url(
                config.url,
                password=config.password or None,
                socket_timeout=config.socket_timeout,
                socket_connect_timeout=config.socket_connect_timeout,
                max_connections=config.max_connections,
                decode_responses=config.decode_responses,
            )
            return client
        except Exception as e:
            raise ConnectionError(f"Redis 连接创建失败: {e}") from e

    def _validate_redis_connection(self) -> None:
        """验证 Redis 连接是否正常。

        Raises:
            ConnectionError: PING 失败时抛出。
        """
        try:
            self.redis.ping()
        except redis_client.RedisError as e:
            raise ConnectionError(f"Redis 连接验证失败: {e}") from e

    def _init_milvus(
        self,
        uri: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> None:
        """初始化 Milvus 连接。

        支持两种方式：
          1. MilvusClient URI（推荐）:
             - "milvus.db" → 嵌入式 Milvus Lite（开发用）
             - "http://localhost:19530" → 远程 Milvus
          2. 旧版 host:port（向后兼容）

        Args:
            uri: Milvus URI（优先级最高）。
            host: Milvus 主机地址。
            port: Milvus 端口。
        """
        if MilvusClient is None:
            logger.warning("pymilvus 未安装，降级为 Redis 纯内存模式")
            return

        config = self._settings.milvus

        try:
            # 方式一：通过 URI 连接（推荐，支持 Milvus Lite）
            if uri:
                self.milvus_client = MilvusClient(uri=uri)
            # 方式二：通过 host:port 构建 URI
            elif host:
                self.milvus_client = MilvusClient(
                    uri=f"http://{host}:{port or config.port}"
                )
            # 方式三：从配置读取 host
            else:
                cfg_host = config.host
                cfg_port = config.port
                self.milvus_client = MilvusClient(
                    uri=f"http://{cfg_host}:{cfg_port}"
                )

            # 创建集合（如不存在）
            if not self.milvus_client.has_collection(self.milvus_collection):
                self._create_milvus_collection()

            self.milvus_available = True
            logger.info(
                "Milvus 连接成功 | collection=%s | dim=%d",
                self.milvus_collection, self.dim,
            )

        except Exception as e:
            logger.warning(
                "Milvus 连接失败，降级为 Redis 纯内存模式: %s", e,
            )
            self.milvus_available = False
            self.milvus_client = None

    def _create_milvus_collection(self) -> None:
        """在 Milvus 中创建向量集合与索引。"""
        if self.milvus_client is None:
            return

        config = self._settings.milvus
        self.milvus_client.create_collection(
            collection_name=self.milvus_collection,
            dimension=self.dim,
            metric_type=config.metric_type,
            auto_id=True,
            id_type="int",
        )

    def _insert_to_milvus(
        self,
        key: str,
        vector: np.ndarray,
        metadata: Dict[str, Any],
        ttl: int,
    ) -> None:
        """将向量插入 Milvus 集合。

        Args:
            key: 关联的 Redis key。
            vector: 向量数据。
            metadata: 元数据。
            ttl: TTL（仅用于记录，Milvus 本身不支持 TTL）。
        """
        if self.milvus_client is None:
            return

        try:
            data = [{
                "key": key,
                "vector": vector.tolist(),
            }]
            mr = self.milvus_client.insert(
                collection_name=self.milvus_collection,
                data=data,
            )
            # 保存 Milvus ID 到 Redis 的映射
            if mr.get("ids") and len(mr["ids"]) > 0:
                self.redis.set(
                    f"{self._KEY_PREFIX_ID_MAP}{key}",
                    str(mr["ids"][0]),
                )
        except Exception as e:
            logger.error("Milvus 插入失败 (key=%s): %s", key, e)

    def _serialize_vector(self, vector: np.ndarray) -> str:
        """将 numpy 向量序列化为 base64 字符串。

        使用 base64 编码确保在 decode_responses=True 的 Redis 连接中
        也能正确存储和读取。

        Args:
            vector: numpy 向量。

        Returns:
            base64 编码的字符串。

        Raises:
            SerializationError: 序列化失败。
        """
        try:
            return base64.b64encode(pickle.dumps(vector)).decode("ascii")
        except Exception as e:
            raise SerializationError(f"序列化失败: {e}") from e

    def _deserialize_vector(self, data: Union[str, bytes]) -> np.ndarray:
        """从 base64 字符串反序列化为 numpy 向量。

        Args:
            data: base64 编码的字符串或字节。

        Returns:
            numpy 向量。

        Raises:
            SerializationError: 反序列化失败。
        """
        try:
            if isinstance(data, str):
                data = data.encode("ascii")
            return pickle.loads(base64.b64decode(data))
        except Exception as e:
            raise SerializationError(f"反序列化失败: {e}") from e

    def _validate_vector(self, vector: np.ndarray) -> None:
        """验证向量维度和类型。

        Args:
            vector: 待验证的向量。

        Raises:
            ValueError: 向量维度或类型不匹配。
        """
        if not isinstance(vector, np.ndarray):
            raise ValueError(
                f"向量类型必须为 numpy.ndarray，当前: {type(vector)}"
            )
        if vector.ndim != 1:
            raise ValueError(
                f"向量必须是 1 维数组，当前维度: {vector.ndim}"
            )
        if vector.shape[0] != self.dim:
            raise ValueError(
                f"向量维度不匹配: 期望 {self.dim}，实际 {vector.shape[0]}"
            )

    @staticmethod
    def _decode_metadata(data: Dict[str, str]) -> Dict[str, Any]:
        """解码 Redis Hash 中的元数据。

        Args:
            data: Redis 返回的原始元数据字典。

        Returns:
            解码后的元数据字典。
        """
        result: Dict[str, Any] = {}
        for k, v in data.items():
            try:
                result[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                result[k] = v
        return result

    def __repr__(self) -> str:
        return (
            f"EmbeddingsCache(dim={self.dim}, "
            f"redis_url={self._settings.redis.url}, "
            f"milvus={'available' if self.milvus_available else 'unavailable'})"
        )
