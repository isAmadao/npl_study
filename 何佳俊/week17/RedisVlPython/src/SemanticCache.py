"""
SemanticCache — 语义缓存模块
=============================
基于语义相似度匹配的 LLM 问答缓存系统。

核心逻辑：
    1. 新查询到来时，计算其 Embedding 向量
    2. 与缓存库中所有历史查询进行向量相似度（余弦相似度）搜索
    3. 最高相似度超过阈值 → 缓存命中，直接返回缓存的回答
    4. 最高相似度低于阈值 → 缓存未命中，需调用 LLM 生成新回答

相似度策略：
    - 余弦相似度（Cosine Similarity）：衡量向量方向的一致性
    - 内积（Inner Product）：当向量已归一化时等价于余弦相似度
    - 阈值默认 0.85，可根据业务场景动态调整

缓存淘汰策略：
    - LRU (Least Recently Used)：淘汰最久未访问的条目
    - LFU (Least Frequently Used)：淘汰访问频率最低的条目
    - TTL：基于时间的过期淘汰
"""

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .EmbeddingsCache import EmbeddingsCache

logger = logging.getLogger(__name__)


# ==================== 数据类定义 ====================


@dataclass
class CacheResult:
    """语义缓存查询结果。

    Attributes:
        question: 缓存的原始问题。
        answer: 缓存的回答。
        similarity: 查询与缓存问题的余弦相似度。
        cached_at: 缓存时间戳（秒）。
        metadata: 关联的元数据。
    """

    question: str
    answer: str
    similarity: float
    cached_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheStats:
    """缓存统计信息。

    Attributes:
        size: 当前缓存条目数。
        hits: 命中次数。
        misses: 未命中次数。
        hit_rate: 命中率 (0.0 ~ 1.0)。
        avg_similarity: 命中结果的平均相似度。
        avg_response_time: 缓存查询平均耗时（秒）。
    """

    size: int = 0
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    avg_similarity: float = 0.0
    avg_response_time: float = 0.0


# ==================== 异常定义 ====================


class SemanticCacheError(Exception):
    """SemanticCache 基础异常。"""


class EmbeddingError(SemanticCacheError):
    """Embedding 生成失败。"""


class SimilarityError(SemanticCacheError):
    """相似度计算异常。"""


# ==================== 主类 ====================


class SemanticCache:
    """语义缓存 — 基于 Embedding 相似度的 Q&A 缓存。

    通过将问题和回答及其 Embedding 存入缓存，实现语义级别的
    相似匹配。新查询到来时，计算 Embedding 并与缓存中的历史
    查询进行向量相似度搜索，匹配度超过阈值时直接返回缓存回答。

    Attributes:
        cache: EmbeddingsCache 实例（底层存储引擎）。
        similarity_threshold: 相似度阈值 (0.0 ~ 1.0)。
        embedding_model: Embedding 模型（可调用对象或 provider 名称）。
        top_k: Top-K 搜索结果数。
    """

    # Redis Key 前缀
    _KEY_PREFIX_QA = "qa:"           # Q&A 数据 Hash
    _KEY_INDEX_SET = "idx:questions"  # 所有缓存的 question key 集合
    _KEY_COUNTER = "counter"          # 自增 ID 计数器

    def __init__(
        self,
        cache: EmbeddingsCache,
        embedding_model: Any,
        similarity_threshold: float = 0.85,
        top_k: int = 5,
        enable_stats: bool = True,
    ) -> None:
        """初始化 SemanticCache。

        Args:
            cache: EmbeddingsCache 实例。
            embedding_model: Embedding 模型。
                可以是可调用对象（接受文本返回向量），
                也可以是字符串标识（如 "openai"、"sentence_transformers"）。
            similarity_threshold: 相似度阈值，范围 0.0 ~ 1.0。
            top_k: 向量搜索返回的最相似结果数。
            enable_stats: 是否启用命中率统计。

        Raises:
            ValueError: threshold 或 top_k 参数不合法。
        """
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold 必须在 [0.0, 1.0] 范围内，当前: {similarity_threshold}"
            )
        if top_k < 1:
            raise ValueError(f"top_k 必须 >= 1，当前: {top_k}")

        self.cache = cache
        self._embedding_func = self._resolve_embedding_model(embedding_model)
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.enable_stats = enable_stats

        # 运行时统计（内存计数，周期性同步到 Redis）
        self._hits = 0
        self._misses = 0
        self._total_similarity = 0.0
        self._total_response_time = 0.0
        self._query_count = 0

        logger.info(
            "SemanticCache 初始化完成 | threshold=%.2f | top_k=%d",
            similarity_threshold, top_k,
        )

    # ==================== 公开 API ====================

    def lookup(
        self,
        query: str,
        threshold: Optional[float] = None,
    ) -> Optional[CacheResult]:
        """根据语义相似度查找缓存的回答。

        计算查询文本的 Embedding，与缓存中所有历史问题进行
        向量相似度搜索，返回超过阈值的匹配结果。

        Args:
            query: 用户查询文本。
            threshold: 覆盖实例的相似度阈值，为 None 使用默认值。

        Returns:
            找到匹配项时返回 CacheResult，否则返回 None。

        Raises:
            EmbeddingError: Embedding 生成失败。
            ConnectionError: Redis/Milvus 连接异常（透传自 EmbeddingsCache）。
        """
        start_time = time.time()
        threshold = threshold or self.similarity_threshold

        try:
            # 1. 生成查询的 Embedding
            query_vector = self._get_embedding(query)

            # 2. 搜索相似缓存
            results = self._search_similar(query_vector, threshold, self.top_k)

            # 3. 更新统计
            self._query_count += 1
            elapsed = time.time() - start_time
            self._total_response_time += elapsed

            if not results:
                self._record_miss()
                return None

            # 4. 返回最佳匹配
            best = results[0]
            self._record_hit(best.similarity)
            logger.debug(
                "缓存命中 | similarity=%.4f | threshold=%.2f | query='%s'",
                best.similarity, threshold, query[:50],
            )
            return best

        except SemanticCacheError:
            raise
        except Exception as e:
            raise SemanticCacheError(f"缓存查询失败: {e}") from e

    def store(
        self,
        question: str,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> bool:
        """存储 Q&A 到语义缓存。

        计算问题的 Embedding，将问题和回答及其向量存入缓存。

        Args:
            question: 用户问题。
            answer: LLM 生成的回答。
            metadata: 关联元数据（如模型名称、token 消耗等）。
            ttl: 过期时间（秒），为 None 使用默认值。

        Returns:
            是否成功存储。

        Raises:
            EmbeddingError: Embedding 生成失败。
        """
        metadata = metadata or {}
        metadata.update({
            "cached_at": time.time(),
            "question": question,
        })

        try:
            # 1. 生成问题 Embedding
            question_vector = self._get_embedding(question)

            # 2. 生成唯一键
            key = self._generate_key(question)
            qa_key = f"{self._KEY_PREFIX_QA}{key}"

            # 3. 存储向量到 EmbeddingsCache
            self.cache.set(
                key=key,
                vector=question_vector,
                metadata=metadata,
                ttl=ttl,
            )

            # 4. 存储 Q&A 文本到 Redis
            pipe = self.cache.redis.pipeline()
            pipe.hset(qa_key, "question", question)
            pipe.hset(qa_key, "answer", answer)
            pipe.hset(qa_key, "cached_at", str(time.time()))
            self.cache.redis.sadd(self._KEY_INDEX_SET, key)
            if ttl is not None and ttl > 0:
                pipe.expire(qa_key, ttl)
            pipe.execute()

            logger.debug("缓存存储成功 | key=%s | question='%s'", key, question[:50])
            return True

        except Exception as e:
            raise SemanticCacheError(f"缓存存储失败: {e}") from e

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[CacheResult]:
        """搜索最相似的 K 个缓存结果。

        与 lookup 的区别：search 返回 top_k 个结果而非仅返回最佳匹配，
        且不判断阈值，用于分析缓存内容。

        Args:
            query: 用户查询文本。
            top_k: 返回的最相似结果数，为 None 使用实例默认值。

        Returns:
            按相似度降序排列的 CacheResult 列表，可能为空。
        """
        top_k = top_k or self.top_k

        try:
            query_vector = self._get_embedding(query)
            return self._search_similar(query_vector, threshold=0.0, top_k=top_k)
        except Exception as e:
            raise SemanticCacheError(f"缓存搜索失败: {e}") from e

    def invalidate(self, query: str) -> bool:
        """使指定查询对应的缓存失效。

        Args:
            query: 要失效的查询文本。

        Returns:
            是否存在并删除了缓存。
        """
        key = self._generate_key(query)
        try:
            # 删除向量
            self.cache.delete(key)
            # 删除 Q&A 数据
            qa_key = f"{self._KEY_PREFIX_QA}{key}"
            self.cache.redis.delete(qa_key)
            self.cache.redis.srem(self._KEY_INDEX_SET, key)
            logger.info("缓存已失效: key=%s", key)
            return True
        except Exception as e:
            raise SemanticCacheError(f"缓存失效失败: {e}") from e

    def update_threshold(self, threshold: float) -> None:
        """更新相似度阈值。

        Args:
            threshold: 新的相似度阈值 (0.0 ~ 1.0)。

        Raises:
            ValueError: 阈值超出范围。
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"阈值必须在 [0.0, 1.0] 范围内，当前: {threshold}")
        old = self.similarity_threshold
        self.similarity_threshold = threshold
        logger.info("相似度阈值已更新: %.2f → %.2f", old, threshold)

    def get_stats(self) -> CacheStats:
        """获取缓存统计信息。

        包含命中率、平均相似度、缓存大小等指标。

        Returns:
            CacheStats 统计信息对象。
        """
        try:
            embed_stats = self.cache.get_stats()
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            avg_sim = (
                self._total_similarity / self._hits if self._hits > 0 else 0.0
            )
            avg_resp = (
                self._total_response_time / self._query_count
                if self._query_count > 0 else 0.0
            )

            return CacheStats(
                size=embed_stats.get("vector_count", 0),
                hits=self._hits,
                misses=self._misses,
                hit_rate=round(hit_rate, 4),
                avg_similarity=round(avg_sim, 4),
                avg_response_time=round(avg_resp, 4),
            )
        except Exception as e:
            raise SemanticCacheError(f"获取统计信息失败: {e}") from e

    def clear(self) -> bool:
        """清空所有语义缓存。

        Returns:
            是否成功清空。
        """
        try:
            # 清空 Q&A Hash 数据
            cursor = 0
            while True:
                cursor, keys = self.cache.redis.scan(
                    cursor, match=f"{self._KEY_PREFIX_QA}*", count=100
                )
                if keys:
                    self.cache.redis.delete(*keys)
                if cursor == 0:
                    break

            # 清空索引集合
            self.cache.redis.delete(self._KEY_INDEX_SET)

            # 清空底层向量缓存
            self.cache.clear()

            # 重置内存统计
            self._hits = 0
            self._misses = 0
            self._total_similarity = 0.0
            self._total_response_time = 0.0
            self._query_count = 0

            logger.info("语义缓存已全部清空")
            return True
        except Exception as e:
            raise SemanticCacheError(f"清空缓存失败: {e}") from e

    # ==================== 内部方法 ====================

    def _get_embedding(self, text: str) -> np.ndarray:
        """生成文本的 Embedding 向量。

        Args:
            text: 输入文本。

        Returns:
            归一化后的 numpy 向量。

        Raises:
            EmbeddingError: Embedding 生成失败。
        """
        try:
            if callable(self._embedding_func):
                vector = self._embedding_func(text)
            else:
                raise EmbeddingError("Embedding 模型未正确配置")

            vector = np.asarray(vector, dtype=np.float32)
            if vector.ndim > 1:
                vector = vector.flatten()
            return self._normalize(vector)

        except Exception as e:
            raise EmbeddingError(f"Embedding 生成失败: {e}") from e

    def _search_similar(
        self,
        query_vector: np.ndarray,
        threshold: float,
        top_k: int,
    ) -> List[CacheResult]:
        """在缓存中搜索与查询向量相似的条目。

        优先使用 Milvus ANN 搜索，不可用时降级为 Redis 暴力搜索。

        Args:
            query_vector: 查询向量（已归一化）。
            threshold: 相似度阈值。
            top_k: 返回数量。

        Returns:
            匹配的 CacheResult 列表，按相似度降序排列。
        """
        # 使用 EmbeddingsCache 的 search_similar（优先 Milvus ANN）
        results = self.cache.search_similar(
            query_vector=query_vector,
            top_k=top_k,
            threshold=threshold,
            key_set={
                k.decode() if isinstance(k, bytes) else k
                for k in self.cache.redis.smembers(self._KEY_INDEX_SET)
            } or None,
        )

        # 组装 CacheResult
        cache_results: List[CacheResult] = []
        for similarity, key in results:
            qa_key = f"{self._KEY_PREFIX_QA}{key}"
            qa_data = self.cache.redis.hgetall(qa_key)
            if not qa_data:
                continue

            metadata = self.cache.get_metadata(key) or {}
            cache_results.append(CacheResult(
                question=self._decode_value(qa_data.get("question", "")),
                answer=self._decode_value(qa_data.get("answer", "")),
                similarity=round(float(similarity), 4),
                cached_at=float(
                    self._decode_value(qa_data.get("cached_at", "0"))
                ),
                metadata=metadata,
            ))

        return cache_results

    def _record_hit(self, similarity: float) -> None:
        """记录一次缓存命中。

        Args:
            similarity: 命中结果的相似度。
        """
        self._hits += 1
        self._total_similarity += similarity
        if self.enable_stats:
            try:
                self.cache.redis.incr(self.cache._KEY_CACHE_HITS)
            except Exception:
                pass

    def _record_miss(self) -> None:
        """记录一次缓存未命中。"""
        self._misses += 1
        if self.enable_stats:
            try:
                self.cache.redis.incr(self.cache._KEY_CACHE_MISSES)
            except Exception:
                pass

    def _generate_key(self, text: str) -> str:
        """根据文本生成唯一的缓存键。

        使用 SHA256 哈希确保键的唯一性和固定长度。

        Args:
            text: 输入文本。

        Returns:
            哈希后的键字符串（前 16 位十六进制）。
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _resolve_embedding_model(model: Any) -> Callable[[str], np.ndarray]:
        """解析 Embedding 模型配置。

        Args:
            model: 可调用对象或 provider 名称。

        Returns:
            接受文本返回向量的可调用对象。
        """
        if callable(model):
            return model

        if isinstance(model, str):
            provider = model.lower()
            if provider == "openai":
                return _create_openai_embedding_func()
            elif provider == "qwen":
                return _create_qwen_embedding_func()
            elif provider == "sentence_transformers":
                return _create_sentence_transformer_func()
            elif provider == "local":
                return _create_sentence_transformer_func()
            else:
                raise EmbeddingError(f"不支持的 Embedding provider: {provider}")

        raise EmbeddingError(
            f"Embedding 模型必须是可调用对象或已知的 provider 名称，"
            f"当前: {type(model)}"
        )

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """L2 归一化向量。

        Args:
            vector: 待归一化的向量。

        Returns:
            归一化后的单位向量。
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """计算两个向量的余弦相似度。

        Args:
            a: 向量 a（已归一化）。
            b: 向量 b（已归一化）。

        Returns:
            余弦相似度 [-1.0, 1.0]，已 clamp 到 [0.0, 1.0]。
        """
        # 归一化向量时，余弦相似度等于内积
        similarity = float(np.dot(a, b))
        # 由于浮点误差可能略超出 [-1, 1]，clamp 到合理范围
        return max(0.0, min(1.0, similarity))

    @staticmethod
    def _decode_value(value: Any) -> str:
        """解码 Redis 返回的值。

        Args:
            value: Redis 原始值（可能是 bytes 或 str）。

        Returns:
            解码后的字符串。
        """
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def __repr__(self) -> str:
        return (
            f"SemanticCache(threshold={self.similarity_threshold}, "
            f"top_k={self.top_k}, "
            f"hits={self._hits}, misses={self._misses})"
        )


# ==================== Embedding 辅助函数 ====================


def _create_openai_embedding_func() -> Callable[[str], np.ndarray]:
    """创建 OpenAI Embedding 调用函数。

    Returns:
        接受文本返回向量的函数。

    Raises:
        EmbeddingError: openai 库未安装或配置不完整。
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise EmbeddingError(
            "openai 库未安装，请执行: pip install openai>=1.0.0"
        )

    from .config import settings

    client = OpenAI(
        api_key=settings.embedding.api_key,
        base_url=settings.embedding.api_base,
    )
    model = settings.embedding.model

    def _embed(text: str) -> np.ndarray:
        try:
            resp = client.embeddings.create(input=text, model=model)
            return np.array(resp.data[0].embedding, dtype=np.float32)
        except Exception as e:
            raise EmbeddingError(f"OpenAI Embedding 调用失败: {e}") from e

    return _embed


def _create_sentence_transformer_func() -> Callable[[str], np.ndarray]:
    """创建本地 Sentence-Transformer Embedding 函数。

    Returns:
        接受文本返回向量的函数。

    Raises:
        EmbeddingError: sentence-transformers 库未安装。
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise EmbeddingError(
            "sentence-transformers 库未安装，请执行: "
            "pip install sentence-transformers>=2.2.0"
        )

    from .config import settings

    model_name = settings.embedding.model or "BAAI/bge-small-zh-v1.5"
    model = SentenceTransformer(model_name)

    def _embed(text: str) -> np.ndarray:
        try:
            return model.encode(text)
        except Exception as e:
            raise EmbeddingError(
                f"SentenceTransformer Embedding 失败: {e}"
            ) from e

    return _embed


def _create_qwen_embedding_func() -> Callable[[str], np.ndarray]:
    """创建 Qwen（通义千问）DashScope Embedding 调用函数。

    支持 DashScope API 方式调用 Qwen 的 Embedding 模型：
      - text-embedding-v1  (1536 维)
      - text-embedding-v2  (1536 维，推荐)

    也可以复用 OpenAI 兼容接口模式，将 api_base 指向 Qwen 的兼容端点，
    此时直接使用 openai provider 即可。

    Returns:
        接受文本返回向量的函数。

    Raises:
        EmbeddingError: dashscope 库未安装或配置不完整。
    """
    # 优先使用 DashScope SDK
    try:
        import dashscope
        from dashscope import TextEmbedding
    except ImportError:
        dashscope = None  # type: ignore
        TextEmbedding = None  # type: ignore

    from .config import settings

    api_key = settings.embedding.api_key or os.environ.get("DASHSCOPE_API_KEY")
    model = settings.embedding.model or "text-embedding-v2"

    # 方案 A：使用 DashScope SDK
    if dashscope is not None and api_key:
        dashscope.api_key = api_key

        def _embed_via_dashscope(text: str) -> np.ndarray:
            try:
                resp = TextEmbedding.call(
                    model=model,
                    input=text,
                )
                if resp.status_code != 200:
                    raise EmbeddingError(
                        f"DashScope API 错误 (status={resp.status_code}): {resp.message}"
                    )
                return np.array(
                    resp.output["embeddings"][0]["embedding"],
                    dtype=np.float32,
                )
            except EmbeddingError:
                raise
            except Exception as e:
                raise EmbeddingError(f"DashScope Embedding 调用失败: {e}") from e

        return _embed_via_dashscope

    # 方案 B：DashScope SDK 未安装但有 api_key → 使用 OpenAI 兼容接口访问
    if api_key:
        try:
            from openai import OpenAI
        except ImportError:
            raise EmbeddingError(
                "请安装依赖: pip install dashscope  或  pip install openai>=1.0.0"
            )

        # DashScope OpenAI 兼容端点
        base_url = settings.embedding.api_base or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        client = OpenAI(api_key=api_key, base_url=base_url)

        def _embed_via_openai_compat(text: str) -> np.ndarray:
            try:
                resp = client.embeddings.create(input=text, model=model)
                return np.array(resp.data[0].embedding, dtype=np.float32)
            except Exception as e:
                raise EmbeddingError(f"Qwen (OpenAI 兼容) Embedding 调用失败: {e}") from e

        return _embed_via_openai_compat

    # 方案 C：都没有 → 报错
    raise EmbeddingError(
        "Qwen Embedding 配置不完整。请设置 EMBEDDING_API_KEY 或 DASHSCOPE_API_KEY，"
        "或安装 dashscope 库: pip install dashscope"
    )
