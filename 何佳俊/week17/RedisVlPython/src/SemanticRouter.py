"""
SemanticRouter — 语义路由模块
==============================
基于语义相似度的 Agent 查询意图路由系统。

功能：
    1. 通过示例查询定义路由意图类别
    2. 新查询到来时，计算 Embedding 并与各路由的示例查询进行相似度匹配
    3. 将查询路由到语义最匹配的处理流程
    4. 支持路由的运行时动态注册和配置热加载

路由匹配策略：
    - 每个路由注册多个示例查询，示例的 Embedding 均值作为路由中心向量
    - 新查询与各路由中心向量计算余弦相似度
    - 最高相似度 > confidence_threshold → 路由到该意图
    - 所有相似度均低于阈值 → 路由到 fallback 默认处理

应用场景：
    - 将用户问题路由到不同的 Agent 或工具
    - 区分"知识问答"、"代码生成"、"数据分析"等不同意图
    - 多 Agent 系统中的查询分发
"""

import base64
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .EmbeddingsCache import EmbeddingsCache

logger = logging.getLogger(__name__)


# ==================== 数据类定义 ====================


@dataclass
class RouteInfo:
    """路由注册信息。

    Attributes:
        name: 路由名称，唯一标识。
        description: 路由描述。
        examples: 示例查询列表。
        handler: 路由处理函数（可调用对象）。
        created_at: 注册时间戳。
    """

    name: str
    description: str = ""
    examples: List[str] = field(default_factory=list)
    handler: Optional[Callable] = None
    created_at: float = 0.0


@dataclass
class RouteResult:
    """路由决策结果。

    Attributes:
        route_name: 匹配的路由名称。
        confidence: 匹配置信度 (0.0 ~ 1.0)。
        query: 原始查询文本。
        handler: 路由的处理函数（可能为 None）。
        is_fallback: 是否使用了兜底路由。
        matched_example: 最匹配的示例查询（若有）。
    """

    route_name: str
    confidence: float
    query: str
    handler: Optional[Callable] = None
    is_fallback: bool = False
    matched_example: Optional[str] = None


@dataclass
class RouterStats:
    """路由统计信息。

    Attributes:
        total_queries: 总查询数。
        route_counts: 各路由被命中次数。
        avg_confidence: 平均置信度。
        fallback_rate: 兜底路由使用率。
    """

    total_queries: int = 0
    route_counts: Dict[str, int] = field(default_factory=dict)
    avg_confidence: float = 0.0
    fallback_rate: float = 0.0
    route_example_counts: Dict[str, int] = field(default_factory=dict)


# ==================== 异常定义 ====================


class SemanticRouterError(Exception):
    """SemanticRouter 基础异常。"""


class RouteNotFoundError(SemanticRouterError):
    """路由不存在。"""


class RouteConflictError(SemanticRouterError):
    """路由名称冲突。"""


class RouterConfigurationError(SemanticRouterError):
    """路由配置错误。"""


# ==================== 主类 ====================


class SemanticRouter:
    """语义路由 — 基于嵌入相似度的查询意图路由。

    将用户查询路由到语义最匹配的处理模块。通过示例查询
    定义每个路由的意图空间，使用向量相似度进行匹配。

    Attributes:
        cache: EmbeddingsCache 实例。
        embedding_func: Embedding 生成函数。
        confidence_threshold: 路由置信度阈值 (0.0 ~ 1.0)。
        default_route: 未匹配时的兜底路由名称。
        routes: 已注册的路由字典。
    """

    # Redis Key 前缀
    _KEY_PREFIX_ROUTE = "route:"           # 路由元数据
    _KEY_PREFIX_ROUTE_VEC = "routevec:"     # 路由示例向量
    _KEY_ROUTE_INDEX = "routes:index"       # 所有路由名称集合
    _KEY_STATS_PREFIX = "stats:route:"      # 路由命中次数统计

    def __init__(
        self,
        cache: EmbeddingsCache,
        embedding_func: Callable[[str], np.ndarray],
        confidence_threshold: float = 0.75,
        default_route: str = "fallback",
    ) -> None:
        """初始化 SemanticRouter。

        Args:
            cache: EmbeddingsCache 实例。
            embedding_func: Embedding 生成函数。
            confidence_threshold: 路由置信度阈值 (0.0 ~ 1.0)。
            default_route: 兜底路由名称。

        Raises:
            ValueError: 参数不合法。
        """
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold 必须在 [0.0, 1.0] 范围内，"
                f"当前: {confidence_threshold}"
            )

        self.cache = cache
        self._embedding_func = embedding_func
        self.confidence_threshold = confidence_threshold
        self.default_route = default_route

        # 内存中的路由注册表
        self.routes: Dict[str, RouteInfo] = {}

        # 路由中心向量缓存（避免重复计算）
        self._route_centroids: Dict[str, np.ndarray] = {}

        # 运行时统计
        self._total_queries = 0
        self._route_counts: Dict[str, int] = {}
        self._total_confidence = 0.0
        self._fallback_count = 0

        # 恢复持久化的路由
        self._load_persisted_routes()

        logger.info(
            "SemanticRouter 初始化完成 | threshold=%.2f | default=%s",
            confidence_threshold, default_route,
        )

    # ==================== 公开 API ====================

    def route(self, query: str) -> "RouteResult":
        """路由查询到最匹配的处理流程。

        计算查询文本的 Embedding，与所有已注册路由的中心向量
        进行相似度匹配，返回置信度最高的路由结果。

        Args:
            query: 用户查询文本。

        Returns:
            RouteResult 包含路由目标和匹配信息。

        Raises:
            SemanticRouterError: 路由处理失败。
        """
        start_time = time.time()

        try:
            # 1. 生成查询 Embedding
            query_vector = self._embed_text(query)

            # 2. 计算与各路由的相似度
            best_route: Optional[str] = None
            best_confidence = 0.0
            best_example: Optional[str] = None

            for route_name, centroid in self._route_centroids.items():
                confidence = self._cosine_similarity(query_vector, centroid)

                # 也检查与具体示例的最大匹配（提高准确率）
                example_confidence = self._best_example_match(
                    query_vector, route_name
                )
                effective_confidence = max(confidence, example_confidence)

                if effective_confidence > best_confidence:
                    best_confidence = effective_confidence
                    best_route = route_name

            # 3. 更新统计
            self._total_queries += 1
            self._total_confidence += best_confidence

            # 4. 判断是否匹配
            is_fallback = False
            if best_route is None or best_confidence < self.confidence_threshold:
                best_route = self.default_route
                best_confidence = 0.0
                is_fallback = True
                self._fallback_count += 1

            # 更新路由命中计数
            self._route_counts[best_route] = self._route_counts.get(best_route, 0) + 1

            # 持久化统计
            self._persist_route_stats(best_route)

            # 获取路由信息
            route_info = self.routes.get(best_route)
            handler = route_info.handler if route_info else None

            elapsed = time.time() - start_time
            logger.debug(
                "路由决策 | route=%s | confidence=%.4f | fallback=%s | time=%.1fms",
                best_route, best_confidence, is_fallback, elapsed * 1000,
            )

            return RouteResult(
                route_name=best_route,
                confidence=round(best_confidence, 4),
                query=query,
                handler=handler,
                is_fallback=is_fallback,
                matched_example=best_example,
            )

        except Exception as e:
            raise SemanticRouterError(f"路由处理失败: {e}") from e

    def register_route(
        self,
        name: str,
        examples: List[str],
        handler: Optional[Callable] = None,
        description: str = "",
    ) -> bool:
        """注册一个新的路由。

        注册后会立即计算示例向量的中心向量并持久化。

        Args:
            name: 路由名称（唯一）。
            examples: 该路由的示例查询列表。
            handler: 路由处理函数。
            description: 路由描述。

        Returns:
            是否注册成功。

        Raises:
            RouteConflictError: 路由名称已存在。
            RouterConfigurationError: 示例列表为空。
            SemanticRouterError: 存储失败。
        """
        if name in self.routes:
            raise RouteConflictError(f"路由已存在: {name}")
        if not examples:
            raise RouterConfigurationError(
                f"路由 '{name}' 的示例列表不能为空"
            )

        route = RouteInfo(
            name=name,
            description=description,
            examples=examples,
            handler=handler,
            created_at=time.time(),
        )

        try:
            # 1. 存入内存
            self.routes[name] = route

            # 2. 计算并缓存中心向量
            centroid = self._compute_centroid(examples)
            self._route_centroids[name] = centroid

            # 3. 持久化到 Redis
            self._persist_route(route, centroid)

            # 4. 存储每个示例的向量
            self._persist_examples(name, examples)

            logger.info(
                "路由已注册: %s | examples=%d | desc=%s",
                name, len(examples), description or "(无描述)",
            )
            return True

        except RouteConflictError:
            raise
        except Exception as e:
            # 注册失败时清理已添加的内存数据
            self.routes.pop(name, None)
            self._route_centroids.pop(name, None)
            raise SemanticRouterError(f"注册路由失败: {e}") from e

    def add_examples(self, route_name: str, examples: List[str]) -> bool:
        """为已注册的路由添加更多示例。

        添加示例后会自动重新计算路由中心向量。

        Args:
            route_name: 路由名称。
            examples: 新增的示例查询列表。

        Returns:
            是否更新成功。

        Raises:
            RouteNotFoundError: 路由不存在。
        """
        if route_name not in self.routes:
            raise RouteNotFoundError(f"路由不存在: {route_name}")

        self.routes[route_name].examples.extend(examples)

        # 重新计算中心向量
        centroid = self._compute_centroid(self.routes[route_name].examples)
        self._route_centroids[route_name] = centroid

        # 持久化更新
        self._persist_examples(route_name, examples)
        self._update_route_centroid(route_name, centroid)

        logger.info(
            "路由 '%s' 已添加 %d 个示例，当前共 %d 个",
            route_name, len(examples), len(self.routes[route_name].examples),
        )
        return True

    def remove_route(self, route_name: str) -> bool:
        """移除一个已注册的路由。

        Args:
            route_name: 要移除的路由名称。

        Returns:
            是否成功移除（路由不存在时返回 False）。

        Raises:
            RouteNotFoundError: 路由不存在。
        """
        if route_name not in self.routes:
            raise RouteNotFoundError(f"路由不存在: {route_name}")

        try:
            # 清理内存
            self.routes.pop(route_name, None)
            self._route_centroids.pop(route_name, None)
            self._route_counts.pop(route_name, None)

            # 清理 Redis
            pipe = self.cache.redis.pipeline()
            pipe.delete(f"{self._KEY_PREFIX_ROUTE}{route_name}")
            pipe.delete(f"{self._KEY_PREFIX_ROUTE_VEC}{route_name}")
            pipe.srem(self._KEY_ROUTE_INDEX, route_name)
            pipe.execute()

            # 清理每个示例的向量
            embed_keys = self.cache.redis.smembers(
                f"embidx:route:{route_name}"
            )
            if embed_keys:
                for key in embed_keys:
                    key_str = (
                        key.decode() if isinstance(key, bytes) else key
                    )
                    self.cache.delete(key_str)
                self.cache.redis.delete(f"embidx:route:{route_name}")

            logger.info("路由已移除: %s", route_name)
            return True

        except Exception as e:
            raise SemanticRouterError(f"移除路由失败: {e}") from e

    def list_routes(self) -> List[RouteInfo]:
        """列出所有已注册的路由。

        Returns:
            RouteInfo 列表。
        """
        return list(self.routes.values())

    def get_route_stats(self) -> RouterStats:
        """获取路由统计信息。

        Returns:
            RouterStats 统计信息。
        """
        total = self._total_queries
        avg_conf = (
            self._total_confidence / total if total > 0 else 0.0
        )
        fallback_rate = (
            self._fallback_count / total if total > 0 else 0.0
        )

        return RouterStats(
            total_queries=total,
            route_counts=dict(self._route_counts),
            avg_confidence=round(avg_conf, 4),
            fallback_rate=round(fallback_rate, 4),
            route_example_counts={
                name: len(info.examples)
                for name, info in self.routes.items()
            },
        )

    def reload_routes(self, config_path: Optional[str] = None) -> int:
        """从配置文件重新加载路由定义。

        读取 JSON/YAML 配置文件中的路由定义并注册/更新路由。
        此操作会保留现有路由，新增配置中的路由。

        Args:
            config_path: 配置文件路径。
                为 None 时使用配置中的 ROUTER_CONFIG_PATH。

        Returns:
            新增/更新的路由数量。

        Raises:
            RouterConfigurationError: 配置加载失败。
        """
        if config_path is None:
            from .config import settings
            config_path = settings.router.config_path

        if not config_path:
            logger.info("无路由配置文件，跳过热加载")
            return 0

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                if config_path.endswith(".json"):
                    config = json.load(f)
                else:
                    raise RouterConfigurationError(
                        f"不支持的配置文件格式: {config_path}（支持 .json）"
                    )

            routes_config = config.get("routes", [])
            count = 0
            for route_cfg in routes_config:
                name = route_cfg.get("name")
                if not name:
                    continue
                # 如果路由已存在则跳过（不覆盖）
                if name in self.routes:
                    continue
                self.register_route(
                    name=name,
                    examples=route_cfg.get("examples", []),
                    handler=None,
                    description=route_cfg.get("description", ""),
                )
                count += 1

            logger.info("路由配置热加载完成: 新增 %d 个路由", count)
            return count

        except FileNotFoundError:
            raise RouterConfigurationError(f"配置文件不存在: {config_path}")
        except json.JSONDecodeError as e:
            raise RouterConfigurationError(f"配置文件格式错误: {e}")
        except Exception as e:
            raise RouterConfigurationError(f"配置加载失败: {e}") from e

    # ==================== 内部方法 ====================

    def _embed_text(self, text: str) -> np.ndarray:
        """生成文本的 Embedding 向量。

        Args:
            text: 输入文本。

        Returns:
            归一化后的 numpy 向量。
        """
        vector = self._embedding_func(text)
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim > 1:
            vector = vector.flatten()
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    def _compute_centroid(self, examples: List[str]) -> np.ndarray:
        """计算多个示例查询的中心向量。

        对所有示例的 Embedding 取均值并归一化。

        Args:
            examples: 示例查询列表。

        Returns:
            归一化后的中心向量。
        """
        if not examples:
            return np.zeros(self.cache.dim, dtype=np.float32)

        vectors = [self._embed_text(ex) for ex in examples]
        centroid = np.mean(vectors, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        return centroid

    def _best_example_match(
        self, query_vector: np.ndarray, route_name: str
    ) -> float:
        """计算查询向量与路由所有示例的最佳匹配相似度。

        Args:
            query_vector: 查询向量（已归一化）。
            route_name: 路由名称。

        Returns:
            与最佳匹配示例的相似度。
        """
        try:
            # 从 Redis 获取该路由所有示例的向量
            example_keys = self.cache.redis.smembers(
                f"embidx:route:{route_name}"
            )
            if not example_keys:
                return 0.0

            best = 0.0
            for key in example_keys:
                key_str = (
                    key.decode() if isinstance(key, bytes) else key
                )
                vec = self.cache.get(key_str)
                if vec is not None:
                    sim = self._cosine_similarity(query_vector, vec)
                    if sim > best:
                        best = sim
            return best

        except Exception:
            return 0.0

    def _persist_route(
        self, route: RouteInfo, centroid: np.ndarray
    ) -> None:
        """将路由持久化到 Redis。

        Args:
            route: 路由信息。
            centroid: 路由中心向量。
        """
        pipe = self.cache.redis.pipeline()

        # 路由元数据（逐字段写入，兼容低版本 Redis）
        rk = f"{self._KEY_PREFIX_ROUTE}{route.name}"
        pipe.hset(rk, "name", route.name)
        pipe.hset(rk, "description", route.description)
        pipe.hset(rk, "created_at", str(route.created_at))
        pipe.hset(rk, "example_count", str(len(route.examples)))

        # 中心向量（base64 编码，兼容 decode_responses=True）
        pipe.set(
            f"{self._KEY_PREFIX_ROUTE_VEC}{route.name}",
            base64.b64encode(centroid.tobytes()).decode("ascii"),
        )

        # 添加到路由索引
        pipe.sadd(self._KEY_ROUTE_INDEX, route.name)

        pipe.execute()

    def _update_route_centroid(
        self, route_name: str, centroid: np.ndarray
    ) -> None:
        """更新路由中心向量。

        Args:
            route_name: 路由名称。
            centroid: 新的中心向量。
        """
        self.cache.redis.set(
            f"{self._KEY_PREFIX_ROUTE_VEC}{route_name}",
            base64.b64encode(centroid.tobytes()).decode("ascii"),
        )

    def _persist_examples(
        self, route_name: str, examples: List[str]
    ) -> None:
        """持久化路由的示例向量。

        Args:
            route_name: 路由名称。
            examples: 示例查询列表。
        """
        for idx, example in enumerate(examples):
            vector = self._embed_text(example)
            embed_key = f"routeex:{route_name}:{idx}"

            self.cache.set(
                key=embed_key,
                vector=vector,
                metadata={
                    "route": route_name,
                    "example": example,
                    "index": idx,
                },
            )
            self.cache.redis.sadd(
                f"embidx:route:{route_name}", embed_key
            )

    def _persist_route_stats(self, route_name: str) -> None:
        """持久化路由命中统计。

        Args:
            route_name: 路由名称。
        """
        try:
            self.cache.redis.hincrby(
                self._KEY_STATS_PREFIX + "hits", route_name, 1
            )
        except Exception:
            pass

    def _load_persisted_routes(self) -> None:
        """从 Redis 加载持久化的路由配置。

        在初始化时调用，恢复之前注册的路由。
        """
        try:
            route_names = self.cache.redis.smembers(self._KEY_ROUTE_INDEX)
            if not route_names:
                return

            for name_bytes in route_names:
                name = (
                    name_bytes.decode() if isinstance(name_bytes, bytes)
                    else name_bytes
                )

                # 读取元数据
                meta = self.cache.redis.hgetall(
                    f"{self._KEY_PREFIX_ROUTE}{name}"
                )
                if not meta:
                    continue

                # 读取中心向量（base64 解码）
                centroid_raw = self.cache.redis.get(
                    f"{self._KEY_PREFIX_ROUTE_VEC}{name}"
                )
                if centroid_raw:
                    centroid = np.frombuffer(
                        base64.b64decode(centroid_raw), dtype=np.float32
                    )
                    self._route_centroids[name] = centroid

                # 重建 RouteInfo
                route = RouteInfo(
                    name=name,
                    description=meta.get("description", ""),
                    examples=[],
                    created_at=float(meta.get("created_at", 0)),
                )

                # 从示例向量重建示例列表（仅加载数量用于统计）
                example_count = int(meta.get("example_count", 0))

                self.routes[name] = route

            logger.info(
                "已从 Redis 恢复 %d 个持久化路由", len(self.routes)
            )

        except Exception as e:
            logger.warning("加载持久化路由失败: %s", e)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """计算两个向量的余弦相似度。

        Args:
            a: 向量 a（已归一化）。
            b: 向量 b（已归一化）。

        Returns:
            余弦相似度 [0.0, 1.0]。
        """
        return max(0.0, min(1.0, float(np.dot(a, b))))

    def __repr__(self) -> str:
        return (
            f"SemanticRouter("
            f"threshold={self.confidence_threshold}, "
            f"default='{self.default_route}', "
            f"routes={len(self.routes)})"
        )
