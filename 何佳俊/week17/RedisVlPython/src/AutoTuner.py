"""
AutoTuner — 阈值自动调优模块
============================
自动化管理缓存相似度阈值和路由置信度阈值，替代手动调整。

核心逻辑：
    1. 收集每次查询的相似度数据
    2. 建立相似度分布直方图
    3. 根据目标命中率自动计算最优阈值
    4. 支持按场景/路由分别设置不同阈值
    5. 定时重算最优阈值

设计原则：
    - 非侵入：作为可插拔组件，不修改现有缓存逻辑
    - 数据驱动：基于历史查询的相似度分布做决策
    - 可观测：所有调整记录可追踪可回滚
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ==================== 数据类 ====================


@dataclass
class TuningRecord:
    """单次调优记录。

    Attributes:
        timestamp: 调优时间戳。
        old_threshold: 调优前的阈值。
        new_threshold: 调优后的阈值。
        target_hit_rate: 目标命中率。
        actual_hit_rate: 当前实际命中率。
        samples: 参与计算的样本数。
        route: 路由名称（None 表示全局）。
    """

    timestamp: float
    old_threshold: float
    new_threshold: float
    target_hit_rate: float
    actual_hit_rate: float
    samples: int
    route: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "old_threshold": self.old_threshold,
            "new_threshold": self.new_threshold,
            "target_hit_rate": self.target_hit_rate,
            "actual_hit_rate": self.actual_hit_rate,
            "samples": self.samples,
            "route": self.route,
        }


@dataclass
class QueryRecord:
    """单次查询记录。

    Attributes:
        query: 查询文本。
        similarity: 与最相似缓存项的相似度。
        is_hit: 是否缓存命中（similarity >= threshold）。
        route: 所属路由名称。
        timestamp: 查询时间戳。
        cache_size: 查询时的缓存大小。
    """

    query: str
    similarity: float
    is_hit: bool
    route: Optional[str]
    timestamp: float
    cache_size: int


@dataclass
class TuningStats:
    """调优统计信息。

    Attributes:
        total_queries: 总查询数。
        total_hits: 总命中数。
        hit_rate: 命中率。
        similarity_mean: 平均相似度。
        similarity_median: 相似度中位数。
        similarity_std: 相似度标准差。
        similarity_p25: 25 百分位。
        similarity_p75: 75 百分位。
        current_threshold: 当前阈值。
        suggested_threshold: 建议阈值。
        tuning_count: 调优次数。
    """

    total_queries: int = 0
    total_hits: int = 0
    hit_rate: float = 0.0
    similarity_mean: float = 0.0
    similarity_median: float = 0.0
    similarity_std: float = 0.0
    similarity_p25: float = 0.0
    similarity_p75: float = 0.0
    current_threshold: float = 0.85
    suggested_threshold: Optional[float] = None
    tuning_count: int = 0


# ==================== 主类 ====================


class AutoTuner:
    """阈值自动调优器。

    通过收集历史查询的相似度数据，分析相似度分布，
    根据目标命中率自动计算出最优阈值。

    Attributes:
        target_hit_rate_min: 目标命中率下限。
        target_hit_rate_max: 目标命中率上限。
        min_samples: 触发调优的最小样本数。
        cooldown_seconds: 两次调优的最小间隔（秒）。
        global_threshold: 全局阈值。
        route_thresholds: 各路由的独立阈值。
    """

    def __init__(
        self,
        global_threshold: float = 0.85,
        target_hit_rate_min: float = 0.30,
        target_hit_rate_max: float = 0.60,
        min_samples: int = 50,
        cooldown_seconds: int = 3600,
        storage: Optional[Any] = None,
    ) -> None:
        """初始化 AutoTuner。

        Args:
            global_threshold: 初始全局阈值。
            target_hit_rate_min: 目标命中率下限（默认 30%）。
            target_hit_rate_max: 目标命中率上限（默认 60%）。
            min_samples: 触发调优的最小样本数。
            cooldown_seconds: 两次调优的最小间隔（秒）。
            storage: 可选的数据持久化存储（如 Redis client）。
                为 None 时仅内存记录。

        Raises:
            ValueError: 参数超出合法范围。
        """
        if not 0.0 <= global_threshold <= 1.0:
            raise ValueError(f"threshold 必须在 [0.0, 1.0] 范围内: {global_threshold}")
        if not 0.0 <= target_hit_rate_min < target_hit_rate_max <= 1.0:
            raise ValueError(
                f"目标命中率范围无效: [{target_hit_rate_min}, {target_hit_rate_max}]"
            )
        if min_samples < 1:
            raise ValueError(f"min_samples 必须 >= 1: {min_samples}")

        self.target_hit_rate_min = target_hit_rate_min
        self.target_hit_rate_max = target_hit_rate_max
        self.min_samples = min_samples
        self.cooldown_seconds = cooldown_seconds
        self.storage = storage

        # 阈值管理
        self.global_threshold = global_threshold
        self.route_thresholds: Dict[str, float] = {}

        # 查询记录（内存环形缓冲区，最多保留 10000 条）
        self._records: List[QueryRecord] = []
        self._max_records = 10000

        # 路由聚合
        self._route_records: Dict[str, List[QueryRecord]] = defaultdict(list)

        # 调优历史
        self._tuning_history: List[TuningRecord] = []

        # 上次调优时间
        self._last_tune_time: float = 0.0

        # Redis Key 前缀（使用 storage 时）
        self._KEY_TUNING_PREFIX = "autotune:"
        self._KEY_RECORDS = f"{self._KEY_TUNING_PREFIX}records"
        self._KEY_THRESHOLD = f"{self._KEY_TUNING_PREFIX}threshold"
        self._KEY_HISTORY = f"{self._KEY_TUNING_PREFIX}history"

        logger.info(
            "AutoTuner 初始化 | global_threshold=%.2f | "
            "target_hit_rate=[%.0f%%, %.0f%%] | min_samples=%d | cooldown=%ds",
            global_threshold,
            target_hit_rate_min * 100, target_hit_rate_max * 100,
            min_samples, cooldown_seconds,
        )

    # ==================== 公开 API ====================

    def record_query(
        self,
        query: str,
        similarity: float,
        is_hit: bool,
        route: Optional[str] = None,
    ) -> None:
        """记录一次查询结果。

        Args:
            query: 查询文本。
            similarity: 与最相似缓存项的相似度。
            is_hit: 是否缓存命中。
            route: 所属路由名称（可选）。
        """
        record = QueryRecord(
            query=query[:200],
            similarity=max(0.0, min(1.0, similarity)),
            is_hit=is_hit,
            route=route,
            timestamp=time.time(),
            cache_size=self._get_cache_size(),
        )

        # 内存存储（环形缓冲区）
        self._records.append(record)
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records:]

        # 按路由分组
        route_key = route or "_global"
        self._route_records[route_key].append(record)
        if len(self._route_records[route_key]) > self._max_records:
            self._route_records[route_key] = \
                self._route_records[route_key][-self._max_records:]

        # 持久化存储（如配置了 Redis）
        if self.storage:
            self._persist_record(record)

    def get_threshold(self, route: Optional[str] = None) -> float:
        """获取指定路由的当前阈值。

        Args:
            route: 路由名称。None 返回全局阈值。

        Returns:
            当前阈值 (0.0 ~ 1.0)。
        """
        if route and route in self.route_thresholds:
            return self.route_thresholds[route]
        return self.global_threshold

    def set_threshold(
        self,
        threshold: float,
        route: Optional[str] = None,
    ) -> None:
        """设置指定路由的阈值。

        Args:
            threshold: 新阈值 (0.0 ~ 1.0)。
            route: 路由名称。None 设置全局阈值。

        Raises:
            ValueError: 阈值超出范围。
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"阈值必须在 [0.0, 1.0] 范围内: {threshold}")

        if route:
            self.route_thresholds[route] = threshold
            logger.info("路由阈值已设置 | route=%s | threshold=%.2f", route, threshold)
        else:
            old = self.global_threshold
            self.global_threshold = threshold
            logger.info("全局阈值已更新 | %.2f → %.2f", old, threshold)

        # 持久化
        if self.storage:
            key = self._KEY_THRESHOLD
            if route:
                key = f"{self._KEY_THRESHOLD}:{route}"
            self.storage.set(key, str(threshold))

    def suggest_threshold(
        self,
        target_hit_rate: Optional[float] = None,
        route: Optional[str] = None,
    ) -> float:
        """根据历史数据建议最优阈值。

        算法：
            1. 收集指定路由的所有查询记录
            2. 按相似度降序排列
            3. 找到使命中率最接近 target_hit_rate 的阈值

        Args:
            target_hit_rate: 目标命中率。None 使用 target_hit_rate_max。
            route: 路由名称。None 使用全量数据。

        Returns:
            建议的阈值 (0.0 ~ 1.0)。
        """
        target = target_hit_rate or self.target_hit_rate_max
        records = self._get_records(route)

        if len(records) < self.min_samples:
            logger.debug(
                "样本不足，返回当前阈值 | samples=%d/%d",
                len(records), self.min_samples,
            )
            return self.get_threshold(route)

        # 按相似度降序排列
        similarities = sorted(
            [r.similarity for r in records],
            reverse=True,
        )

        # 在相似度分布中找到目标命中率对应的阈值
        n = len(similarities)
        target_count = int(n * target)
        target_count = max(1, min(target_count, n - 1))

        # 目标阈值 = 排序后第 target_count 个值的相似度
        suggested = similarities[target_count - 1]

        # 平滑：稍微降低阈值以提高命中率
        suggested = max(0.1, min(0.99, suggested * 0.98))

        logger.info(
            "阈值建议 | route=%s | target_hit_rate=%.0f%% | "
            "suggested=%.4f | current=%.4f | samples=%d",
            route or "global", target * 100,
            suggested, self.get_threshold(route), n,
        )

        return round(suggested, 4)

    def auto_tune(self, route: Optional[str] = None) -> Optional[TuningRecord]:
        """自动调优阈值。

        根据历史数据自动调整阈值，使命中率维持在目标范围内。

        Args:
            route: 路由名称。None 调优全局阈值。

        Returns:
            调优记录，如果未满足条件返回 None。
        """
        # 1. 检查冷却时间
        now = time.time()
        if now - self._last_tune_time < self.cooldown_seconds:
            logger.debug("冷却中，跳过调优 | 上次 %.0fs 前", now - self._last_tune_time)
            return None

        # 2. 检查样本数量
        records = self._get_records(route)
        if len(records) < self.min_samples:
            logger.debug(
                "样本不足，跳过调优 | route=%s | samples=%d/%d",
                route, len(records), self.min_samples,
            )
            return None

        # 3. 计算当前命中率
        current_hit_rate = sum(1 for r in records if r.is_hit) / len(records)
        current_threshold = self.get_threshold(route)

        # 4. 判断是否需要调优
        if self.target_hit_rate_min <= current_hit_rate <= self.target_hit_rate_max:
            logger.debug(
                "命中率在目标范围内，无需调整 | route=%s | hit_rate=%.1f%%",
                route, current_hit_rate * 100,
            )
            return None

        # 5. 计算目标命中率
        if current_hit_rate < self.target_hit_rate_min:
            # 命中率太低 → 降低阈值
            target = self.target_hit_rate_max
            direction = "下降"
        else:
            # 命中率太高 → 提高阈值（提高精度）
            target = self.target_hit_rate_min
            direction = "上升"

        # 6. 计算建议阈值
        new_threshold = self.suggest_threshold(target, route)

        # 7. 安全限制：单次变化不超过 0.1
        max_change = 0.1
        if abs(new_threshold - current_threshold) > max_change:
            if new_threshold > current_threshold:
                new_threshold = current_threshold + max_change
            else:
                new_threshold = current_threshold - max_change
            logger.debug("限制单次变化量 | 最大 %.2f", max_change)

        # 8. 应用新阈值
        old_threshold = current_threshold
        self.set_threshold(new_threshold, route)
        self._last_tune_time = now

        # 9. 记录调优
        record = TuningRecord(
            timestamp=now,
            old_threshold=old_threshold,
            new_threshold=new_threshold,
            target_hit_rate=target,
            actual_hit_rate=current_hit_rate,
            samples=len(records),
            route=route,
        )
        self._tuning_history.append(record)

        # 持久化调优历史
        if self.storage:
            self._persist_tuning_record(record)

        logger.info(
            "阈值自动调优 | route=%s | %s: %.4f → %.4f | "
            "current_hit=%.1f%% | target=%.0f%% | samples=%d",
            route or "global", direction,
            old_threshold, new_threshold,
            current_hit_rate * 100, target * 100,
            len(records),
        )

        return record

    def auto_tune_all(self) -> List[TuningRecord]:
        """调优所有阈值（全局 + 所有路由）。

        Returns:
            本次调优的记录列表。
        """
        records: List[TuningRecord] = []

        # 调优全局阈值
        global_record = self.auto_tune(route=None)
        if global_record:
            records.append(global_record)

        # 调优各路由阈值
        for route_key in list(self._route_records.keys()):
            # 过滤掉 _global
            if route_key == "_global":
                continue
            route_record = self.auto_tune(route=route_key)
            if route_record:
                records.append(route_record)

        return records

    def get_similarity_distribution(
        self,
        route: Optional[str] = None,
        bins: int = 20,
    ) -> Dict[str, Any]:
        """获取相似度分布数据。

        Args:
            route: 路由名称。None 使用全量数据。
            bins: 直方图分桶数。

        Returns:
            分布数据，包含直方图、百分位数等。
        """
        records = self._get_records(route)
        if not records:
            return {"samples": 0, "histogram": [], "percentiles": {}}

        similarities = [r.similarity for r in records]
        arr = np.array(similarities)

        # 直方图
        hist, edges = np.histogram(arr, bins=bins, range=(0.0, 1.0))

        histogram_data = []
        for i in range(len(hist)):
            histogram_data.append({
                "bin_start": round(float(edges[i]), 3),
                "bin_end": round(float(edges[i + 1]), 3),
                "count": int(hist[i]),
                "percentage": round(float(hist[i] / len(similarities) * 100), 2),
            })

        # 百分位数
        percentiles = {}
        for p in [5, 10, 25, 50, 75, 90, 95]:
            percentiles[str(p)] = round(float(np.percentile(arr, p)), 4)

        return {
            "samples": len(similarities),
            "mean": round(float(arr.mean()), 4),
            "median": round(float(np.median(arr)), 4),
            "std": round(float(arr.std()), 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "histogram": histogram_data,
            "percentiles": percentiles,
            "route": route,
        }

    def get_stats(self, route: Optional[str] = None) -> TuningStats:
        """获取调优统计信息。

        Args:
            route: 路由名称。None 使用全量数据。

        Returns:
            TuningStats 统计信息。
        """
        records = self._get_records(route)
        dist = self.get_similarity_distribution(route)

        total = len(records)
        hits = sum(1 for r in records if r.is_hit) if records else 0
        hit_rate = hits / total if total > 0 else 0.0

        current_threshold = self.get_threshold(route)
        suggested = self.suggest_threshold(
            target_hit_rate=self.target_hit_rate_max,
            route=route,
        )
        # 如果样本不足，suggest_threshold 返回当前阈值，不视为建议值
        if total < self.min_samples:
            suggested = None

        tuning_count = len([
            r for r in self._tuning_history
            if r.route == route or (route is None and r.route is None)
        ])

        return TuningStats(
            total_queries=total,
            total_hits=hits,
            hit_rate=round(hit_rate, 4),
            similarity_mean=dist.get("mean", 0.0),
            similarity_median=dist.get("median", 0.0),
            similarity_std=dist.get("std", 0.0),
            similarity_p25=dist.get("percentiles", {}).get("25", 0.0),
            similarity_p75=dist.get("percentiles", {}).get("75", 0.0),
            current_threshold=current_threshold,
            suggested_threshold=suggested,
            tuning_count=tuning_count,
        )

    def get_tuning_history(
        self,
        limit: int = 100,
        route: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """获取调优历史。

        Args:
            limit: 返回的最大记录数。
            route: 按路由过滤。

        Returns:
            调优记录字典列表。
        """
        records = self._tuning_history
        if route:
            records = [r for r in records if r.route == route]
        return [r.to_dict() for r in records[-limit:]]

    def reset(self) -> None:
        """重置所有记录和调优历史。"""
        self._records.clear()
        self._route_records.clear()
        self._tuning_history.clear()
        self._last_tune_time = 0.0
        if self.storage:
            self.storage.delete(self._KEY_RECORDS)
        logger.info("AutoTuner 已重置")

    # ==================== 内部方法 ====================

    def _get_records(self, route: Optional[str] = None) -> List[QueryRecord]:
        """获取指定路由的查询记录。

        Args:
            route: 路由名称。None 返回全量数据。

        Returns:
            查询记录列表。
        """
        if route is None:
            return self._records
        route_key = route
        return self._route_records.get(route_key, [])

    def _get_cache_size(self) -> int:
        """获取当前缓存大小。

        Returns:
            缓存条目数（0 表示未知）。
        """
        return len(self._records)

    def _persist_record(self, record: QueryRecord) -> None:
        """持久化查询记录到 Redis。

        Args:
            record: 查询记录。
        """
        if not self.storage:
            return
        try:
            data = {
                "query": record.query,
                "similarity": record.similarity,
                "is_hit": record.is_hit,
                "route": record.route,
                "timestamp": record.timestamp,
            }
            self.storage.lpush(self._KEY_RECORDS, json.dumps(data))
            self.storage.ltrim(self._KEY_RECORDS, 0, self._max_records - 1)
        except Exception as e:
            logger.debug("持久化查询记录失败: %s", e)

    def _persist_tuning_record(self, record: TuningRecord) -> None:
        """持久化调优记录到 Redis。

        Args:
            record: 调优记录。
        """
        if not self.storage:
            return
        try:
            self.storage.rpush(self._KEY_HISTORY, json.dumps(record.to_dict()))
        except Exception as e:
            logger.debug("持久化调优记录失败: %s", e)

    def __repr__(self) -> str:
        return (
            f"AutoTuner(global_threshold={self.global_threshold:.2f}, "
            f"routes={len(self.route_thresholds)}, "
            f"records={len(self._records)}, "
            f"tunings={len(self._tuning_history)})"
        )
