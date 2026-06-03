"""
阈值自动调优示例 — 展示 AutoTuner 的完整功能
=============================================

演示内容：
  1. 模拟真实查询，收集相似度分布数据
  2. 分析相似度分布并建议最优阈值
  3. 自动调优缓存命中率到目标范围
  4. 支持按路由分别调优
  5. 查看调优历史和统计

运行方式：
  python examples/auto_tuning_demo.py

无需任何外部依赖，使用模拟数据展示调优流程。
"""

import logging
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.AutoTuner import AutoTuner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ==================== 模拟数据生成 ====================


def generate_similarity_samples(
    n: int,
    hit_rate: float,
    hit_similarity_mean: float = 0.88,
    miss_similarity_mean: float = 0.55,
    noise: float = 0.08,
) -> list:
    """生成模拟的相似度样本数据。

    生成的相似度分布中，约 hit_rate 比例的样本高于阈值（命中），
    其余低于阈值（未命中）。

    Args:
        n: 样本数。
        hit_rate: 期望命中率 (0.0 ~ 1.0)。
        hit_similarity_mean: 命中样本的相似度均值。
        miss_similarity_mean: 未命中样本的相似度均值。
        noise: 随机噪声标准差。

    Returns:
        (similarity, is_hit) 元组列表。
    """
    samples = []
    n_hits = int(n * hit_rate)
    n_misses = n - n_hits

    for _ in range(n_hits):
        sim = min(0.99, max(0.0, random.gauss(hit_similarity_mean, noise)))
        samples.append((sim, True))

    for _ in range(n_misses):
        sim = min(0.99, max(0.0, random.gauss(miss_similarity_mean, noise)))
        samples.append((sim, False))

    random.shuffle(samples)
    return samples


# ==================== 路由示例查询 ====================

QUERY_TEMPLATES = [
    "什么是{}", "{}怎么用", "{}的优缺点", "如何实现{}",
    "{}是什么", "{}有哪些特性", "{}和{}的区别",
    "为什么{}", "{}在哪里", "{}多少钱",
]

TOPICS = {
    "knowledge": ["Redis", "Milvus", "Docker", "Python", "Kubernetes",
                   "Elasticsearch", "MongoDB", "PostgreSQL", "Kafka", "Nginx"],
    "technical": ["部署失败", "报错排查", "性能优化", "配置问题",
                   "网络故障", "内存泄漏", "权限错误", "版本兼容"],
    "chitchat": ["你好", "再见", "谢谢", "很高兴", "今天天气",
                  "你是谁", "你能做什么", "早上好", "晚安", "哈哈"],
}


# ==================== 演示 1：基础用法 ====================


def demo_basic():
    """演示 1：AutoTuner 基础用法。"""
    print("\n" + "=" * 65)
    print("🎛️  演示 1：AutoTuner 基础用法")
    print("=" * 65)

    # 初始化调优器
    tuner = AutoTuner(
        global_threshold=0.85,
        target_hit_rate_min=0.30,   # 目标命中率 30%~60%
        target_hit_rate_max=0.60,
        min_samples=30,
        cooldown_seconds=0,  # 演示中无冷却
    )

    print(f"\n📋 初始配置:")
    print(f"   全局阈值: {tuner.global_threshold}")
    print(f"   目标命中率: {tuner.target_hit_rate_min:.0%} ~ {tuner.target_hit_rate_max:.0%}")
    print(f"   最小样本数: {tuner.min_samples}")

    # 模拟一些命中率偏低的查询（相似度分布偏低）
    print(f"\n📝 模拟 50 条低命中率查询...")
    samples = generate_similarity_samples(50, hit_rate=0.20)
    for sim, is_hit in samples:
        tuner.record_query("模拟查询", sim, is_hit)

    stats = tuner.get_stats()
    print(f"   当前命中率: {stats.hit_rate:.1%}")
    print(f"   当前阈值: {stats.current_threshold:.4f}")
    print(f"   建议阈值: {stats.suggested_threshold:.4f}")

    # 执行自动调优
    print(f"\n🔄 执行自动调优...")
    record = tuner.auto_tune()

    if record:
        print(f"   旧阈值: {record.old_threshold:.4f}")
        print(f"   新阈值: {record.new_threshold:.4f}")
        print(f"   目标命中率: {record.target_hit_rate:.0%}")
        print(f"   当前命中率: {record.actual_hit_rate:.1%}")
        direction = "↑ 上升" if record.new_threshold > record.old_threshold else "↓ 下降"
        print(f"   方向: {direction}")
    else:
        print(f"   无需调整（命中率已在目标范围内）")

    # 验证调整效果
    print(f"\n✅ 调整后阈值: {tuner.global_threshold:.4f}")
    new_stats = tuner.get_stats()
    print(f"   预计命中率: ~{new_stats.target_hit_rate:.0%}")

    return tuner


# ==================== 演示 2：相似度分布分析 ====================


def demo_distribution(tuner: AutoTuner):
    """演示 2：相似度分布分析。

    Args:
        tuner: AutoTuner 实例。
    """
    print("\n" + "=" * 65)
    print("📊 演示 2：相似度分布分析")
    print("=" * 65)

    # 添加更多样本
    print(f"\n📝 追加 150 条样本...")
    samples = generate_similarity_samples(150, hit_rate=0.45)
    for sim, is_hit in samples:
        tuner.record_query("分布分析查询", sim, is_hit)

    # 获取分布数据
    dist = tuner.get_similarity_distribution(bins=10)

    print(f"\n📊 相似度分布 (共 {dist['samples']} 条):")
    print(f"   均值:   {dist['mean']:.4f}")
    print(f"   中位数: {dist['median']:.4f}")
    print(f"   标准差: {dist['std']:.4f}")
    print(f"   范围:   [{dist['min']:.4f}, {dist['max']:.4f}]")

    print(f"\n📈 百分位数:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        val = dist.get("percentiles", {}).get(str(p), 0)
        bar = "█" * int(val * 50)
        print(f"   P{p:02d}: {val:.4f}  {bar}")

    print(f"\n📊 直方图:")
    for bucket in dist["histogram"]:
        bar = "█" * max(1, bucket["count"])
        print(f"   [{bucket['bin_start']:.2f}-{bucket['bin_end']:.2f}) "
              f"{bar} {bucket['count']} ({bucket['percentage']:.1f}%)")


# ==================== 演示 3：按路由调优 ====================


def demo_per_route():
    """演示 3：按路由分别调优阈值。"""
    print("\n" + "=" * 65)
    print("🧭 演示 3：按路由分别调优")
    print("=" * 65)

    tuner = AutoTuner(
        global_threshold=0.80,
        target_hit_rate_min=0.30,
        target_hit_rate_max=0.60,
        min_samples=20,
        cooldown_seconds=0,
    )

    # 不同路由有不同的相似度分布
    route_configs = {
        "knowledge": {"hit_rate": 0.35, "hit_mean": 0.82},
        "chitchat": {"hit_rate": 0.65, "hit_mean": 0.90},
        "technical": {"hit_rate": 0.15, "hit_mean": 0.72},
    }

    print(f"\n📝 模拟各路由查询...")
    for route, config in route_configs.items():
        n = 40
        samples = generate_similarity_samples(
            n,
            hit_rate=config["hit_rate"],
            hit_similarity_mean=config["hit_mean"],
        )
        for sim, is_hit in samples:
            tuner.record_query(f"[{route}] 查询", sim, is_hit, route=route)

        route_stats = tuner.get_stats(route=route)
        print(f"   {route}: {n} 条, 命中率 {route_stats.hit_rate:.0%}, "
              f"阈值 {tuner.get_threshold(route):.2f}")

    # 分别调优
    print(f"\n🔄 按路由自动调优...")
    records = tuner.auto_tune_all()

    print(f"\n📋 调优结果:")
    for record in records:
        route_name = record.route or "global"
        direction = "↑" if record.new_threshold > record.old_threshold else "↓"
        print(f"   {route_name}: {record.old_threshold:.4f} → "
              f"{record.new_threshold:.4f} {direction} "
              f"(命中率 {record.actual_hit_rate:.0%} → 目标 {record.target_hit_rate:.0%})")

    # 最终各路由阈值
    print(f"\n✅ 最终各路由阈值:")
    print(f"   全局: {tuner.global_threshold:.4f}")
    for route in route_configs:
        print(f"   {route}: {tuner.get_threshold(route):.4f}")


# ==================== 演示 4：自动规划调优周期 ====================


def demo_scheduling():
    """演示 4：定时调优与渐进式优化。"""
    print("\n" + "=" * 65)
    print("⏰ 演示 4：渐进式调优（模拟多轮）")
    print("=" * 65)

    tuner = AutoTuner(
        global_threshold=0.90,    # 初始阈值偏高
        target_hit_rate_min=0.40,
        target_hit_rate_max=0.60,
        min_samples=30,
        cooldown_seconds=0,
    )

    print(f"\n📋 初始状态:")
    print(f"   阈值: {tuner.global_threshold:.2f}")
    print(f"   目标命中率: 40%~60%")

    # 模拟多轮优化
    for round_num in range(5):
        # 生成这轮的数据（命中率在真实场景会逐渐变化）
        base_hit_rate = 0.15 + round_num * 0.10  # 从 15% 逐渐升到 55%
        samples = generate_similarity_samples(
            20,
            hit_rate=min(base_hit_rate, 0.55),
        )

        for sim, is_hit in samples:
            tuner.record_query(f"第{round_num + 1}轮查询", sim, is_hit)

        old_threshold = tuner.global_threshold
        record = tuner.auto_tune()

        if record:
            change = record.new_threshold - record.old_threshold
            print(f"\n   第 {round_num + 1} 轮:")
            print(f"     样本: {len(samples)} 条")
            print(f"     当前命中率: {record.actual_hit_rate:.0%}")
            print(f"     阈值: {record.old_threshold:.4f} → {record.new_threshold:.4f} "
                  f"({'↓' if change < 0 else '↑'})")
        else:
            print(f"\n   第 {round_num + 1} 轮: 命中率在目标范围内，无需调整")

    print(f"\n✅ 最终阈值: {tuner.global_threshold:.4f}")
    print(f"   (初始 0.90 → 调整后通过 {tuner.get_stats().tuning_count} 次优化)")


# ==================== 演示 5：阈值建议 vs 实际效果 ====================


def demo_threshold_comparison():
    """演示 5：不同阈值的命中率对比。"""
    print("\n" + "=" * 65)
    print("📈 演示 5：不同阈值的效果对比")
    print("=" * 65)

    tuner = AutoTuner(global_threshold=0.85, min_samples=10)

    # 生成基准样本
    samples = generate_similarity_samples(200, hit_rate=0.40)
    for sim, is_hit in samples:
        tuner.record_query("对比测试", sim, is_hit)

    dist = tuner.get_similarity_distribution()

    print(f"\n📊 基准数据: {dist['samples']} 条样本")
    print(f"   相似度均值: {dist['mean']:.4f}")
    print(f"   相似度中位数: {dist['median']:.4f}")

    print(f"\n📈 不同阈值的预期命中率:")
    print(f"   {'阈值':<10} {'预期命中率':<12} {'效果评估':<20}")
    print(f"   {'-' * 42}")

    test_thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    for t in test_thresholds:
        suggested = tuner.suggest_threshold(
            target_hit_rate=0.50, route=None
        )

        # 计算在该阈值下的命中率
        hit_count = sum(1 for s, _ in samples if s >= t)
        hit_rate = hit_count / len(samples)

        if t > dist["mean"] + dist["std"]:
            assessment = "⚪ 精度高，命中少"
        elif t < dist["mean"] - dist["std"]:
            assessment = "🔵 命中多，精度低"
        elif abs(t - suggested) < 0.03:
            assessment = "🟢 推荐值附近"
        else:
            assessment = "🟡 平衡区域"

        marker = " ◀ 建议" if abs(t - suggested) < 0.03 else ""
        print(f"   {t:<8.2f} {hit_rate:<10.1%} {assessment}{marker}")


# ==================== 主函数 ====================


def main():
    """运行所有 AutoTuner 演示。"""
    print("=" * 70)
    print("🎛️  阈值自动调优示例 — AutoTuner 功能演示")
    print("=" * 70)

    tuner = demo_basic()
    demo_distribution(tuner)
    print("\n")
    demo_per_route()
    print("\n")
    demo_scheduling()
    print("\n")
    demo_threshold_comparison()

    print("\n" + "=" * 70)
    print("✅ AutoTuner 演示完成!")
    print("   核心结论:")
    print("   1. 阈值过高 → 命中率低 → 需降低阈值")
    print("   2. 阈值过低 → 命中率高但精度低 → 需提高阈值")
    print("   3. 不同路由的相似度分布不同 → 需要独立阈值")
    print("   4. 推荐通过多轮渐进调优获得最佳效果")
    print("=" * 70)


if __name__ == "__main__":
    main()
