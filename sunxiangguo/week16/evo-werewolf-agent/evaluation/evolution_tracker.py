"""进化效果追踪器

追踪自进化效果，记录每次进化前后的指标变化。
"""

from typing import Dict, Any, List, Optional
import json
import os
from datetime import datetime


class EvolutionTracker:
    """追踪自进化效果"""

    def __init__(self, data_dir: str = "data/evolution"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def _filepath(self) -> str:
        return os.path.join(self.data_dir, "evolution_history.json")

    def _load_history(self) -> List[Dict]:
        fp = self._filepath()
        if os.path.exists(fp):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save_history(self, history: List[Dict]):
        with open(self._filepath(), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    def record_evolution(
        self,
        role_type: str,
        version: int,
        metrics_before: Dict[str, float],
        metrics_after: Dict[str, float],
        evolution_advice: Optional[Dict] = None,
    ):
        """记录一次进化的效果

        Args:
            role_type: 角色类型
            version: 进化版本号
            metrics_before: 进化前指标
            metrics_after: 进化后指标
            evolution_advice: 进化建议（来自EvolutionAgent）
        """
        history = self._load_history()
        record = {
            "role": role_type,
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "before": metrics_before,
            "after": metrics_after,
            "improvement": self._calc_improvement(metrics_before, metrics_after),
        }
        if evolution_advice:
            record["advice_summary"] = evolution_advice.get("trend_analysis", "")
            record["priority"] = evolution_advice.get("priority", "")
        history.append(record)
        self._save_history(history)

    def get_evolution_report(self, role_type: str) -> Dict[str, Any]:
        """获取某个角色的进化报告

        Args:
            role_type: 角色类型

        Returns:
            进化报告
        """
        history = self._load_history()
        role_history = [h for h in history if h["role"] == role_type]

        if not role_history:
            return {
                "role": role_type,
                "total_evolutions": 0,
                "trend": [],
                "latest": None,
                "summary": "暂无进化记录",
            }

        trends = [h["improvement"] for h in role_history]
        latest = role_history[-1]

        # 计算整体趋势
        avg_improvement = {}
        all_keys = set()
        for t in trends:
            all_keys.update(t.keys())
        for key in all_keys:
            values = [t.get(key, 0) for t in trends]
            avg_improvement[key] = round(sum(values) / len(values), 3) if values else 0

        return {
            "role": role_type,
            "total_evolutions": len(role_history),
            "trend": trends,
            "latest": latest,
            "avg_improvement": avg_improvement,
            "summary": self._summarize_trend(trends),
        }

    def get_all_reports(self) -> Dict[str, Dict[str, Any]]:
        """获取所有角色的进化报告"""
        history = self._load_history()
        roles = set(h["role"] for h in history)
        return {role: self.get_evolution_report(role) for role in roles}

    def get_version_comparison(self, role_type: str) -> List[Dict]:
        """获取某角色各版本的对比"""
        history = self._load_history()
        role_history = [h for h in history if h["role"] == role_type]
        comparisons = []
        for h in role_history:
            comparisons.append({
                "version": h["version"],
                "timestamp": h["timestamp"],
                "before": h["before"],
                "after": h["after"],
                "improvement": h["improvement"],
            })
        return comparisons

    @staticmethod
    def _calc_improvement(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
        """计算改进幅度"""
        improvement = {}
        all_keys = set(before.keys()) | set(after.keys())
        for key in all_keys:
            b = before.get(key, 0)
            a = after.get(key, 0)
            if b != 0:
                improvement[key] = round((a - b) / abs(b), 3)
            elif a != 0:
                improvement[key] = 1.0
            else:
                improvement[key] = 0.0
        return improvement

    @staticmethod
    def _summarize_trend(trends: List[Dict]) -> str:
        """生成趋势总结"""
        if not trends:
            return "暂无数据"
        if len(trends) == 1:
            return "仅有一次进化记录"

        # 看最新趋势
        latest = trends[-1]
        positive = sum(1 for v in latest.values() if v > 0)
        total = len(latest)

        if positive == total:
            return "全面正向进化"
        elif positive > total / 2:
            return "大部分指标改善"
        elif positive > 0:
            return "部分指标改善，部分退步"
        else:
            return "需要重新审视进化策略"

    def clear(self):
        """清空进化记录"""
        self._save_history([])
