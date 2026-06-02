"""策略优化器

根据对局分析结果优化Agent策略
"""

from typing import Dict, Any, List
from evolution.experience import get_role_statistics, save_experience


class StrategyOptimizer:
    """策略优化器"""

    def __init__(self):
        self.role_weights: Dict[str, Dict[str, float]] = {
            "werewolf": {"aggression": 0.5, "coordination": 0.5},
            "seer": {"revelation_timing": 0.5, "check_target": 0.5},
            "witch": {"heal_usage": 0.5, "poison_usage": 0.5},
            "hunter": {"shoot_timing": 0.5, "target_selection": 0.5},
            "villager": {"reasoning": 0.5, "vote_accuracy": 0.5},
        }

    def optimize(self, analysis: Any) -> Dict[str, Any]:
        """根据分析结果优化策略"""
        suggestions = {}

        for role, perf in analysis.role_performance.items():
            if role not in self.role_weights:
                continue

            if perf.correct_kills < 2:
                self.role_weights[role]["aggression"] = min(1.0, self.role_weights[role]["aggression"] + 0.1)

            if perf.speeches_quality < 0.5:
                self.role_weights[role]["reasoning"] = max(0.0, self.role_weights[role]["reasoning"] - 0.1)

            suggestions[role] = self._generate_role_suggestions(role, perf)

        return suggestions

    def _generate_role_suggestions(self, role: str, perf: Any) -> List[str]:
        suggestions = []
        if perf.overall_score < 0.5:
            suggestions.append("建议降低决策风险，保守策略可能更有效")
        if perf.speeches_quality < 0.5:
            suggestions.append("发言质量有待提升，注意逻辑推理和表达清晰度")
        return suggestions

    def get_optimized_prompt(self, role: str, base_prompt: str) -> str:
        """获取优化后的Prompt"""
        weights = self.role_weights.get(role, {})
        if not weights:
            return base_prompt

        optimization_notes = ["\n\n## 策略优化建议"]
        for key, value in weights.items():
            if value > 0.7:
                optimization_notes.append(f"- {key}: 提高权重到 {value:.1%}")
            elif value < 0.3:
                optimization_notes.append(f"- {key}: 降低权重到 {value:.1%}")

        return base_prompt + "\n".join(optimization_notes)


def run_evolution(game_record: Dict[str, Any], analysis: Any):
    """运行自进化流程"""
    optimizer = StrategyOptimizer()
    suggestions = optimizer.optimize(analysis)

    role = game_record.get("role", "")
    is_winner = game_record.get("is_winner", False)
    win_rate = get_role_statistics(role).get("win_rate", 0.0)

    save_experience(role, {
        "game_id": game_record.get("game_id", ""),
        "is_winner": is_winner,
        "win_rate": win_rate,
        "analysis": suggestions,
    })

    return suggestions