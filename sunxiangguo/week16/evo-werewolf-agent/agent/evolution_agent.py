"""自进化分析 Agent

在多局游戏后，分析经验趋势，提炼出可执行的策略进化建议。
实现 "对局 → 分析 → 优化 → 再对局" 的自进化循环。
"""

from typing import Dict, Any, List, Optional
import json
import re

from agents import Agent, Runner
from agents import set_default_openai_api, set_tracing_disabled
from schema.system_config import load_system_config

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

config = load_system_config("config/system_config.json")

ROLE_NAMES = {
    "werewolf": "狼人",
    "seer": "预言家",
    "witch": "女巫",
    "hunter": "猎人",
    "villager": "村民",
}


EVOLUTION_PROMPT = """你是一个狼人杀策略进化分析师。
以下是{role_name}角色在最近{num_games}局游戏中的表现记录：

{experience_records}

## 评测数据
{metrics_summary}

请分析这些记录，输出JSON格式的进化建议：
{{
    "trend_analysis": "这些对局中体现出的整体趋势（2-3句话）",
    "recurring_mistakes": ["反复出现的错误模式1", "错误模式2"],
    "effective_strategies": ["被验证有效的策略1", "策略2"],
    "evolution_plan": [
        {{
            "aspect": "需要进化的方面",
            "current": "当前行为",
            "target": "目标行为",
            "method": "如何实现这个转变"
        }}
    ],
    "prompt_update_suggestion": "对角色Prompt的具体修改建议（可直接使用的文本）",
    "priority": "最优先改进的方向"
}}
"""


class EvolutionAgent:
    """自进化分析Agent

    分析多个对局经验，提炼进化建议。
    """

    def __init__(self):
        self.agent = Agent(
            name="EvolutionAnalyst",
            model=config.default_model,
            instructions="你是一个专业的狼人杀策略进化分析师，擅长从多局游戏中提炼可执行的改进方案。",
        )

    async def analyze_evolution(
        self,
        role_type: str,
        experiences: List[Dict],
        metrics: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """分析一个角色的进化方向

        Args:
            role_type: 角色类型（werewolf/seer/witch/hunter/villager）
            experiences: 该角色的历史经验列表
            metrics: 可选的评测指标数据

        Returns:
            进化分析结果
        """
        role_name = ROLE_NAMES.get(role_type, role_type)
        records = self._format_experiences(experiences)
        metrics_text = self._format_metrics(metrics)

        prompt = EVOLUTION_PROMPT.format(
            role_name=role_name,
            num_games=len(experiences),
            experience_records=records,
            metrics_summary=metrics_text,
        )

        result = await Runner.run(self.agent, prompt)
        return self._parse_json_output(result.final_output)

    async def batch_analyze(
        self,
        all_experiences: Dict[str, List[Dict]],
        all_metrics: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """批量分析所有角色的进化方向

        Args:
            all_experiences: {role_type: [experiences]} 格式
            all_metrics: {role_type: metrics} 格式

        Returns:
            {role_type: evolution_result}
        """
        results = {}
        for role_type, experiences in all_experiences.items():
            if not experiences:
                continue
            metrics = all_metrics.get(role_type) if all_metrics else None
            try:
                result = await self.analyze_evolution(role_type, experiences, metrics)
                results[role_type] = result
            except Exception as e:
                results[role_type] = {
                    "error": str(e),
                    "trend_analysis": "分析失败",
                    "evolution_plan": [],
                }
        return results

    @staticmethod
    def _format_experiences(experiences: List[Dict]) -> str:
        """格式化经验记录"""
        lines = []
        for i, exp in enumerate(experiences[-10:], 1):
            outcome = "✅胜利" if exp.get("is_winner", False) else "❌失败"
            lines.append(f"\n--- 第{i}局（{outcome}）---")
            if exp.get("summary"):
                lines.append(f"  总结：{exp['summary'][:150]}")
            if exp.get("strategies"):
                lines.append(f"  策略：{exp['strategies'][:150]}")
            if exp.get("mistakes"):
                lines.append(f"  错误：{exp['mistakes'][:150]}")
            if exp.get("lessons"):
                lines.append(f"  教训：{exp['lessons'][:150]}")
            if exp.get("strategy_tags"):
                lines.append(f"  标签：{', '.join(exp['strategy_tags'])}")
        return "\n".join(lines) if lines else "暂无经验记录"

    @staticmethod
    def _format_metrics(metrics: Optional[Dict]) -> str:
        """格式化评测指标"""
        if not metrics:
            return "暂无评测数据"
        lines = []
        if "win_rate" in metrics:
            lines.append(f"- 胜率：{metrics['win_rate']:.1%}")
        if "avg_score" in metrics:
            lines.append(f"- 平均得分：{metrics['avg_score']:.1f}")
        if "avg_speech_quality" in metrics:
            lines.append(f"- 发言质量：{metrics['avg_speech_quality']:.1f}")
        if "avg_vote_accuracy" in metrics:
            lines.append(f"- 投票准确率：{metrics['avg_vote_accuracy']:.1%}")
        if "avg_survival_days" in metrics:
            lines.append(f"- 平均存活天数：{metrics['avg_survival_days']:.1f}")
        return "\n".join(lines) if lines else "暂无评测数据"

    @staticmethod
    def _parse_json_output(output: str) -> Dict[str, Any]:
        """解析JSON输出"""
        output = output.strip()

        if output.startswith('{'):
            try:
                return json.loads(output)
            except json.JSONDecodeError:
                pass

        code_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', output, re.DOTALL)
        if code_block:
            try:
                return json.loads(code_block.group(1))
            except json.JSONDecodeError:
                pass

        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return {
            "trend_analysis": output[:500],
            "recurring_mistakes": [],
            "effective_strategies": [],
            "evolution_plan": [],
            "prompt_update_suggestion": "解析失败，请重新分析",
            "priority": "unknown",
        }
