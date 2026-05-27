"""复盘归因 Agent

基于游戏评测数据，由LLM自动进行复盘分析：
- 关键转折点识别
- 胜负归因分析
- 各玩家表现点评
- 改进建议
"""

from typing import Dict, Any, List
import json
import re

from agents import Agent, Runner
from agents import set_default_openai_api, set_tracing_disabled
from schema.system_config import load_system_config

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

config = load_system_config("config/system_config.json")


REVIEW_PROMPT = """你是狼人杀游戏的复盘分析师。请基于以下对局数据进行深度复盘分析。

## 对局概况
- 游戏ID: {game_id}
- 胜利方: {winner}
- 总天数: {total_days}
- 总死亡数: {total_deaths}
- 对局质量分: {game_quality}

## 玩家评测
{player_stats}

## 关键事件时间线
{timeline}

## 分析要求
请从以下维度进行复盘：

1. **关键转折点**：哪些事件改变了游戏走向？（如：预言家首夜查验、狼人悍跳、猎人开枪等）
2. **胜负归因**：胜利方的关键策略是什么？失败方的主要失误在哪里？
3. **MVP与背锅**：表现最好的玩家是谁？谁的表现最差？
4. **策略建议**：各角色在类似局面下应该如何改进？

请输出JSON格式：
{{
  "key_turning_points": ["转折点1", "转折点2"],
  "win_factors": ["胜利因素1", "胜利因素2"],
  "loss_factors": ["失败因素1", "失败因素2"],
  "mvp_player": {{"player_id": 0, "reason": "原因"}},
  "worst_player": {{"player_id": 1, "reason": "原因"}},
  "strategy_advice": {{"werewolf": "建议", "seer": "建议", "witch": "建议", "hunter": "建议", "villager": "建议"}},
  "overall_review": "总体复盘总结（2-3句话）"
}}
"""


class ReviewAgent:
    """复盘归因Agent"""

    def __init__(self):
        self.agent = Agent(
            name="Reviewer",
            model=config.default_model,
            instructions="你是一个专业的狼人杀游戏复盘分析师，擅长从数据中提炼洞察。",
        )

    async def generate_review(
        self,
        game_eval: Dict[str, Any],
        timeline: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """生成复盘报告"""
        player_stats = self._format_player_stats(game_eval.get("players", []))
        timeline_text = self._format_timeline(timeline)

        prompt = REVIEW_PROMPT.format(
            game_id=game_eval.get("game_id", "unknown"),
            winner="善良阵营" if game_eval.get("winner") == "good" else "狼人阵营",
            total_days=game_eval.get("total_days", 0),
            total_deaths=game_eval.get("total_deaths", 0),
            game_quality=game_eval.get("game_quality_score", 0),
            player_stats=player_stats,
            timeline=timeline_text,
        )

        result = await Runner.run(self.agent, prompt)
        return self._parse_json(result.final_output)

    @staticmethod
    def _format_player_stats(players: list) -> str:
        """格式化玩家评测数据"""
        lines = []
        for p in players:
            winner_mark = "🏆" if p.get("is_winner") else "💀"
            lines.append(
                f"{winner_mark} 玩家{p.get('player_id', '?')} ({p.get('player_name', '')}) "
                f"- {p.get('role', '?')} [{p.get('camp', '?')}] "
                f"| 结果分:{p.get('result_score', 0)} "
                f"| 过程分:{p.get('process_score', 0)} "
                f"| 对抗分:{p.get('adversarial_score', 0)} "
                f"| 总分:{p.get('total_score', 0)}"
            )
        return "\n".join(lines) if lines else "无玩家数据"

    @staticmethod
    def _format_timeline(timeline: list) -> str:
        """格式化关键事件时间线"""
        lines = []
        for i, event in enumerate(timeline[:20]):
            day = event.get("day", "?")
            phase = event.get("phase", "?")
            desc = event.get("description", str(event))
            lines.append(f"[第{day}天 {phase}] {desc}")
        return "\n".join(lines) if lines else "无事件记录"

    @staticmethod
    def _parse_json(output: str) -> Dict[str, Any]:
        """解析JSON输出"""
        output = output.strip()

        # 尝试直接解析
        if output.startswith('{'):
            try:
                return json.loads(output)
            except json.JSONDecodeError:
                pass

        # 提取代码块
        code_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', output, re.DOTALL)
        if code_block:
            try:
                return json.loads(code_block.group(1))
            except json.JSONDecodeError:
                pass

        # 提取花括号
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return {
            "overall_review": output[:500],
            "key_turning_points": [],
            "win_factors": [],
            "loss_factors": [],
            "mvp_player": {"player_id": 0, "reason": "解析失败"},
            "worst_player": {"player_id": 0, "reason": "解析失败"},
            "strategy_advice": {},
        }


def build_timeline(dialogues: list, death_records: list) -> List[Dict[str, Any]]:
    """从对话和死亡记录中构建关键事件时间线"""
    events = []

    # 死亡事件
    for d in death_records:
        events.append({
            "day": d.get("day", 1),
            "phase": "死亡",
            "description": f"玩家{d.get('player_id', '?')}({d.get('player_name', '')} - {d.get('role', '?')}) 因{d.get('cause', '?')}死亡",
            "priority": 3,
        })

    # 关键对话事件
    for d in dialogues:
        action = d.get("action", "")
        if action == "hunter_shot":
            events.append({
                "day": d.get("day", 1),
                "phase": d.get("phase", ""),
                "description": f"猎人{d.get('player_id', '?')}开枪带走玩家{d.get('target', '?')}",
                "priority": 5,
            })
        elif action == "vote":
            events.append({
                "day": d.get("day", 1),
                "phase": "投票",
                "description": f"玩家{d.get('player_id', '?')} 投票给 玩家{d.get('target', '?')}",
                "priority": 1,
            })

    events.sort(key=lambda x: (x.get("day", 0), -x.get("priority", 0)))
    return events
