"""玩家AI代理 - 基于 LangGraph 实现

每个玩家角色对应一个AI代理，用于在游戏中做决策
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目根目录到路径，导入 config
# config.py 在 D:\Codes\python\LLM-learn\config.py
_root_config = Path(__file__).parent.parent.parent.parent.parent / "config.py"
import importlib.util
_spec = importlib.util.spec_from_file_location("root_config", _root_config)
_root_config_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root_config_module)

OPENAI_API_KEY = _root_config_module.OPENAI_API_KEY
OPENAI_BASE_URL = _root_config_module.OPENAI_BASE_URL
DEFAULT_MODEL = _root_config_module.DEFAULT_MODEL

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


# 决策风格定义
DECISION_STYLES = {
    "cautious": {
        "name": "谨慎型",
        "description": "宁可放过可疑玩家，也不轻易误杀好人",
        "speech_tendency": "保守分析，少说少错",
        "vote_tendency": "跟票，不首先提名",
        "night_tendency": "不轻易用药/查验",
    },
    "bold": {
        "name": "大胆型",
        "description": "敢于冒险，快速做出判断",
        "speech_tendency": "激进指控，主动带队",
        "vote_tendency": "果断投票，不怕误杀",
        "night_tendency": "冒险用药/深夜击杀",
    },
    "random": {
        "name": "随机型",
        "description": "不要过度分析，依靠直觉",
        "speech_tendency": "随机发言，看心情",
        "vote_tendency": "随机投票",
        "night_tendency": "随机目标",
    },
    "balanced": {
        "name": "平衡型",
        "description": "综合考虑各种因素",
        "speech_tendency": "客观分析",
        "vote_tendency": "理性分析后投票",
        "night_tendency": "合理使用能力",
    },
}


class PlayerAgent:
    """玩家AI代理 - 基于 LangGraph"""

    def __init__(self, player_id: int, role_name: str, private_context: Dict[str, Any],
                 camp: str, decision_style: str = "balanced", role_type: str = ""):
        self.player_id = player_id
        self.role_name = role_name
        self.private_context = private_context
        self.camp = camp
        self.decision_style = decision_style
        self.role_type = role_type or ""

        model_name = DEFAULT_MODEL or "gpt-4o"
        model = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None,
        )
        instructions = self._build_instructions()
        from langchain_core.messages import SystemMessage
        self.agent = create_react_agent(
            model,
            tools=[],
            prompt=SystemMessage(content=instructions)
        )

    def _build_instructions(self) -> str:
        camp_desc = "善良阵营" if self.camp == "good" else "邪恶阵营"
        style_info = DECISION_STYLES.get(self.decision_style, DECISION_STYLES["balanced"])

        return f"""你是一个狼人杀游戏中的玩家。
你的角色是：{self.role_name}
你的阵营是：{camp_desc}
你的玩家ID是：{self.player_id}

## 游戏规则
1. 夜晚：狼人需要协调击杀目标，预言家查验玩家，女巫决定是否用药
2. 白天：公开辩论后投票选出嫌疑人处决
3. 胜利条件：
   - 善良阵营：消灭所有狼人
   - 邪恶阵营：消灭所有神职或所有村民

## 我的私有信息
{json.dumps(self.private_context, ensure_ascii=False, indent=2)}

## 决策风格
你的决策风格是：{style_info['name']}
特点：{style_info['description']}
- 发言倾向：{style_info['speech_tendency']}
- 投票倾向：{style_info['vote_tendency']}
- 夜间行动倾向：{style_info['night_tendency']}

## 输出要求
你必须输出JSON格式的决策，格式如下：
- 夜晚行动：{{"action": "night_action", "target": 玩家ID或null, "reasoning": "决策理由"}}
- 白天发言：{{"action": "speech", "content": "发言内容"}}
- 投票：{{"action": "vote", "target": 玩家ID或null}}

请根据当前游戏状态做出最优决策。
"""

    async def decide_night_action(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_night_prompt(game_state)
        result = await self.agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
        output = result["messages"][-1].content
        return self._parse_json_output(output)

    async def decide_speech(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_speech_prompt(game_state)
        result = await self.agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
        output = result["messages"][-1].content
        return self._parse_json_output(output)

    async def decide_vote(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_vote_prompt(game_state)
        result = await self.agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
        output = result["messages"][-1].content
        return self._parse_json_output(output)

    def _build_night_prompt(self, game_state: Dict[str, Any]) -> str:
        style_info = DECISION_STYLES.get(self.decision_style, DECISION_STYLES["balanced"])
        return f"""## 当前游戏状态
{json.dumps(game_state, ensure_ascii=False, indent=2)}

## 你的角色是 {self.role_name}
请决定今晚的行动。

夜晚行动选项：
- 狼人：选择要击杀的目标玩家ID
- 预言家：选择要查验的目标玩家ID
- 女巫：选择是否用药（heal/poison）和目标
- 猎人：如果死亡，选择要开枪带走的目标玩家ID
- 村民：无夜间行动

{style_info['night_tendency']}

请以JSON格式输出你的决策：
"""

    def _build_speech_prompt(self, game_state: Dict[str, Any]) -> str:
        style_info = DECISION_STYLES.get(self.decision_style, DECISION_STYLES["balanced"])
        return f"""## 当前游戏状态
{json.dumps(game_state, ensure_ascii=False, indent=2)}

## 你的角色是 {self.role_name}
现在是公开辩论时间，请发表你的发言。

{style_info['speech_tendency']}

发言要点：
1. 根据已有信息分析谁可能是狼人
2. 说明你的推理过程
3. 质疑你认为可疑的玩家
4. 保护队友（如你是神职）

请以JSON格式输出你的发言：
{{"action": "speech", "content": "发言内容"}}
"""

    def _build_vote_prompt(self, game_state: Dict[str, Any]) -> str:
        style_info = DECISION_STYLES.get(self.decision_style, DECISION_STYLES["balanced"])
        return f"""## 当前游戏状态
{json.dumps(game_state, ensure_ascii=False, indent=2)}

## 你的角色是 {self.role_name}
现在进入投票环节，请选择你要投票的目标。

{style_info['vote_tendency']}

规则：
- 你可以选择投某位玩家
- 也可以选择跳过（不投票）
- 票数最多的玩家将被处决

请以JSON格式输出你的投票：
{{"action": "vote", "target": 玩家ID或null}}
"""

    def _parse_json_output(self, output: str) -> Dict[str, Any]:
        json_match = re.search(r'\{[^{}]*\}', output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {"action": "unknown", "raw_output": output}


class JudgeAgent:
    """主持人AI代理（裁判）"""

    def __init__(self):
        model_name = DEFAULT_MODEL or "gpt-4o"
        model = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None,
        )
        instructions = """你是狼人杀游戏的主持人（裁判）。
你的职责是：
1. 按照游戏规则推进游戏流程
2. 收集并执行玩家的决策
3. 宣布每天的死亡结果
4. 判断游戏是否结束及胜利方

游戏流程：
1. 夜晚：依次执行狼人杀人、预言家查验、女巫用药
2. 白天：宣布死亡、公开辩论、投票处决

你必须输出JSON格式的游戏指令。
"""
        self.agent = create_react_agent(model, tools=[], prompt=SystemMessage(content=instructions))


def create_player_agent(player_id: int, role_name: str, private_context: Dict[str, Any],
                       camp: str, decision_style: str = "balanced",
                       role_type: str = "") -> PlayerAgent:
    """工厂函数：创建玩家代理"""
    return PlayerAgent(player_id, role_name, private_context, camp, decision_style, role_type)


def create_judge_agent() -> JudgeAgent:
    """工厂函数：创建主持人代理"""
    return JudgeAgent()