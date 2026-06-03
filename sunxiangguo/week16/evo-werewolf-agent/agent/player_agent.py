"""玩家AI代理

每个玩家角色对应一个AI代理，用于在游戏中做决策。
包含角色专属策略指导、信息隔离原则和优化的Prompt模板。
"""

from typing import Dict, Any, List, Optional

from agents import Agent, Runner
from agents import set_default_openai_api, set_tracing_disabled
from schema.system_config import load_system_config
from memory.experience import get_experience_prompt

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

config = load_system_config("config/system_config.json")


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

# 角色专属策略指导
ROLE_STRATEGIES = {
    "werewolf": """## 角色策略：狼人
- 白天伪装：你需要假装好人，避免暴露狼人身份
- 可以悍跳预言家（声称自己是预言家并给出假的查验结果）
- 与队友协调：夜间杀人目标要统一
- 白天发言不要自相矛盾，保持逻辑一致性
- 投票时避免暴露同伴，可以投给其他狼人怀疑的对象来转移注意力
- 观察哪些人在怀疑你，优先处理对你威胁最大的玩家""",

    "seer": """## 角色策略：预言家
- 你的查验结果是关键信息，要合理分享
- 不要过早暴露身份（第一天可以不跳），但如果被怀疑必须跳出来
- 查验顺序：优先查验白天发言可疑的玩家
- 如果被狼人悍跳，要及时反驳并拿出真实查验结果
- 记录你每晚查验的结果，这是你最有力的武器""",

    "witch": """## 角色策略：女巫
- 解药（救人药）：第一晚可以考虑救人（平安夜），也可以留着应对关键人物
- 毒药：谨慎使用，只在有充分理由时使用
- 不要过早暴露自己用药情况
- 你是好人阵营的关键角色，保命很重要
- 如果有人跳女巫，需要判断真假""",

    "hunter": """## 角色策略：猎人
- 你的枪是威慑力，白天可以适当暗示自己有枪
- 被投票出局或被夜杀时可以开枪带走一人
- 开枪要基于发言和行为的分析，选择最可疑的玩家
- 被毒杀时无法开枪（闷枪），所以要注意保护身份
- 如果你被怀疑，跳出来表明猎人身份可以让狼人不敢轻易刀你""",

    "villager": """## 角色策略：村民
- 你没有特殊技能，但你的发言和投票同样重要
- 通过仔细分析每个人的发言、行为和逻辑漏洞来推理
- 可以跳神职来挡刀（高风险策略）
- 不要做划水玩家，每轮都要有有价值的发言
- 关注谁在引导投票、谁在转移话题""",
}


class PlayerAgent:
    """玩家AI代理

    根据角色类型生成对应的LLM代理，进行游戏决策
    """

    def __init__(self, player_id: int, role_name: str, private_context: Dict[str, Any],
                 camp: str, decision_style: str = "balanced", role_type: str = ""):
        self.player_id = player_id
        self.role_name = role_name
        self.private_context = private_context
        self.camp = camp
        self.decision_style = decision_style
        self.role_type = role_type or ""  # 英文角色类型，如 "werewolf", 用于加载经验

        instructions = self._build_instructions()

        self.agent = Agent(
            name=f"Player_{player_id}",
            model=config.default_model,
            instructions=instructions,
        )

    def _build_instructions(self) -> str:
        """构建角色专属提示词（优化版）"""
        camp_desc = "善良阵营" if self.camp == "good" else "邪恶阵营"
        style_info = DECISION_STYLES.get(self.decision_style, DECISION_STYLES["balanced"])
        role_strategy = ROLE_STRATEGIES.get(self.role_type, "## 请根据当前局势做出最佳决策")

        instructions = f"""你是一个狼人杀游戏中的玩家。

## 你的身份
- 玩家编号：{self.player_id}号
- 角色：{self.role_name}
- 阵营：{camp_desc}

## 核心规则
- 胜利条件：善良阵营消灭所有狼人 / 邪恶阵营屠边成功（消灭所有神职 或 所有平民）
- 白天流程：发言→投票→得票最多者出局
- 夜晚流程：狼人杀人→预言家查验→女巫用药

## 你的私有信息
{self.private_context}

## {role_strategy}

## 决策风格：{style_info['name']}
{style_info['description']}
- 发言：{style_info['speech_tendency']}
- 投票：{style_info['vote_tendency']}
- 夜间：{style_info['night_tendency']}

## 信息隔离原则
- 你只能看到你视野内的信息，绝不能暴露你不知道的信息
- 善良阵营玩家不知道其他人的身份
- 狼人只能看到其他狼人，不知道谁是神职、谁是平民
- 绝对不要在发言中暴露自己的角色（除非你是狼人故意悍跳）
{self._get_experience_section()}

## 输出格式
严格按JSON格式输出，不要包含其他内容。
"""
        return instructions

    def _get_experience_section(self) -> str:
        """获取过往经验文本，用于注入到提示词中"""
        if not self.role_type:
            return ""
        return get_experience_prompt(self.role_type)

    async def decide_night_action(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """夜晚行动决策"""
        prompt = self._build_night_prompt(game_state)
        result = await Runner.run(self.agent, prompt)
        return self._parse_json_output(result.final_output)

    async def decide_speech(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """白天发言决策"""
        prompt = self._build_speech_prompt(game_state)
        result = await Runner.run(self.agent, prompt)
        return self._parse_json_output(result.final_output)

    async def decide_vote(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """投票决策"""
        prompt = self._build_vote_prompt(game_state)
        result = await Runner.run(self.agent, prompt)
        return self._parse_json_output(result.final_output)

    async def decide_last_words(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """遗言决策"""
        prompt = self._build_last_words_prompt(game_state)
        result = await Runner.run(self.agent, prompt)
        return self._parse_json_output(result.final_output)

    async def decide_hunter_shot(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """猎人开枪决策"""
        prompt = self._build_hunter_shot_prompt(game_state)
        result = await Runner.run(self.agent, prompt)
        return self._parse_json_output(result.final_output)

    def _build_night_prompt(self, game_state: Dict[str, Any]) -> str:
        """构建夜间决策提示（优化版）"""
        day = game_state.get("day_number", 1)
        alive = game_state.get("alive_players", [])
        style_info = DECISION_STYLES.get(self.decision_style, DECISION_STYLES["balanced"])
        action_guide = self._get_night_action_guide(game_state)

        return f"""## 夜间决策
当前：第{day}夜 | 存活玩家：{alive}

{action_guide}

## 你的决策风格
{style_info['night_tendency']}

## 输出要求
严格输出JSON，格式：
{{"action": "night_action", "target": 玩家ID整数或null, "reasoning": "简短的决策理由"}}

如果是女巫用药：
{{"action": "heal", "target": 今晚死者ID, "reasoning": "理由"}}
或
{{"action": "poison", "target": 目标ID, "reasoning": "理由"}}
或
{{"action": "none", "reasoning": "不用药的理由"}}
"""

    def _get_night_action_guide(self, game_state: Dict[str, Any]) -> str:
        """根据角色返回夜间行动指南"""
        guides = {
            "werewolf": "你是狼人，请选择今晚要击杀的目标。优先击杀：神职嫌疑最大的人、发言逻辑强的人、可能威胁你的玩家。避免击杀已死玩家的队友（会被怀疑）。",
            "seer": "你是预言家，请选择今晚要查验的玩家。优先查验：白天发言可疑的人、新跳身份的人、你直觉不对的人。",
            "witch": f"你是女巫。今晚被刀的玩家是：{game_state.get('tonight_death', '无')}。你还有{'解药' if game_state.get('has_heal', True) else ''}{'和' if game_state.get('has_heal', True) and game_state.get('has_poison', True) else ''}{'毒药' if game_state.get('has_poison', True) else ''}可用。同夜不能同时使用双药。",
            "hunter": "你是猎人，夜间无行动。",
            "villager": "你是村民，夜间无行动。",
        }
        return guides.get(self.role_type, "你今夜无行动。")

    def _build_speech_prompt(self, game_state: Dict[str, Any]) -> str:
        """构建白天发言提示（优化版）"""
        day = game_state.get("day_number", 1)
        alive = game_state.get("alive_players", [])
        dialogues = game_state.get("dialogues", [])
        speech_summary = self._build_dialogue_summary(dialogues)

        return f"""## 白天发言
当前：第{day}天 | 存活：{alive}

## 历史记录摘要
{speech_summary}

## 发言要求
1. 分析历史发言，找出每个人的逻辑漏洞和可疑之处
2. 结合你的角色身份，做出合理的表水（自证清白）或带节奏
3. 如果你是狼人：保持伪装身份，转移注意力
4. 如果你是好人：用逻辑推理指出可疑玩家
5. 发言要有信息量，避免空话套话，控制在2-4句话

## 输出
{{"action": "speech", "content": "你的发言内容（自然语言，2-4句话）"}}
"""

    def _build_vote_prompt(self, game_state: Dict[str, Any]) -> str:
        """构建投票提示（优化版）"""
        day = game_state.get("day_number", 1)
        alive = game_state.get("alive_players", [])

        return f"""## 投票环节
当前：第{day}天 | 存活：{alive}

## 投票规则
- 得票最多的玩家将被处决
- 你可以投任意存活玩家，也可以弃票（target设为null）

## 投票建议
1. 结合白天发言，选择最可疑/威胁最大的玩家
2. 如果你是狼人，避免投给队友
3. 如果你是好人，投给你分析后最可能是狼的人

## 输出
{{"action": "vote", "target": 玩家ID整数或null, "reasoning": "简短的投票理由"}}
"""

    def _build_last_words_prompt(self, game_state: Dict[str, Any]) -> str:
        """构建遗言提示"""
        day = game_state.get("day_number", 1)

        return f"""## 遗言
你已被淘汰，这是你最后的机会留下信息。

当前：第{day}天

## 遗言要求
1. 如果你是好人：分享你的分析和怀疑对象，帮助队友
2. 如果你是狼人：可以混淆视听、保护队友
3. 简短有力，2-3句话即可

## 输出
{{"action": "last_words", "content": "你的遗言内容"}}
"""

    def _build_hunter_shot_prompt(self, game_state: Dict[str, Any]) -> str:
        """构建猎人开枪提示"""
        alive = game_state.get("alive_players", [])

        return f"""## 猎人开枪
你已死亡，现在可以开枪带走一名玩家。

存活玩家：{alive}

## 开枪建议
1. 基于发言和行为分析，选择最可疑的玩家
2. 如果你是好人阵营，优先击杀你判断为狼人的玩家
3. 不要开枪带走队友

## 输出
{{"action": "hunter_shot", "target": 玩家ID整数, "reasoning": "开枪理由"}}
"""

    @staticmethod
    def _build_dialogue_summary(dialogues: list) -> str:
        """将对话历史压缩为可读摘要"""
        if not dialogues:
            return "暂无历史记录。"

        # 只取最近20条，避免超出上下文
        recent = dialogues[-20:] if len(dialogues) > 20 else dialogues
        lines = []
        for d in recent:
            day = d.get("day", "?")
            player = d.get("player_name", f"玩家{d.get('player_id', '?')}")
            action = d.get("action", "")

            if action == "speech":
                content = d.get("content", "")[:80]
                lines.append(f"[第{day}天] {player}：{content}")
            elif action == "vote":
                target = d.get("target")
                lines.append(f"[第{day}天] {player} 投票给 玩家{target}")
            elif action in ("night_kill", "seer_check", "heal", "poison"):
                phase = d.get("phase", "夜间")
                lines.append(f"[第{day}夜] {phase}: {player} 行动")
            elif action == "last_words":
                content = d.get("content", "")[:80]
                lines.append(f"[第{day}天] {player} 遗言：{content}")

        return "\n".join(lines) if lines else "暂无历史记录。"

    def _parse_json_output(self, output: str) -> Dict[str, Any]:
        """解析LLM输出的JSON（增强版）"""
        import json
        import re

        output = output.strip()

        # 1. 尝试直接解析
        if output.startswith('{'):
            try:
                return json.loads(output)
            except json.JSONDecodeError:
                pass

        # 2. 提取代码块中的JSON
        code_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', output, re.DOTALL)
        if code_block:
            try:
                return json.loads(code_block.group(1))
            except json.JSONDecodeError:
                pass

        # 3. 提取花括号中的JSON（支持嵌套）
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # 4. 兜底
        print(f"[WARNING] JSON解析失败，返回原始输出: {output[:100]}")
        return {"action": "unknown", "raw_output": output}


class JudgeAgent:
    """主持人AI代理（裁判）

    负责判定游戏规则、计算结果、推进游戏流程
    """

    def __init__(self):
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
        self.agent = Agent(
            name="Judge",
            model=config.default_model,
            instructions=instructions,
        )

    async def announce_death(self, deaths: List[int], cause: str, game_state: Dict[str, Any]) -> str:
        """宣布死亡结果"""
        if not deaths:
            return "今晚无人死亡。"

        death_names = [f"玩家{p}" for p in deaths]
        cause_desc = {
            "night_kill": "昨夜",
            "vote": "投票",
            "shoot": "枪杀",
            "poison": "毒杀",
        }.get(cause, cause)

        announcement = f"{cause_desc}，以下玩家死亡：{', '.join(death_names)}"

        for player_id in deaths:
            player = game_state.get("players", {}).get(player_id)
            if player:
                announcement += f"\n{player.get('name', f'玩家{player_id}')} 说："

        return announcement

    async def announce_phase(self, phase: str, day_number: int) -> str:
        """宣布游戏阶段"""
        if "night" in phase:
            return f"第{day_number}夜开始，请各位保持安静。"
        else:
            return f"第{day_number}天，阳光照耀，请各位玩家开始发言。"


def create_player_agent(player_id: int, role_name: str, private_context: Dict[str, Any],
                       camp: str, decision_style: str = "balanced",
                       role_type: str = "") -> PlayerAgent:
    """工厂函数：创建玩家代理"""
    return PlayerAgent(player_id, role_name, private_context, camp, decision_style, role_type)


def create_judge_agent() -> JudgeAgent:
    """工厂函数：创建主持人代理"""
    return JudgeAgent()
