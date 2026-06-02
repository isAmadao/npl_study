# evo-werewolf-agent 项目优化建议（业务优先版）

> **核心原则**: 先把主流程跑通、跑聪明，架构优化（Redis持久化、Docker容器化等）延后。

## 📋 优化路线图

按优先级从高到低，分为四个阶段：

| 阶段 | 目标 | 关键任务 |
|------|------|----------|
| **P0 - 跑通** | 核心流程完整无Bug | 猎人智能决策、平票PK/遗言、异常处理 |
| **P1 - 跑聪明** | AI决策质量提升 | **Prompt工程优化**、并发决策、上下文管理 |
| **P2 - 能评测** | 可量化分析对局质量 | 评测体系 + 复盘归因 + Leaderboard |
| **P3 - 能进化** | Agent自我迭代提升 | 自进化循环、经验沉淀、策略进化 |

---

## 🔴 P0: 跑通 — 核心流程完整无Bug

### 0.1 实现猎人智能决策

**当前问题**:
- 猎人开枪目标硬编码为"第一个存活玩家" (`engine/game_engine.py:544`)
- 代码中有TODO注释表明应由AI代理决策

**修复方案**:

```python
# engine/game_engine.py  _handle_hunter_death 方法
async def _handle_hunter_death(self, killed_player_ids: List[int]):
    """处理猎人死亡开枪"""
    for player_id in killed_player_ids:
        player = self.game_state.get_player(player_id)
        if player and player.role_type == RoleType.HUNTER and player.role.can_shoot:
            # 被毒死不能开枪（闷枪）
            if self._night_death_causes.get(player_id) == "poison":
                player.role.lock_shoot()
                self._log("log_event", f"猎人{player.name}被毒杀，无法开枪")
                continue

            if player.agent:
                context = self.game_state.get_player_private_context(player_id)
                context["death_cause"] = self._night_death_causes.get(player_id, "unknown")
                decision = await player.agent.decide_hunter_shot(context)
                target = decision.get("target")

                if target is not None:
                    player.role.lock_shoot()
                    target_player = self.game_state.get_player(target)
                    if target_player and target_player.is_alive:
                        target_player.role.is_alive = False
                        # ... 记录死亡和对话 ...
```

**Agent侧新增**:
```python
# agent/player_agent.py 新增 decide_hunter_shot 方法
async def decide_hunter_shot(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
    """猎人开枪决策"""
    prompt = self._build_hunter_shot_prompt(game_state)
    result = await Runner.run(self.agent, prompt)
    return self._parse_json_output(result.final_output)
```

---

### 0.2 添加平票PK和遗言机制

**当前问题**:
- 平票处理简化为 `pass` (`engine/game_engine.py:701`)
- 缺少遗言功能（被投出/被夜杀的玩家无法留下遗言）

**实现方案**:

```python
# engine/game_engine.py 修改 _vote 方法
async def _vote(self):
    # ... 现有投票逻辑 ...

    if vote_count:
        max_votes = max(vote_count.values())
        eliminated = [p for p, c in vote_count.items() if c == max_votes]

        if len(eliminated) == 1:
            # 正常出局
            player = self.game_state.get_player(eliminated[0])
            player.kill("vote", self.game_state.to_dict())
            # 遗言
            await self._handle_last_words(eliminated[0])
            await self._handle_hunter_death([eliminated[0]])
        else:
            # 平票PK
            eliminated_id = await self._handle_tie(eliminated)
            if eliminated_id is not None:
                player = self.game_state.get_player(eliminated_id)
                player.kill("vote", self.game_state.to_dict())
                await self._handle_last_words(eliminated_id)
                await self._handle_hunter_death([eliminated_id])
            else:
                self._log("log_event", "平票无人出局")

async def _handle_tie(self, tied_players: List[int]) -> Optional[int]:
    """处理平票PK：PK发言 → 重新投票 → 仍平则无人出局"""
    # 1. PK发言
    for pid in tied_players:
        player = self.game_state.get_player(pid)
        if player and player.agent:
            ctx = self.game_state.get_player_private_context(pid)
            ctx["pk_players"] = tied_players
            speech = await player.agent.decide_speech(ctx)
            self.game_state.dialogues.append({
                "day": self.game_state.day_number,
                "phase": "平票PK",
                "player_id": pid,
                "player_name": player.name,
                "action": "speech",
                "content": speech.get("content", ""),
            })

    # 2. 非PK玩家重新投票（仅投给平票玩家）
    voters = [p for p in self.game_state.get_alive_players()
              if p.player_id not in tied_players]
    votes = {}
    for voter in voters:
        if voter.agent:
            ctx = self.game_state.get_player_private_context(voter.player_id)
            ctx["pk_candidates"] = tied_players
            decision = await voter.agent.decide_vote(ctx)
            target = decision.get("target")
            if target in tied_players:
                votes[voter.player_id] = target

    # 3. 统计PK投票
    if votes:
        vote_count = {}
        for t in votes.values():
            vote_count[t] = vote_count.get(t, 0) + 1
        max_v = max(vote_count.values())
        pk_eliminated = [p for p, c in vote_count.items() if c == max_v]
        if len(pk_eliminated) == 1:
            return pk_eliminated[0]
    return None  # 仍平票，无人出局

async def _handle_last_words(self, player_id: int):
    """处理遗言"""
    player = self.game_state.get_player(player_id)
    if player and player.agent:
        ctx = self.game_state.get_player_private_context(player_id)
        ctx["is_dying"] = True
        last_words = await player.agent.decide_last_words(ctx)
        self.game_state.dialogues.append({
            "day": self.game_state.day_number,
            "phase": "遗言",
            "player_id": player_id,
            "player_name": player.name,
            "action": "last_words",
            "content": last_words.get("content", ""),
        })
```

**Agent侧新增**:
```python
# agent/player_agent.py
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
```

---

### 0.3 完善异常处理

**当前问题**:
- LLM调用失败时无兜底
- JSON解析失败返回默认值过于粗糙

**修复方案**:

```python
# agent/player_agent.py  _parse_json_output 优化
def _parse_json_output(self, output: str) -> Dict[str, Any]:
    """解析LLM输出的JSON（增强版）"""
    import json, re

    # 1. 尝试直接解析
    output = output.strip()
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

    # 3. 提取花括号中的JSON
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', output, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # 4. 兜底：返回原始文本，由调用方处理
    print(f"[WARNING] JSON解析失败，返回原始输出: {output[:100]}")
    return {"action": "unknown", "raw_output": output, "parse_failed": True}
```

---

## 🟡 P1: 跑聪明 — AI决策质量提升

### 1.1 🔴 优化Prompt指令模板（最重要的优化）

**当前问题**:
1. **系统提示词太笼统** — 角色行为缺乏具体策略指导
2. **白天发言提示太弱** — 缺少"如何分析历史发言"的指引
3. **投票提示缺少推理要求** — 投票没有要求说明理由
4. **JSON输出不稳定** — 经常解析失败
5. **缺少信息隔离意识** — Prompt没有强调"你不能知道别人不知道的信息"
6. **缺少角色专属策略** — 所有角色共用一套泛化Prompt，没有针对性

**优化方案 — 按角色定制系统提示词**:

#### 1.1.1 重写 `_build_instructions()`

```python
def _build_instructions(self) -> str:
    """构建角色专属提示词（优化版）"""
    style_info = DECISION_STYLES.get(self.decision_style, DECISION_STYLES["balanced"])
    camp_desc = "善良阵营" if self.camp == "good" else "邪恶阵营"

    # 角色专属策略指导
    role_strategy = self._get_role_strategy()

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

## 角色专属策略
{role_strategy}

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

def _get_role_strategy(self) -> str:
    """根据角色类型返回专属策略"""
    strategies = {
        "werewolf": """## 狼人策略要点
- 白天伪装：你需要假装好人，避免暴露狼人身份
- 可以悍跳预言家（声称自己是预言家并给出假的查验结果）
- 与队友协调：夜间杀人目标要统一
- 白天发言不要自相矛盾，保持逻辑一致性
- 投票时避免暴露同伴，可以投给其他狼人怀疑的对象来转移注意力
- 观察哪些人在怀疑你，优先处理对你威胁最大的玩家""",

        "seer": """## 预言家策略要点
- 你的查验结果是关键信息，要合理分享
- 不要过早暴露身份（第一天可以不跳），但如果被怀疑必须跳出来
- 查验顺序：优先查验发言可疑的玩家
- 如果被狼人悍跳，要及时反驳并拿出真实查验结果
- 记录你每晚查验的结果，这是你最有力的武器""",

        "witch": """## 女巫策略要点
- 解药（救人药）：第一晚可以考虑救人（平安夜），也可以留着应对关键人物
- 毒药：谨慎使用，只在有充分理由时使用
- 不要过早暴露自己用药情况
- 你是好人阵营的关键角色，保命很重要
- 如果有人跳女巫，需要判断真假""",

        "hunter": """## 猎人策略要点
- 你的枪是威慑力，白天可以适当暗示自己有枪
- 被投票出局或被夜杀时可以开枪带走一人
- 开枪要基于发言和行为的分析，选择最可疑的玩家
- 被毒杀时无法开枪（闷枪），所以要注意保护身份
- 如果你被怀疑，跳出来表明猎人身份可以让狼人不敢轻易刀你""",

        "villager": """## 村民策略要点
- 你没有特殊技能，但你的发言和投票同样重要
- 通过仔细分析每个人的发言、行为和逻辑漏洞来推理
- 可以跳神职来挡刀（高风险策略）
- 不要做划水玩家，每轮都要有有价值的发言
- 关注谁在引导投票、谁在转移话题""",
    }
    return strategies.get(self.role_type, "## 请根据当前局势做出最佳决策")
```

#### 1.1.2 重写夜间决策提示 `_build_night_prompt()`

```python
def _build_night_prompt(self, game_state: Dict[str, Any]) -> str:
    """夜间决策提示（优化版）"""
    day = game_state.get("day_number", 1)
    alive = game_state.get("alive_players", [])
    style_info = DECISION_STYLES.get(self.decision_style, DECISION_STYLES["balanced"])

    # 根据角色构建不同的行动选项
    action_guide = self._get_night_action_guide(game_state)

    return f"""## 夜间决策
当前：第{day}夜 | 存活玩家：{[p.get('name') for p in alive]}

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
        "witch": f"你是女巫。今晚被刀的玩家是：{game_state.get('tonight_death', '无')}。你还有{'解药' if game_state.get('has_heal', True) else ''}{'毒药' if game_state.get('has_poison', True) else ''}可用。",
        "hunter": "你是猎人，夜间无行动。",
        "villager": "你是村民，夜间无行动。",
    }
    return guides.get(self.role_type, "")
```

#### 1.1.3 重写发言提示 `_build_speech_prompt()`

```python
def _build_speech_prompt(self, game_state: Dict[str, Any]) -> str:
    """白天发言提示（优化版）"""
    day = game_state.get("day_number", 1)
    alive = game_state.get("alive_players", [])
    history = game_state.get("dialogues", [])

    # 构建对话摘要
    speech_summary = self._build_dialogue_summary(history)

    return f"""## 白天发言
当前：第{day}天 | 存活：{[p.get('name') for p in alive]}

## 历史记录摘要
{speech_summary}

## 发言要求
1. 分析历史发言，找出每个人的逻辑漏洞和可疑之处
2. 结合你的角色身份，做出合理的表水（自证清白）或带节奏
3. 如果你是狼人：保持伪装身份，转移注意力
4. 如果你是好人：用逻辑推理指出可疑玩家
5. 发言要有信息量，避免空话套话

## 输出
{{"action": "speech", "content": "你的发言内容（自然语言，2-4句话）"}}
"""

def _build_dialogue_summary(self, dialogues: list) -> str:
    """将对话历史压缩为可读摘要"""
    if not dialogues:
        return "暂无历史记录。"

    lines = []
    for d in dialogues[-20:]:  # 只取最近20条
        day = d.get("day", "?")
        phase = d.get("phase", "")
        player = d.get("player_name", f"玩家{d.get('player_id', '?')}")
        action = d.get("action", "")

        if action == "speech":
            content = d.get("content", "")[:80]
            lines.append(f"[第{day}天] {player}：{content}")
        elif action == "vote":
            target = d.get("target")
            lines.append(f"[第{day}天] {player} 投票给 玩家{target}")
        elif action in ("night_kill", "seer_check", "heal", "poison"):
            lines.append(f"[第{day}夜] {phase}: {player} 行动")

    return "\n".join(lines) if lines else "暂无历史记录。"
```

#### 1.1.4 重写投票提示 `_build_vote_prompt()`

```python
def _build_vote_prompt(self, game_state: Dict[str, Any]) -> str:
    """投票提示（优化版）"""
    day = game_state.get("day_number", 1)
    alive = game_state.get("alive_players", [])

    return f"""## 投票环节
当前：第{day}天 | 存活：{[p.get('name') for p in alive]}

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
```

---

### 1.2 实现并发决策机制

**当前问题**:
- 玩家决策串行执行，6人局要等每个LLM调用（约5秒/次）

**优化方案**:

```python
# engine/game_engine.py  _wolf_kill 并发化
async def _wolf_kill(self):
    alive_wolves = [p for p in self.game_state.players
                    if p.role_type == RoleType.WEREWOLF and p.is_alive]
    if not alive_wolves:
        return

    # 并发收集所有狼人决策
    tasks = []
    for wolf in alive_wolves:
        if wolf.agent:
            ctx = self.game_state.get_player_private_context(wolf.player_id)
            tasks.append(wolf.agent.decide_night_action(ctx))

    decisions = await asyncio.gather(*tasks, return_exceptions=True)

    kill_targets = []
    for wolf, decision in zip(alive_wolves, decisions):
        if isinstance(decision, Exception):
            self._log("error", f"狼人{wolf.player_id}决策失败: {decision}")
            continue
        target = decision.get("target")
        if target is not None and target != wolf.player_id:
            kill_targets.append(target)

    if kill_targets:
        target = self._count_vote(kill_targets)
        self.game_state.add_night_death(target)

# _vote 同理并发化
```

**预估提升**: 6人局从 ~60秒 降至 ~20秒

---

### 1.3 优化对话历史长度管理

**当前问题**:
- 对话历史越来越长，可能超出LLM上下文窗口

**优化方案**:

```python
def _build_dialogue_summary(self, dialogues: list) -> str:
    """智能压缩对话历史"""
    if not dialogues:
        return "暂无历史记录。"

    if len(dialogues) <= 15:
        # 历史较短，完整保留
        return self._format_dialogues(dialogues)

    # 历史较长：保留最近10条 + 早期摘要
    early = dialogues[:-10]
    recent = dialogues[-10:]

    early_summary = f"[早期共{len(early)}条记录] "
    # 统计早期关键事件
    deaths = [d for d in early if d.get("action") in ("night_kill", "vote")]
    if deaths:
        early_summary += f"关键事件：{len(deaths)}次死亡/投票。"

    return early_summary + "\n" + self._format_dialogues(recent)
```

---

## 🟠 P2: 能评测 — 多维可量化评测体系

### 2.1 评测维度设计

构建一个完整的评测体系，从**结果**和**过程**两个维度量化Agent表现：

```
评测体系
├── 结果评测（Result Evaluation）
│   ├── 胜率（Win Rate）：按角色/阵营统计
│   ├── 平均存活天数（Avg Survival Days）
│   ├── 关键行动正确率（Key Action Accuracy）
│   │   ├── 狼人首刀准确率（杀到神职/关键人物的比例）
│   │   ├── 预言家查验效率（验出狼人的速度）
│   │   ├── 女巫用药合理率（救人/毒人的正确性）
│   │   └── 投票放逐正确率（投对狼人的比例）
│   └── 阵营胜率差异（不同角色的胜率分布）
│
├── 过程评测（Process Evaluation）
│   ├── 发言质量评分（Speech Quality Score）
│   │   ├── 信息量（Information Density）
│   │   ├── 逻辑连贯性（Logic Coherence）
│   │   ├── 说服力（Persuasiveness）
│   │   └── 身份一致性（Identity Consistency）
│   ├── 决策合理性评分（Decision Rationality）
│   │   ├── 夜间行动合理性
│   │   ├── 投票选择合理性
│   │   └── 发言策略合理性
│   └── 信息利用效率（Information Utilization）
│       ├── 对历史发言的引用准确率
│       └── 推理链条完整度
│
└── 对抗评测（Adversarial Evaluation）
    ├── 狼人伪装成功率（不被识别的轮次数）
    ├── 好人识破率（正确识别狼人的比例）
    └── 悍跳成功率（假预言家被当真的比例）
```

### 2.2 实现评测模块

```python
# evaluator/game_evaluator.py（新文件）
from typing import Dict, List, Any

class GameEvaluator:
    """游戏评测器"""

    def __init__(self):
        self.result_evaluator = ResultEvaluator()
        self.process_evaluator = ProcessEvaluator()

    async def evaluate_game(
        self,
        game_record: Dict,
        dialogues: List[Dict],
        death_records: List[Dict],
        winner: str,
        role_assignment: Dict[int, str],
    ) -> Dict[str, Any]:
        """评测一局完整游戏"""
        result_scores = self.result_evaluator.evaluate(
            game_record, death_records, winner, role_assignment
        )
        process_scores = await self.process_evaluator.evaluate(
            dialogues, role_assignment, winner
        )
        return {
            "game_id": game_record.get("game_id"),
            "result_scores": result_scores,
            "process_scores": process_scores,
            "overall_score": self._calc_overall(result_scores, process_scores),
        }

class ResultEvaluator:
    """结果评测"""

    def evaluate(self, game_record, death_records, winner, role_assignment):
        scores = {}
        for pid, role in role_assignment.items():
            pid = int(pid)
            # 计算每个玩家的结果得分
            player_deaths = [d for d in death_records if d["player_id"] == pid]
            survival_day = player_deaths[0]["day"] if player_deaths else 99

            scores[pid] = {
                "role": role,
                "survival_days": survival_day,
                "is_winner": (
                    "good" if role in ("seer","witch","hunter","villager") else "evil"
                ) == winner,
                "death_cause": player_deaths[0]["cause"] if player_deaths else None,
            }
        return scores

class ProcessEvaluator:
    """过程评测（调用LLM评估发言质量）"""

    async def evaluate(self, dialogues, role_assignment, winner):
        """调用AI复盘官评估发言和决策质量"""
        scores = {}
        speeches = [d for d in dialogues if d.get("action") == "speech"]

        for speech in speeches:
            pid = speech.get("player_id")
            if pid not in scores:
                scores[pid] = {"speech_scores": [], "total_speeches": 0}

            # 使用复盘Agent评估单次发言质量
            score = await self._evaluate_single_speech(speech, role_assignment)
            scores[pid]["speech_scores"].append(score)
            scores[pid]["total_speeches"] += 1

        # 计算平均分
        for pid in scores:
            if scores[pid]["speech_scores"]:
                avg = sum(scores[pid]["speech_scores"]) / len(scores[pid]["speech_scores"])
                scores[pid]["avg_speech_quality"] = round(avg, 2)

        return scores

    async def _evaluate_single_speech(self, speech: Dict, role_assignment: Dict) -> float:
        """评估单次发言质量（1-10分）"""
        # 使用一个轻量级Agent评估发言
        # 评估维度：信息量、逻辑性、与身份的一致性
        evaluator_prompt = f"""评估以下狼人杀发言的质量（1-10分）：
        玩家角色：{role_assignment.get(speech['player_id'], 'unknown')}
        发言内容：{speech.get('content', '')}

        评分标准：
        - 信息量（是否有具体分析和推理）
        - 逻辑连贯性（论述是否自洽）
        - 身份一致性（发言是否符合其角色应有表现）

        只输出数字分数。"""

        # ... 调用LLM获取分数 ...
        return 5.0  # placeholder
```

### 2.3 复盘归因模块

```python
# evaluator/review_agent.py（新文件）
class ReviewAgent:
    """复盘归因Agent — 分析一局游戏中每个关键事件的因果链"""

    REVIEW_PROMPT = """你是一位资深的狼人杀复盘分析师。请对以下对局进行深度复盘。

## 对局信息
- 游戏ID: {game_id}
- 胜利方: {winner}
- 角色分配: {role_assignment}

## 关键事件时间线
{timeline}

## 要求
请分析以下内容，输出JSON格式：
{{
    "turning_points": [
        {{
            "day": 天数,
            "event": "关键事件描述",
            "impact": "high/medium/low",
            "analysis": "这个事件如何影响了游戏走向"
        }}
    ],
    "key_players": [
        {{
            "player_id": 玩家ID,
            "role": "角色",
            "contribution": "positive/negative/neutral",
            "analysis": "该玩家的关键贡献或失误"
        }}
    ],
    "win_reason": "胜利方的核心胜因",
    "lose_reason": "失败方的核心败因"
}}
"""

    async def review_game(self, game_record: Dict, dialogues: List[Dict]) -> Dict:
        """复盘一局游戏"""
        timeline = self._build_timeline(dialogues)
        prompt = self.REVIEW_PROMPT.format(
            game_id=game_record.get("game_id"),
            winner=game_record.get("winner"),
            role_assignment=game_record.get("role_assignment"),
            timeline=timeline,
        )
        result = await Runner.run(self.agent, prompt)
        return self._parse_json_output(result.final_output)
```

### 2.4 Leaderboard 排行榜

```python
# evaluator/leaderboard.py（新文件）
class Leaderboard:
    """Agent能力排行榜"""

    def __init__(self, filepath: str = "data/leaderboard.json"):
        self.filepath = filepath
        self.data = self._load()

    def update(self, model_name: str, version: str, game_result: Dict):
        """更新排行榜"""
        key = f"{model_name}@{version}"
        if key not in self.data:
            self.data[key] = {
                "model": model_name,
                "version": version,
                "games_played": 0,
                "wins": 0,
                "role_stats": {},  # {role: {games, wins}}
                "avg_speech_quality": 0,
                "avg_survival_days": 0,
            }

        entry = self.data[key]
        entry["games_played"] += 1
        if game_result.get("is_winner"):
            entry["wins"] += 1

        # 更新角色统计
        role = game_result.get("role", "unknown")
        if role not in entry["role_stats"]:
            entry["role_stats"][role] = {"games": 0, "wins": 0}
        entry["role_stats"][role]["games"] += 1
        if game_result.get("is_winner"):
            entry["role_stats"][role]["wins"] += 1

        self._save()

    def get_rankings(self) -> List[Dict]:
        """获取排名"""
        rankings = []
        for key, entry in self.data.items():
            win_rate = entry["wins"] / max(1, entry["games_played"])
            rankings.append({
                "model": entry["model"],
                "version": entry["version"],
                "games": entry["games_played"],
                "win_rate": round(win_rate, 3),
                "role_stats": entry["role_stats"],
            })
        return sorted(rankings, key=lambda x: x["win_rate"], reverse=True)
```

---

## 🟢 P3: 能进化 — 自进化Agent循环

### 3.1 自进化循环架构

```
对局结束
    ↓
复盘归因（ReviewAgent 分析关键事件和败因）
    ↓
经验提炼（从总结中提取可执行策略）
    ↓
策略更新（更新角色经验库 memory/experiences/）
    ↓
Prompt 注入（下一局游戏时加载最新经验到指令中）
    ↓
新一轮对局
    ↓
效果评估（对比进化前后的胜率变化）
    ↓
（循环）
```

### 3.2 增强经验沉淀机制

**当前问题**:
- 现有 `memory/experience.py` 只保存简单文本
- 没有结构化的策略提取
- 经验没有被有效利用

**优化方案**:

```python
# memory/experience.py 增强版

def save_experience(role_type: str, experience: Dict) -> None:
    """保存结构化经验"""
    _ensure_dir()
    filepath = _get_file_path(role_type)
    experiences = load_experiences(role_type)

    # 增加策略标签，便于后续检索
    enriched = {
        **experience,
        "strategy_tags": _extract_strategy_tags(experience),
        "timestamp": datetime.now().isoformat(),
        "game_version": "v1.0",
    }
    experiences.append(enriched)

    # 保留最近20条经验（避免无限增长）
    if len(experiences) > 20:
        experiences = experiences[-20:]

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(experiences, f, ensure_ascii=False, indent=2)

def _extract_strategy_tags(experience: Dict) -> List[str]:
    """从经验中提取策略标签"""
    tags = []
    if experience.get("is_winner"):
        tags.append("winning_strategy")
    else:
        tags.append("failure_case")

    role = experience.get("role_type", "")
    if role == "werewolf":
        if "悍跳" in experience.get("strategies", ""):
            tags.append("jump_seer")
        if "倒钩" in experience.get("strategies", ""):
            tags.append("reverse_hook")
    elif role == "seer":
        if "跳" in experience.get("strategies", ""):
            tags.append("reveal_early")
    # ... 更多标签提取 ...
    return tags

def get_experience_prompt(role_type: str, situation: str = "") -> str:
    """根据当前局势智能检索经验"""
    experiences = load_experiences(role_type)
    if not experiences:
        return ""

    # 优先选择与当前局势相关的经验
    relevant = _find_relevant_experiences(experiences, situation)
    if not relevant:
        relevant = experiences[-3:]  # 兜底：取最近3条

    lines = ["", "## 你的过往游戏经验"]
    for i, exp in enumerate(relevant, 1):
        outcome = "胜利" if exp.get("is_winner", False) else "失败"
        lines.append(f"\n--- 第{i}次经验（{outcome}）---")
        if exp.get("summary"):
            lines.append(f"  总结：{exp['summary']}")
        if exp.get("lessons"):
            lines.append(f"  建议：{exp['lessons']}")

    return "\n".join(lines)
```

### 3.3 自进化分析Agent

```python
# agent/evolution_agent.py（新文件）
class EvolutionAgent:
    """自进化分析Agent

    在多局游戏后，分析经验趋势，提炼出可执行的策略进化建议
    """

    EVOLUTION_PROMPT = """你是一个狼人杀策略进化分析师。
以下是{role_name}角色在最近{num_games}局游戏中的表现记录：

{experience_records}

请分析这些记录，输出JSON格式的进化建议：
{{
    "trend_analysis": "这些对局中体现出的整体趋势",
    "recurring_mistakes": "反复出现的错误模式",
    "effective_strategies": "被验证有效的策略",
    "evolution_plan": [
        {{
            "aspect": "需要进化的方面",
            "current": "当前行为",
            "target": "目标行为",
            "method": "如何实现这个转变"
        }}
    ],
    "prompt_update_suggestion": "对角色Prompt的具体修改建议"
}}
"""

    async def analyze_evolution(self, role_type: str, experiences: List[Dict]) -> Dict:
        """分析一个角色的进化方向"""
        role_name = {
            "werewolf": "狼人", "seer": "预言家",
            "witch": "女巫", "hunter": "猎人", "villager": "村民"
        }.get(role_type, role_type)

        records = self._format_experiences(experiences)
        prompt = self.EVOLUTION_PROMPT.format(
            role_name=role_name,
            num_games=len(experiences),
            experience_records=records,
        )

        result = await Runner.run(self.agent, prompt)
        return self._parse_json_output(result.final_output)
```

### 3.4 进化效果度量

```python
# evaluator/evolution_tracker.py（新文件）
class EvolutionTracker:
    """追踪自进化效果"""

    def __init__(self, filepath: str = "data/evolution_history.json"):
        self.filepath = filepath

    def record_evolution(self, role_type: str, version: int,
                         metrics_before: Dict, metrics_after: Dict):
        """记录一次进化的效果"""
        history = self._load_history()
        history.append({
            "role": role_type,
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "before": metrics_before,
            "after": metrics_after,
            "improvement": self._calc_improvement(metrics_before, metrics_after),
        })
        self._save_history(history)

    def get_evolution_report(self, role_type: str) -> Dict:
        """获取某个角色的进化报告"""
        history = self._load_history()
        role_history = [h for h in history if h["role"] == role_type]
        return {
            "total_evolutions": len(role_history),
            "trend": [h["improvement"] for h in role_history],
            "latest": role_history[-1] if role_history else None,
        }
```

---

## 📊 优化优先级矩阵

| 优化项 | 阶段 | 实施难度 | 预期收益 | 建议周期 |
|--------|------|----------|----------|----------|
| **Prompt优化** | P1 | 低 | **极高** | 2天 |
| 猎人智能决策 | P0 | 低 | 高 | 半天 |
| 平票PK/遗言 | P0 | 低 | 高 | 1天 |
| 异常处理增强 | P0 | 低 | 中 | 半天 |
| 并发决策 | P1 | 低 | 高 | 1天 |
| 对话历史压缩 | P1 | 低 | 中 | 半天 |
| 评测体系 | P2 | 中 | 高 | 2天 |
| 复盘归因 | P2 | 中 | 高 | 2天 |
| Leaderboard | P2 | 中 | 中 | 1天 |
| 自进化循环 | P3 | 中 | 高 | 3天 |
| 经验沉淀增强 | P3 | 低 | 中 | 1天 |
| 进化效果度量 | P3 | 低 | 中 | 1天 |

> Redis持久化、Docker容器化、WebSocket推送、CI/CD等架构优化延后，等核心业务跑通后再做。

---

## 🎯 实施计划

### 第1周：跑通 + 跑聪明

| 天 | 任务 | 产出 |
|----|------|------|
| Day 1 | 猎人智能决策 + 异常处理增强 | P0完成 |
| Day 2 | 平票PK + 遗言机制 | P0完成 |
| Day 3 | **Prompt全面重写**（系统提示 + 夜间 + 发言 + 投票） | P1核心 |
| Day 4 | 并发决策 + 对话历史压缩 | P1完成 |
| Day 5 | 跑测 main_demo.py + 前端验证 | 验收P0+P1 |

### 第2周：能评测

| 天 | 任务 | 产出 |
|----|------|------|
| Day 1 | 结果评测 + 过程评测模块 | 评测框架 |
| Day 2 | 复盘归因Agent | 复盘能力 |
| Day 3 | Leaderboard排行榜 | 排名展示 |
| Day 4 | 前端集成评测+排行榜展示 | UI更新 |
| Day 5 | 多模型对比测试 | 评测报告 |

### 第3周：能进化

| 天 | 任务 | 产出 |
|----|------|------|
| Day 1 | 经验沉淀增强 | 结构化经验 |
| Day 2 | 自进化分析Agent | 进化建议 |
| Day 3 | 进化效果度量 | 效果追踪 |
| Day 4 | 闭环测试（对局→评测→进化→再对局） | 完整闭环 |
| Day 5 | 全链路验收 | 最终验收 |

---

## 💡 总结

当前阶段的核心理念是 **"先聪明，再工程"**：

1. **Prompt是一切的基础** — 好的Prompt可以让Agent从"会玩"到"玩得好"，投入产出比最高
2. **评测是进化的前提** — 没有量化评测就无法衡量进步，也无法指导进化方向
3. **自进化是最终目标** — 通过"对局→评测→进化→对局"的闭环，让Agent持续变强

**预估提升**:
- 🧠 AI决策质量: 大幅提升（Prompt优化是核心）
- ⚡ 响应速度: 50-70%（并发决策）
- 📊 可观测性: 从"看日志"到"看排行榜+复盘报告"
- 🔄 自我进化: 从"固定策略"到"越玩越强"
