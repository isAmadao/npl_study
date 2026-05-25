"""
AI狼人杀 - 游戏引擎核心
Phase 1: 状态机 + 规则引擎 + 角色/行动定义 + 规则AI
"""
from __future__ import annotations

import random
import json
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Any

# ═══════════════════════════════════════════════
# 角色定义
# ═══════════════════════════════════════════════

class Team(Enum):
    VILLAGER = "村民阵营"
    WEREWOLF = "狼人阵营"

class Role(Enum):
    WEREWOLF = ("狼人", Team.WEREWOLF)
    SEER = ("预言家", Team.VILLAGER)
    WITCH = ("女巫", Team.VILLAGER)
    VILLAGER = ("村民", Team.VILLAGER)

    @property
    def label(self) -> str:
        return self.value[0]

    @property
    def team(self) -> Team:
        return self.value[1]


# ═══════════════════════════════════════════════
# 行动定义
# ═══════════════════════════════════════════════

class ActionType(Enum):
    KILL = "狼人杀人"
    CHECK = "预言家查验"
    SAVE = "女巫救人"
    POISON = "女巫毒人"
    VOTE = "投票放逐"
    SPEAK = "发言"
    SKIP = "跳过"

@dataclass
class Action:
    actor_id: int
    action_type: ActionType
    target_id: Optional[int] = None
    content: str = ""  # 发言内容 或 额外信息


# ═══════════════════════════════════════════════
# 游戏阶段
# ═══════════════════════════════════════════════

class Phase(Enum):
    INIT = "初始化"
    NIGHT_WEREWOLF = "夜晚-狼人行动"
    NIGHT_SEER = "夜晚-预言家查验"
    NIGHT_WITCH = "夜晚-女巫行动"
    DAY_ANNOUNCE = "天亮-公布死讯"
    DAY_DISCUSS = "白天-自由发言"
    DAY_VOTE = "白天-投票放逐"
    GAME_OVER = "游戏结束"

PHASE_ORDER = [
    Phase.NIGHT_WEREWOLF,
    Phase.NIGHT_SEER,
    Phase.NIGHT_WITCH,
    Phase.DAY_ANNOUNCE,
    Phase.DAY_DISCUSS,
    Phase.DAY_VOTE,
]


# ═══════════════════════════════════════════════
# 玩家状态
# ═══════════════════════════════════════════════

@dataclass
class Player:
    id: int
    name: str
    role: Role
    alive: bool = True
    is_human: bool = False

    def __repr__(self):
        status = "存活" if self.alive else "死亡"
        return f"Player({self.id}, {self.name}, {self.role.label}, {status})"


# ═══════════════════════════════════════════════
# 游戏状态
# ═══════════════════════════════════════════════

@dataclass
class GameState:
    players: list[Player]
    phase: Phase = Phase.INIT
    round_num: int = 0
    winner: Optional[Team] = None

    # 夜晚行动记录（本回合）
    werewolf_target: Optional[int] = None   # 狼人选中的目标
    seer_check_target: Optional[int] = None  # 预言家查验目标
    seer_check_result: Optional[str] = None   # 查验结果
    witch_save_used: bool = False
    witch_poison_used: bool = False
    witch_save_target: Optional[int] = None
    witch_poison_target: Optional[int] = None

    # 本回合死亡
    deaths_this_round: list[int] = field(default_factory=list)

    # 日志
    event_log: list[dict] = field(default_factory=list)

    # 投票
    votes: dict[int, int] = field(default_factory=dict)

    def log(self, event_type: str, data: dict[str, Any]):
        entry = {"round": self.round_num, "phase": self.phase.value, "type": event_type, **data}
        self.event_log.append(entry)

    def alive_players(self) -> list[Player]:
        return [p for p in self.players if p.alive]

    def alive_werewolves(self) -> list[Player]:
        return [p for p in self.players if p.alive and p.role == Role.WEREWOLF]

    def alive_villagers(self) -> list[Player]:
        return [p for p in self.players if p.alive and p.role.team == Team.VILLAGER]

    def get_player(self, pid: int) -> Player:
        for p in self.players:
            if p.id == pid:
                return p
        raise ValueError(f"Player {pid} not found")


# ═══════════════════════════════════════════════
# 规则引擎
# ═══════════════════════════════════════════════

class RuleEngine:
    """裁决行动合法性、结算行动、判定胜负"""

    @staticmethod
    def check_win(state: GameState) -> Optional[Team]:
        werewolves = len(state.alive_werewolves())
        villagers = len(state.alive_villagers())
        if werewolves == 0:
            return Team.VILLAGER
        if werewolves >= villagers:
            return Team.WEREWOLF
        return None

    @staticmethod
    def resolve_night_actions(state: GameState) -> list[int]:
        """结算夜晚行动，返回本回合死亡玩家ID列表"""
        deaths: list[int] = []
        killed_by_wolves: Optional[int] = state.werewolf_target

        # 女巫救人
        saved = False
        if state.witch_save_target is not None and state.witch_save_target == killed_by_wolves:
            saved = True

        # 女巫毒人
        if state.witch_poison_target is not None:
            deaths.append(state.witch_poison_target)

        # 狼人杀人（未被救）
        if killed_by_wolves is not None and not saved:
            deaths.append(killed_by_wolves)

        return list(set(deaths))  # 去重

    @staticmethod
    def resolve_vote(state: GameState) -> Optional[int]:
        """统计投票，返回被放逐的玩家ID（平票返回None）"""
        if not state.votes:
            return None
        tally: dict[int, int] = {}
        for voter, target in state.votes.items():
            tally[target] = tally.get(target, 0) + 1
        max_votes = max(tally.values())
        top = [pid for pid, cnt in tally.items() if cnt == max_votes]
        if len(top) == 1:
            return top[0]
        return None  # 平票


# ═══════════════════════════════════════════════
# 规则AI（Phase 1 使用，Phase 2 替换为LLM）
# ═══════════════════════════════════════════════

class RuleAgent:
    """基于启发式策略的简单AI"""

    @staticmethod
    def werewolf_act(state: GameState, player: Player) -> Action:
        """狼人：随机选择一名存活的好人作为目标"""
        werewolf_ids = {p.id for p in state.players if p.role == Role.WEREWOLF and p.alive}
        targets = [p for p in state.alive_players() if p.id not in werewolf_ids]
        if not targets:
            return Action(player.id, ActionType.SKIP)
        target = random.choice(targets)
        return Action(player.id, ActionType.KILL, target.id, f"选择击杀 {target.name}")

    @staticmethod
    def seer_act(state: GameState, player: Player) -> Action:
        """预言家：随机查验一名未查验过的存活玩家"""
        checked = set()
        for log in state.event_log:
            if log.get("type") == "seer_check" and log.get("actor_id") == player.id:
                checked.add(log.get("target_id"))
        candidates = [p for p in state.alive_players() if p.id != player.id and p.id not in checked]
        if not candidates:
            candidates = [p for p in state.alive_players() if p.id != player.id]
        target = random.choice(candidates) if candidates else None
        if target is None:
            return Action(player.id, ActionType.SKIP)
        return Action(player.id, ActionType.CHECK, target.id)

    @staticmethod
    def witch_act(state: GameState, player: Player) -> Action:
        """女巫：首夜救人，其他夜晚50%概率用毒"""
        if not state.witch_save_used and state.werewolf_target is not None:
            return Action(player.id, ActionType.SAVE, state.werewolf_target, "使用解药救人")
        if not state.witch_poison_used and random.random() < 0.5:
            werewolf_ids = {p.id for p in state.players if p.role == Role.WEREWOLF and p.alive}
            candidates = [p for p in state.alive_players() if p.id != player.id and p.id not in werewolf_ids]
            if candidates:
                target = random.choice(candidates)
                return Action(player.id, ActionType.POISON, target.id, f"使用毒药毒杀 {target.name}")
        return Action(player.id, ActionType.SKIP)

    @staticmethod
    def villager_speak(state: GameState, player: Player) -> Action:
        """村民发言：随机发言"""
        speeches = [
            "我是好人，没有信息，跟票走。",
            "过，没信息。",
            "听预言家报查验，我跟预言家走。",
            "我觉得可以投发言不好的。",
            "我是村民，过。",
        ]
        return Action(player.id, ActionType.SPEAK, content=random.choice(speeches))

    @staticmethod
    def werewolf_speak(state: GameState, player: Player) -> Action:
        """狼人发言：伪装成好人"""
        speeches = [
            "我是好人，过。",
            "我也没信息，听预言家的吧。",
            "我觉得可以先把不说话的人投出去。",
            "我是村民，过。",
        ]
        return Action(player.id, ActionType.SPEAK, content=random.choice(speeches))

    @staticmethod
    def seer_speak(state: GameState, player: Player) -> Action:
        """预言家发言：报查验结果"""
        checked = []
        for log in state.event_log:
            if log.get("type") == "seer_check" and log.get("actor_id") == player.id:
                tid = log.get("target_id")
                result = log.get("result", "")
                tplayer = state.get_player(tid)
                checked.append((tplayer.name, result))
        if checked:
            reports = [f"查验{name}，身份是{res}" for name, res in checked]
            return Action(player.id, ActionType.SPEAK, content=f"我是预言家。{'；'.join(reports)}。今天出狼。")
        return Action(player.id, ActionType.SPEAK, content="我是预言家，还没查验。过。")

    @staticmethod
    def vote(state: GameState, player: Player, is_werewolf: bool) -> Action:
        """投票：随机投给一名存活的其他玩家"""
        candidates = [p for p in state.alive_players() if p.id != player.id]
        if not candidates:
            return Action(player.id, ActionType.SKIP)
        # 狼人尽量不投狼队友
        if is_werewolf:
            non_wolf = [p for p in candidates if p.role != Role.WEREWOLF]
            if non_wolf:
                target = random.choice(non_wolf)
            else:
                target = random.choice(candidates)
        else:
            target = random.choice(candidates)
        return Action(player.id, ActionType.VOTE, target.id)

    @classmethod
    def get_action(cls, state: GameState, player: Player) -> Action:
        phase = state.phase
        is_wolf = player.role == Role.WEREWOLF

        if phase == Phase.NIGHT_WEREWOLF and is_wolf:
            return cls.werewolf_act(state, player)
        elif phase == Phase.NIGHT_SEER and player.role == Role.SEER:
            return cls.seer_act(state, player)
        elif phase == Phase.NIGHT_WITCH and player.role == Role.WITCH:
            return cls.witch_act(state, player)
        elif phase == Phase.DAY_DISCUSS:
            if is_wolf:
                return cls.werewolf_speak(state, player)
            elif player.role == Role.SEER:
                return cls.seer_speak(state, player)
            else:
                return cls.villager_speak(state, player)
        elif phase == Phase.DAY_VOTE:
            return cls.vote(state, player, is_wolf)
        return Action(player.id, ActionType.SKIP)


# ═══════════════════════════════════════════════
# 状态机
# ═══════════════════════════════════════════════

class WerewolfGame:
    """狼人杀游戏主控"""

    # 6人局角色分配
    ROLE_POOL = [Role.WEREWOLF, Role.WEREWOLF, Role.SEER, Role.WITCH, Role.VILLAGER, Role.VILLAGER]

    def __init__(self, player_names: Optional[list[str]] = None, seed: Optional[int] = None,
                 agent=None):
        if seed is not None:
            random.seed(seed)
        names = player_names or [f"玩家{i+1}" for i in range(6)]
        roles = list(self.ROLE_POOL)
        random.shuffle(roles)
        self.state = GameState(
            players=[Player(i, names[i], roles[i]) for i in range(6)]
        )
        self.engine = RuleEngine()
        self._phase_index = 0
        self.state.round_num = 1
        self.state.phase = Phase.NIGHT_WEREWOLF
        # 可插拔 Agent：默认使用规则AI
        self._get_action = agent or RuleAgent.get_action

    # ── 运行入口 ──

    def run(self, verbose: bool = True) -> GameState:
        """运行完整对局，返回最终游戏状态"""
        while self.state.phase != Phase.GAME_OVER:
            self._step(verbose)
        return self.state

    # ── 单步 ──

    def _step(self, verbose: bool):
        phase = self.state.phase
        if verbose:
            print(f"\n{'='*50}")
            print(f"  第{self.state.round_num}轮 - {phase.value}")
            print(f"{'='*50}")

        if phase == Phase.NIGHT_WEREWOLF:
            self._handle_werewolf_night(verbose)
        elif phase == Phase.NIGHT_SEER:
            self._handle_seer_night(verbose)
        elif phase == Phase.NIGHT_WITCH:
            self._handle_witch_night(verbose)
        elif phase == Phase.DAY_ANNOUNCE:
            self._handle_day_announce(verbose)
        elif phase == Phase.DAY_DISCUSS:
            self._handle_day_discuss(verbose)
        elif phase == Phase.DAY_VOTE:
            self._handle_day_vote(verbose)

        # 检查胜负
        winner = self.engine.check_win(self.state)
        if winner:
            self.state.winner = winner
            self.state.phase = Phase.GAME_OVER
            self.state.log("game_over", {"winner": winner.value})
            if verbose:
                print(f"\n{'='*50}")
                print(f"  🏆 游戏结束！{winner.value}获胜！")
                print(f"{'='*50}")
            return

        # 推进阶段
        self._advance_phase()

    # ── 夜晚：狼人 ──

    def _handle_werewolf_night(self, verbose: bool):
        wolves = [p for p in self.state.alive_players() if p.role == Role.WEREWOLF]
        if not wolves:
            return
        # 简单策略：第一个狼人做决定
        leader = wolves[0]
        action = self._get_action(self.state, leader)
        self.state.werewolf_target = action.target_id
        self.state.log("werewolf_kill", {"actor_id": leader.id, "target_id": action.target_id})
        if verbose:
            target_name = self.state.get_player(action.target_id).name if action.target_id else "无"
            print(f"  🐺 狼人选杀目标：{target_name}")

    # ── 夜晚：预言家 ──

    def _handle_seer_night(self, verbose: bool):
        seer = next((p for p in self.state.alive_players() if p.role == Role.SEER), None)
        if seer is None:
            return
        action = self._get_action(self.state, seer)
        self.state.seer_check_target = action.target_id
        if action.target_id is not None:
            target = self.state.get_player(action.target_id)
            result = "狼人" if target.role.team == Team.WEREWOLF else "好人"
            self.state.seer_check_result = result
            self.state.log("seer_check", {"actor_id": seer.id, "target_id": action.target_id, "result": result})
            if verbose:
                print(f"  🔮 预言家查验 {target.name}：{result}")
        else:
            if verbose:
                print(f"  🔮 预言家跳过查验")

    # ── 夜晚：女巫 ──

    def _handle_witch_night(self, verbose: bool):
        witch = next((p for p in self.state.alive_players() if p.role == Role.WITCH), None)
        if witch is None:
            return
        action = self._get_action(self.state, witch)
        if action.action_type == ActionType.SAVE and action.target_id is not None:
            self.state.witch_save_used = True
            self.state.witch_save_target = action.target_id
            self.state.log("witch_save", {"actor_id": witch.id, "target_id": action.target_id})
            if verbose:
                print(f"  💚 女巫使用解药救 {self.state.get_player(action.target_id).name}")
        elif action.action_type == ActionType.POISON and action.target_id is not None:
            self.state.witch_poison_used = True
            self.state.witch_poison_target = action.target_id
            self.state.log("witch_poison", {"actor_id": witch.id, "target_id": action.target_id})
            if verbose:
                print(f"  🧪 女巫使用毒药毒杀 {self.state.get_player(action.target_id).name}")
        else:
            if verbose:
                print(f"  💚 女巫本回合不行动")

    # ── 天亮 ──

    def _handle_day_announce(self, verbose: bool):
        deaths = self.engine.resolve_night_actions(self.state)
        self.state.deaths_this_round = deaths
        for pid in deaths:
            player = self.state.get_player(pid)
            player.alive = False
            self.state.log("player_death", {"player_id": pid, "name": player.name, "role": player.role.label})
        if verbose:
            if deaths:
                for pid in deaths:
                    p = self.state.get_player(pid)
                    print(f"  ☠️  天亮后 {p.name} 死亡，身份：{p.role.label}")
            else:
                print(f"  🌅 平安夜，无人死亡")

    # ── 白天发言 ──

    def _handle_day_discuss(self, verbose: bool):
        alive = self.state.alive_players()
        for player in alive:
            action = self._get_action(self.state, player)
            self.state.log("speech", {"actor_id": player.id, "name": player.name, "content": action.content})
            if verbose:
                print(f"  💬 [{player.name}]({player.role.label})：{action.content}")

    # ── 投票 ──

    def _handle_day_vote(self, verbose: bool):
        self.state.votes.clear()
        alive = self.state.alive_players()
        for player in alive:
            action = self._get_action(self.state, player)
            if action.target_id is not None:
                self.state.votes[player.id] = action.target_id

        eliminated = self.engine.resolve_vote(self.state)
        vote_tally = self._compute_vote_tally()

        if verbose:
            for pid, cnt in vote_tally.items():
                print(f"  🗳️  {self.state.get_player(pid).name} 获得 {cnt} 票")

        if eliminated is not None:
            player = self.state.get_player(eliminated)
            player.alive = False
            self.state.log("vote_eliminate", {"player_id": eliminated, "name": player.name, "role": player.role.label, "votes": vote_tally})
            if verbose:
                print(f"  🚫 {player.name} 被投票放逐，身份：{player.role.label}")
        else:
            self.state.log("vote_tie", {"message": "平票，无人被放逐"})
            if verbose:
                print(f"  🤝 平票，无人被放逐")

        # 进入新一轮
        self.state.round_num += 1
        # 重置夜晚状态
        self.state.werewolf_target = None
        self.state.seer_check_target = None
        self.state.seer_check_result = None
        self.state.witch_save_target = None
        self.state.witch_poison_target = None
        self.state.deaths_this_round = []
        self.state.votes.clear()

    def _compute_vote_tally(self) -> dict[int, int]:
        """统计投票分布"""
        result: dict[int, int] = {}
        for voter, target in self.state.votes.items():
            result[target] = result.get(target, 0) + 1
        return result

    # ── 阶段推进 ──

    def _advance_phase(self):
        if self.state.winner:
            self.state.phase = Phase.GAME_OVER
            return
        idx = PHASE_ORDER.index(self.state.phase)
        self.state.phase = PHASE_ORDER[(idx + 1) % len(PHASE_ORDER)]

    # ── 导出日志 ──

    def export_log(self) -> str:
        return json.dumps(self.state.event_log, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════

def create_default_game(seed: Optional[int] = None) -> WerewolfGame:
    """创建6人标准局"""
    return WerewolfGame(seed=seed)

def print_roles(game: WerewolfGame):
    """Print role assignment (debug only)"""
    print()
    print("Roles:")
    for p in game.state.players:
        print(f"  {p.name}: {p.role.label} ({p.role.team.value})")
