"""游戏引擎（裁判）

控制游戏流程，协调各角色行动，判定胜负
"""

import asyncio
from typing import List, Dict, Any, Optional
from collections import Counter

from engine.state import GameState
from engine.phase import GamePhase, TurnOrder
from engine.player import Player
from roles.base import RoleType
from roles.werewolf import Werewolf
from roles.seer import Seer
from roles.witch import Witch
from roles.hunter import Hunter
from roles.villager import Villager


class GameEngine:
    """游戏引擎（裁判）

    负责：
    1. 初始化游戏（分配角色、创建玩家代理）
    2. 推进游戏流程（白天/夜晚切换）
    3. 收集玩家决策
    4. 执行游戏规则（死亡判定、胜负判定）
    5. 管理游戏状态
    6. 记录游戏对话
    """

    def __init__(self, player_names: List[str], logger=None):
        self.player_names = player_names
        self.game_state = GameState()
        self.player_agents: Dict[int, Any] = {}
        self._is_running = False
        self.logger = logger
        self.death_records: List[Dict[str, Any]] = []
        self._night_death_causes: Dict[int, str] = {}
        self._step_index = 0
        self.game_id = None
        self._summaries_done = False
        self.summaries: List[Dict] = []

    def _log(self, level: str, msg: str):
        if self.logger:
            getattr(self.logger, level.lower())(msg)

    async def initialize(self, role_assignment: Dict[int, str], player_styles: Dict[int, str] = None):
        if player_styles is None:
            player_styles = {}

        for player_id, name in enumerate(self.player_names):
            role_type = role_assignment.get(player_id, "villager")
            role = self._create_role(role_type, player_id)
            player = Player(player_id=player_id, role=role, name=name)
            self.game_state.players.append(player)

        self.game_state.set_speaker_order([p.player_id for p in self.game_state.players])
        print(f"游戏初始化完成，{len(self.game_state.players)} 名玩家已就位。")

    def _create_role(self, role_type: str, player_id: int):
        role_map = {
            "werewolf": Werewolf,
            "seer": Seer,
            "witch": Witch,
            "hunter": Hunter,
            "villager": Villager,
        }
        role_class = role_map.get(role_type, Villager)
        return role_class(player_id=player_id)

    def set_player_agent(self, player_id: int, agent: Any):
        """设置玩家代理"""
        self.player_agents[player_id] = agent
        player = self.game_state.get_player(player_id)
        if player:
            player.agent = agent

    async def start(self, controller=None):
        """开始游戏"""
        self._is_running = True
        self._controller = controller
        print("=" * 50)
        print("狼人杀游戏开始！")
        print("=" * 50)

        while self._is_running and not self.game_state.is_game_over():
            await self._night_phase()
            if self.game_state.is_game_over():
                break
            await self._day_phase()
            self.game_state.day_number += 1

        await self._end_game()

    async def step(self) -> Dict[str, Any]:
        """逐步执行一个游戏阶段"""
        if not self.game_state.players:
            raise RuntimeError("Game not initialized. Call initialize() first.")

        if self._step_index == -1 or (self.game_state.is_game_over() and self._step_index not in (-1, 7, 8)):
            winner = self.game_state.get_winner()
            if self.game_state.is_game_over() and self._step_index not in (-1, 7, 8) and not self._summaries_done:
                self._step_index = 8
            return {
                "phase": "game_over",
                "day_number": self.game_state.day_number,
                "step_data": {},
                "players": [p.to_dict() for p in self.game_state.players],
                "dialogues": [],
                "deaths": list(self.death_records),
                "is_game_over": True,
                "winner": winner,
            }

        dialog_len_before = len(self.game_state.dialogues)
        death_len_before = len(self.death_records)
        step_data: Dict[str, Any] = {}
        phase_name = ""

        if self._step_index == 0:
            self.game_state.clear_night_deaths()
            self._night_death_causes.clear()
            self.game_state.phase = GamePhase.NIGHT_WOLF
            phase_name = "night_wolf"
            print(f"\n{'='*30} 第 {self.game_state.day_number} 夜 {'='*30}")
            await self._wolf_kill()
            new_dialogues = self.game_state.dialogues[dialog_len_before:]
            wolf_votes = [
                {"player_id": d["player_id"], "target": d.get("target"), "reasoning": d.get("reasoning")}
                for d in new_dialogues if d.get("action") == "night_kill"
            ]
            step_data = {
                "wolf_votes": wolf_votes,
                "final_target": self.game_state.night_deaths[-1] if self.game_state.night_deaths else None,
            }

        elif self._step_index == 1:
            self.game_state.phase = GamePhase.NIGHT_SEER
            phase_name = "night_seer"
            await self._seer_check()
            new_dialogues = self.game_state.dialogues[dialog_len_before:]
            seer_data = {}
            for d in new_dialogues:
                if d.get("action") == "seer_check":
                    seer_data = {
                        "seer_id": d["player_id"],
                        "target": d.get("target"),
                        "result": d.get("result"),
                        "reasoning": d.get("reasoning"),
                    }
            step_data = seer_data

        elif self._step_index == 2:
            self.game_state.phase = GamePhase.NIGHT_WITCH
            phase_name = "night_witch"
            await self._witch_action()
            new_dialogues = self.game_state.dialogues[dialog_len_before:]
            witch_data = {}
            for d in new_dialogues:
                if d.get("action") in ("heal", "poison"):
                    witch_data = {
                        "action": d["action"],
                        "target": d.get("target"),
                    }
            step_data = witch_data

        elif self._step_index == 3:
            phase_name = "night_result"
            await self._announce_night_deaths()
            await self._handle_hunter_death(list(self.game_state.night_deaths))
            new_deaths = self.death_records[death_len_before:]
            step_data = {
                "night_deaths": list(self.game_state.night_deaths),
                "deaths": new_deaths,
            }

        elif self._step_index == 4:
            self.game_state.phase = GamePhase.DAY_START
            phase_name = "day_start"
            print(f"\n{'='*30} 第 {self.game_state.day_number} 天 {'='*30}")
            await self._announce_day_start()
            step_data = {}

        elif self._step_index == 5:
            self.game_state.phase = GamePhase.SPEECH
            phase_name = "speech"
            await self._public_speeches()
            new_dialogues = self.game_state.dialogues[dialog_len_before:]
            speeches = [
                {"player_id": d["player_id"], "player_name": d.get("player_name"), "content": d.get("content")}
                for d in new_dialogues if d.get("action") == "speech"
            ]
            step_data = {"speeches": speeches}

        elif self._step_index == 6:
            self.game_state.phase = GamePhase.VOTE
            phase_name = "vote"
            await self._vote()
            new_dialogues = self.game_state.dialogues[dialog_len_before:]
            votes = {}
            for d in new_dialogues:
                if d.get("action") == "vote":
                    votes[d["player_id"]] = d.get("target")
            new_deaths = self.death_records[death_len_before:]
            eliminated = None
            for death in new_deaths:
                if death.get("cause") in ("vote", "shoot"):
                    eliminated = death["player_id"]
            step_data = {"votes": votes, "eliminated": eliminated}

        elif self._step_index == 7:
            phase_name = "day_end"
            game_over = self.game_state.is_game_over()
            winner = self.game_state.get_winner()
            if game_over:
                self._is_running = False
                self._step_index = 8
            else:
                self.game_state.day_number += 1
                self._step_index = 0

            new_dialogues = self.game_state.dialogues[dialog_len_before:]
            new_deaths = self.death_records[death_len_before:]
            step_data = {
                "game_over": game_over,
                "winner": winner,
                "next_day": self.game_state.day_number if not game_over else None,
            }
            return {
                "phase": phase_name,
                "day_number": self.game_state.day_number,
                "step_data": step_data,
                "players": [p.to_dict() for p in self.game_state.players],
                "dialogues": new_dialogues,
                "deaths": new_deaths,
                "is_game_over": game_over,
                "winner": winner,
            }

        elif self._step_index == 8:
            phase_name = "summary"
            self._step_index = -1
            self._summaries_done = True
            winner = self.game_state.get_winner()
            step_data = {"summaries_complete": True, "summaries": list(self.summaries)}
            return {
                "phase": phase_name,
                "day_number": self.game_state.day_number,
                "step_data": step_data,
                "players": [p.to_dict() for p in self.game_state.players],
                "dialogues": [],
                "deaths": list(self.death_records),
                "is_game_over": True,
                "winner": winner,
            }

        new_dialogues = self.game_state.dialogues[dialog_len_before:]
        new_deaths = self.death_records[death_len_before:]
        self._step_index += 1

        return {
            "phase": phase_name,
            "day_number": self.game_state.day_number,
            "step_data": step_data,
            "players": [p.to_dict() for p in self.game_state.players],
            "dialogues": new_dialogues,
            "deaths": new_deaths,
            "is_game_over": False,
            "winner": None,
        }

    async def _night_phase(self):
        """执行夜晚阶段"""
        day = self.game_state.day_number
        print(f"\n{'='*30} 第 {day} 夜 {'='*30}")

        self.game_state.clear_night_deaths()
        self._night_death_causes.clear()

        night_order = TurnOrder.get_night_order()
        for phase in night_order:
            if self.game_state.is_game_over():
                break
            self.game_state.phase = phase
            print(f"\n[{phase.value}]")

            if phase == GamePhase.NIGHT_WOLF:
                await self._wolf_kill()
            elif phase == GamePhase.NIGHT_SEER:
                await self._seer_check()
            elif phase == GamePhase.NIGHT_WITCH:
                await self._witch_action()

        await self._announce_night_deaths()
        await self._handle_hunter_death(self.game_state.night_deaths)

    async def _wolf_kill(self):
        """狼人杀人阶段"""
        alive_wolves = [
            p for p in self.game_state.players
            if p.role_type == RoleType.WEREWOLF and p.is_alive
        ]

        if not alive_wolves:
            return

        kill_targets = []
        for wolf in alive_wolves:
            agent = self.player_agents.get(wolf.player_id)
            if agent:
                game_state = self.game_state.get_player_private_context(wolf.player_id)
                decision = await agent.decide_night_action(game_state)
                target = decision.get("target")
                reasoning = decision.get("reasoning", "")
                if target is not None and target != wolf.player_id:
                    kill_targets.append(target)
                print(f"狼人 {wolf.player_id} 决策：击杀玩家 {target}")
                self.game_state.dialogues.append({
                    "day": self.game_state.day_number,
                    "phase": "狼人杀人",
                    "player_id": wolf.player_id,
                    "player_name": wolf.name,
                    "role": wolf.role.name,
                    "action": "night_kill",
                    "target": target,
                    "reasoning": reasoning,
                })

        if kill_targets:
            target = self._count_vote(kill_targets)
            self.game_state.add_night_death(target)
            self._night_death_causes[target] = "night_kill"
            print(f"狼人今晚击杀：玩家 {target}")

    async def _seer_check(self):
        """预言家查验阶段"""
        alive_seers = [
            p for p in self.game_state.players
            if p.role_type == RoleType.SEER and p.is_alive
        ]

        for seer in alive_seers:
            agent = self.player_agents.get(seer.player_id)
            if agent:
                game_state = self.game_state.get_player_private_context(seer.player_id)
                decision = await agent.decide_night_action(game_state)
                target = decision.get("target")
                reasoning = decision.get("reasoning", "")
                if target is not None:
                    player = self.game_state.get_player(target)
                    if player:
                        result = "wolf" if player.role_type == RoleType.WEREWOLF else "good"
                        print(f"预言家 {seer.player_id} 查验玩家 {target}：{result}")
                        seer.role._check_result = {"target": target, "result": result}
                        self.game_state.dialogues.append({
                            "day": self.game_state.day_number,
                            "phase": "预言家查验",
                            "player_id": seer.player_id,
                            "player_name": seer.name,
                            "role": seer.role.name,
                            "action": "seer_check",
                            "target": target,
                            "result": result,
                            "reasoning": reasoning,
                        })

    async def _witch_action(self):
        """女巫用药阶段"""
        alive_witches = [
            p for p in self.game_state.players
            if p.role_type == RoleType.WITCH and p.is_alive
        ]

        tonight_death = self.game_state.night_deaths[-1] if self.game_state.night_deaths else None

        for witch in alive_witches:
            agent = self.player_agents.get(witch.player_id)
            if agent and (witch.role.has_heal or witch.role.has_poison):
                game_state = self.game_state.get_player_private_context(witch.player_id)
                game_state["tonight_death"] = tonight_death
                decision = await agent.decide_night_action(game_state)

                action = decision.get("action", "")
                target = decision.get("target")

                used_heal = "heal" in action and tonight_death is not None and witch.role.has_heal
                used_poison = "poison" in action and target is not None and witch.role.has_poison and target != witch.player_id

                if used_heal:
                    witch.role.use_heal()
                    if tonight_death in self.game_state.night_deaths:
                        self.game_state.night_deaths.remove(tonight_death)
                    print(f"女巫 {witch.player_id} 使用救人药，救活了玩家 {tonight_death}")
                elif used_poison:
                    witch.role.use_poison()
                    self.game_state.add_night_death(target)
                    self._night_death_causes[target] = "poison"
                    print(f"女巫 {witch.player_id} 使用毒药，毒死了玩家 {target}")

                if used_heal or used_poison:
                    self.game_state.dialogues.append({
                        "day": self.game_state.day_number,
                        "phase": "女巫用药",
                        "player_id": witch.player_id,
                        "player_name": witch.name,
                        "role": witch.role.name,
                        "action": "heal" if used_heal else "poison",
                        "target": tonight_death if used_heal else (target if used_poison else None),
                    })

    async def _handle_hunter_death(self, killed_player_ids: List[int]):
        """处理猎人死亡开枪"""
        for player_id in killed_player_ids:
            player = self.game_state.get_player(player_id)
            if player and player.role_type == RoleType.HUNTER and player.role.can_shoot:
                alive_players = self.game_state.get_alive_players()
                if alive_players:
                    target = alive_players[0].player_id
                    player.role.lock_shoot()
                    target_player = self.game_state.get_player(target)
                    if target_player:
                        target_player.role.is_alive = False
                        print(f"猎人 {player_id} 开枪带走了玩家 {target}")
                        self.death_records.append({
                            "player_id": target,
                            "player_name": target_player.name,
                            "role": target_player.role.name,
                            "cause": "shoot",
                            "day": self.game_state.day_number,
                        })
                        self.game_state.dialogues.append({
                            "day": self.game_state.day_number,
                            "phase": "猎人开枪",
                            "player_id": player_id,
                            "player_name": player.name,
                            "role": player.role.name,
                            "action": "hunter_shot",
                            "target": target,
                        })

    async def _announce_night_deaths(self):
        """宣布夜晚死亡"""
        if self.game_state.night_deaths:
            for player_id in self.game_state.night_deaths:
                player = self.game_state.get_player(player_id)
                if player:
                    cause = self._night_death_causes.get(player_id, "night_kill")
                    player.kill(cause, self.game_state.to_dict())
                    print(f"玩家 {player_id} ({player.role.name}) 死亡")
                    self.death_records.append({
                        "player_id": player_id,
                        "player_name": player.name,
                        "role": player.role.name,
                        "cause": cause,
                        "day": self.game_state.day_number,
                    })
        else:
            print("今晚无人死亡")

    async def _day_phase(self):
        """执行白天阶段"""
        print(f"\n{'='*30} 第 {self.game_state.day_number} 天 {'='*30}")

        self.game_state.phase = GamePhase.DAY_START
        await self._announce_day_start()

        self.game_state.phase = GamePhase.SPEECH
        await self._public_speeches()

        self.game_state.phase = GamePhase.VOTE
        await self._vote()

    async def _announce_day_start(self):
        """宣布白天开始"""
        if self.game_state.night_deaths:
            death_names = [f"玩家{p}" for p in self.game_state.night_deaths]
            print(f"昨晚死亡：{', '.join(death_names)}")
        else:
            print("昨晚是平安夜，无人死亡")

    async def _public_speeches(self):
        """公开演讲阶段"""
        alive_players = self.game_state.get_alive_players()
        print(f"\n公开演讲开始，共 {len(alive_players)} 名存活玩家")

        for player in alive_players:
            agent = self.player_agents.get(player.player_id)
            if agent:
                game_state = self.game_state.get_player_private_context(player.player_id)
                decision = await agent.decide_speech(game_state)
                content = decision.get("content", "")
                print(f"\n{player.name} 发言：")
                print(content)
                self.game_state.dialogues.append({
                    "day": self.game_state.day_number,
                    "phase": "公开演讲",
                    "player_id": player.player_id,
                    "player_name": player.name,
                    "role": player.role.name,
                    "action": "speech",
                    "content": content,
                })

    async def _vote(self):
        """投票阶段"""
        alive_players = self.game_state.get_alive_players()

        votes = {}
        for player in alive_players:
            agent = self.player_agents.get(player.player_id)
            if agent:
                game_state = self.game_state.get_player_private_context(player.player_id)
                decision = await agent.decide_vote(game_state)
                target = decision.get("target")
                if target is not None and self.game_state.get_player(target) and self.game_state.get_player(target).is_alive:
                    votes[player.player_id] = target
                    self.game_state.add_vote(player.player_id, target)
                    self.game_state.dialogues.append({
                        "day": self.game_state.day_number,
                        "phase": "投票",
                        "player_id": player.player_id,
                        "player_name": player.name,
                        "role": player.role.name,
                        "action": "vote",
                        "target": target,
                    })

        vote_count = {}
        for target in votes.values():
            vote_count[target] = vote_count.get(target, 0) + 1

        if vote_count:
            max_votes = max(vote_count.values())
            eliminated = [p for p, c in vote_count.items() if c == max_votes]

            if len(eliminated) == 1:
                player = self.game_state.get_player(eliminated[0])
                print(f"\n玩家 {eliminated[0]} ({player.name}) 被投票出局")
                player.kill("vote", self.game_state.to_dict())
                self.death_records.append({
                    "player_id": eliminated[0],
                    "player_name": player.name,
                    "role": player.role.name,
                    "cause": "vote",
                    "day": self.game_state.day_number,
                })
                await self._handle_hunter_death([eliminated[0]])
            else:
                print(f"\n平票，进入PK：{eliminated}")

        self.game_state.reset_vote_record()

    async def _end_game(self):
        """游戏结束"""
        winner = self.game_state.get_winner()
        print(f"\n{'='*50}")
        print("游戏结束！")
        print(f"胜利方：{'善良阵营' if winner == 'good' else '邪恶阵营'}")
        print(f"{'='*50}")
        self._is_running = False

    def _count_vote(self, targets: List[int]) -> int:
        count = Counter(targets)
        return count.most_common(1)[0][0]

    def stop(self):
        self._is_running = False


async def create_game(player_names: List[str], role_assignment: Dict[int, str]) -> GameEngine:
    """工厂函数：创建并初始化游戏"""
    engine = GameEngine(player_names)
    await engine.initialize(role_assignment)
    return engine