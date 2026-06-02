"""游戏状态管理"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from engine.player import Player
from engine.phase import GamePhase
from roles.base import RoleType


@dataclass
class GameState:
    """游戏状态记录"""
    players: List[Player] = field(default_factory=list)
    phase: GamePhase = GamePhase.DAY_START
    day_number: int = 1
    speaker_order: List[int] = field(default_factory=list)
    current_speaker: int = 0
    last_words: List[Dict[str, Any]] = field(default_factory=list)
    vote_record: List[Dict[str, Any]] = field(default_factory=list)
    night_deaths: List[int] = field(default_factory=list)
    dialogues: List[Dict[str, Any]] = field(default_factory=list)

    def get_alive_players(self) -> List[Player]:
        return [p for p in self.players if p.is_alive]

    def get_player(self, player_id: int) -> Optional[Player]:
        for p in self.players:
            if p.player_id == player_id:
                return p
        return None

    def get_players_by_role(self, role_type: str) -> List[Player]:
        return [p for p in self.players if p.role_type.value == role_type]

    def get_players_by_camp(self, camp: str) -> List[Player]:
        return [p for p in self.players if p.camp.value == camp]

    def is_game_over(self) -> bool:
        alive_players = self.get_alive_players()
        if not alive_players:
            return True

        alive_wolves = [p for p in alive_players if p.role_type.value == "werewolf"]
        if not alive_wolves:
            return True

        alive_good = [p for p in alive_players if p.camp.value == "good"]
        if not alive_good:
            return True

        return False

    def get_winner(self) -> Optional[str]:
        if not self.is_game_over():
            return None

        alive_players = self.get_alive_players()
        alive_wolves = [p for p in alive_players if p.role_type.value == "werewolf"]

        if not alive_wolves:
            return "good"
        else:
            return "evil"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "day_number": self.day_number,
            "phase": self.phase.value,
            "players": [p.to_dict() for p in self.players],
            "alive_players": [p.player_id for p in self.get_alive_players()],
            "speaker_order": self.speaker_order,
            "current_speaker": self.current_speaker,
            "last_words": self.last_words,
            "vote_record": self.vote_record,
            "night_deaths": self.night_deaths,
        }

    def get_public_context(self) -> Dict[str, Any]:
        return {
            "day_number": self.day_number,
            "phase": self.phase.value,
            "alive_players": [
                {"player_id": p.player_id, "name": p.name}
                for p in self.get_alive_players()
            ],
            "last_words": self.last_words,
            "vote_record": self.vote_record,
            "night_deaths": self.night_deaths,
            "dialogues": self.dialogues,
        }

    def get_player_private_context(self, player_id: int) -> Dict[str, Any]:
        player = self.get_player(player_id)
        if not player:
            return {}

        public_ctx = self.get_public_context()
        private_ctx = player.role.get_private_context()
        visible_dialogues = self._filter_visible_dialogues(player)

        return {
            **public_ctx,
            **private_ctx,
            "dialogues": visible_dialogues,
            "other_players": [
                {
                    "player_id": p.player_id,
                    "name": p.name,
                    "is_alive": p.is_alive,
                    "is_sheriff": p.is_sheriff,
                }
                for p in self.players if p.player_id != player_id
            ],
        }

    def _filter_visible_dialogues(self, player: Player) -> List[Dict[str, Any]]:
        visible = []
        for d in self.dialogues:
            if d["phase"] in ["公开演讲", "投票"]:
                visible.append(d)
            elif d["player_id"] == player.player_id:
                visible.append(d)
            elif d["action"] == "night_kill" and player.role_type == RoleType.WEREWOLF:
                target_player = self.get_player(d["player_id"])
                if target_player and target_player.role_type == RoleType.WEREWOLF:
                    visible.append(d)

        return visible

    def add_last_words(self, player_id: int, words: str):
        self.last_words.append({
            "player_id": player_id,
            "words": words,
            "day": self.day_number,
        })

    def add_vote(self, voter_id: int, target_id: int):
        self.vote_record.append({
            "voter_id": voter_id,
            "target_id": target_id,
            "day": self.day_number,
        })

    def reset_vote_record(self):
        self.vote_record = []

    def next_speaker(self):
        if self.speaker_order:
            self.current_speaker = (self.current_speaker + 1) % len(self.speaker_order)

    def set_speaker_order(self, order: List[int]):
        self.speaker_order = order
        self.current_speaker = 0

    def clear_night_deaths(self):
        self.night_deaths = []

    def add_night_death(self, player_id: int):
        self.night_deaths.append(player_id)