"""玩家数据结构"""

from typing import Optional, Dict, Any
from dataclasses import dataclass

from roles.base import BaseRole, RoleType, Camp


@dataclass
class Player:
    """游戏中的玩家"""
    player_id: int
    role: BaseRole
    name: str = ""

    @property
    def role_type(self) -> RoleType:
        return self.role.role_type

    @property
    def camp(self) -> Camp:
        return self.role.camp

    @property
    def is_alive(self) -> bool:
        return self.role.is_alive

    @property
    def is_sheriff(self) -> bool:
        return self.role.is_sheriff

    def kill(self, cause: str, game_state: Dict[str, Any]) -> Optional[list]:
        return self.role.on_death(cause, game_state)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_id": self.player_id,
            "name": self.name,
            "role": self.role.role_type.value,
            "camp": self.role.camp.value,
            "is_alive": self.role.is_alive,
            "is_sheriff": self.role.is_sheriff,
        }

    def __repr__(self):
        status = "alive" if self.is_alive else "dead"
        sheriff = " (sheriff)" if self.is_sheriff else ""
        return f"Player({self.player_id}, {self.role.name}, {status}{sheriff})"