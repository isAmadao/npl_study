"""游戏角色基类。

所有角色都继承自此类，包含：
- 角色类型和阵营归属
- 夜间行动能力（如有）
- 白天发言能力
- 胜利条件检查
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


class RoleType(Enum):
    """All available role types in the game."""
    WEREWOLF = "werewolf"
    SEER = "seer"
    WITCH = "witch"
    HUNTER = "hunter"
    VILLAGER = "villager"


class Camp(Enum):
    """Team camps in the game."""
    GOOD = "good"
    EVIL = "evil"


@dataclass
class NightAction:
    """Represents a night action taken by a role."""
    action_type: str
    target: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VoteAction:
    """Represents a vote action during day."""
    target: int


class BaseRole:
    """Base class for all roles in the game."""

    def __init__(self, player_id: int):
        self.player_id = player_id
        self._is_alive = True
        self._is_sheriff = False

    @property
    def role_type(self) -> RoleType:
        raise NotImplementedError

    @property
    def camp(self) -> Camp:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.role_type.value.capitalize()

    @property
    def is_alive(self) -> bool:
        return self._is_alive

    @is_alive.setter
    def is_alive(self, value: bool):
        self._is_alive = value

    @property
    def is_sheriff(self) -> bool:
        return self._is_sheriff

    @is_sheriff.setter
    def is_sheriff(self, value: bool):
        self._is_sheriff = value

    def is_night_actionable(self) -> bool:
        return False

    def get_night_action(self, game_state: Dict[str, Any]) -> Optional[NightAction]:
        return None

    def can_speak(self) -> bool:
        return self._is_alive

    def can_vote(self) -> bool:
        return self._is_alive

    def on_death(self, cause: str, game_state: Dict[str, Any]) -> Optional[List[int]]:
        self._is_alive = False
        return None

    def check_win(self, game_state: Dict[str, Any]) -> bool:
        raise NotImplementedError

    def get_private_context(self) -> Dict[str, Any]:
        return {
            "role": self.name,
            "player_id": self.player_id,
            "camp": self.camp.value,
        }

    def __repr__(self) -> str:
        status = "alive" if self._is_alive else "dead"
        sheriff = " (sheriff)" if self._is_sheriff else ""
        return f"{self.__class__.__name__}(player_id={self.player_id}, {status}{sheriff})"