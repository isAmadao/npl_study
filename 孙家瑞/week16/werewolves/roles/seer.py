"""预言家角色实现。

阵营：善良阵营
胜利条件：消灭所有狼人
夜间行动：每晚查验一名玩家，返回"好人"或"狼人"
"""

from typing import Optional, Dict, Any
from roles.base import BaseRole, RoleType, Camp, NightAction


class Seer(BaseRole):
    """Seer role - belongs to good camp.

    Win Condition: Eliminate all wolves.
    Night Action: Check one player per night, returns "good" or "wolf" only.
    """

    @property
    def role_type(self) -> RoleType:
        return RoleType.SEER

    @property
    def camp(self) -> Camp:
        return Camp.GOOD

    def is_night_actionable(self) -> bool:
        return self._is_alive

    def get_night_action(self, game_state: Dict[str, Any]) -> Optional[NightAction]:
        return NightAction(
            action_type="check",
            target=None,
            metadata={}
        )

    def get_private_context(self) -> Dict[str, Any]:
        ctx = super().get_private_context()
        ctx["note"] = "你是预言家。每晚可以查验一名玩家的身份，结果只有'好人'或'狼人'。要选择合适的时机跳身份或隐忍。"
        return ctx

    def check_win(self, game_state: Dict[str, Any]) -> bool:
        players = game_state.get("players", [])
        alive_wolves = [
            p for p in players
            if p.get("role") == RoleType.WEREWOLF.value and p.get("is_alive")
        ]
        return len(alive_wolves) == 0