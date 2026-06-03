"""狼人角色实现。

阵营：邪恶阵营
胜利条件：杀死所有神职 OR 杀死所有村民
夜间行动：与其他狼人协调，每晚杀死一名玩家
"""

from typing import Optional, Dict, Any
from roles.base import BaseRole, RoleType, Camp, NightAction


class Werewolf(BaseRole):
    """Werewolf role - belongs to evil camp.

    Win Condition: Kill all gods OR kill all villagers.
    Night Action: Coordinate with other wolves to kill one player per night.
    """

    @property
    def role_type(self) -> RoleType:
        return RoleType.WEREWOLF

    @property
    def camp(self) -> Camp:
        return Camp.EVIL

    def is_night_actionable(self) -> bool:
        return self._is_alive

    def get_night_action(self, game_state: Dict[str, Any]) -> Optional[NightAction]:
        other_wolves = [
            p for p in game_state.get("players", [])
            if p.get("role") == RoleType.WEREWOLF.value
            and p.get("player_id") != self.player_id
            and p.get("is_alive", False)
        ]
        return NightAction(
            action_type="kill",
            target=None,
            metadata={"allies": [w["player_id"] for w in other_wolves]}
        )

    def get_private_context(self) -> Dict[str, Any]:
        ctx = super().get_private_context()
        ctx["knows_wolves"] = True
        ctx["note"] = "你是狼人。你知道队友是谁，但不知道其他人的身份。夜间和队友协调杀人。白天要伪装自己，不要暴露身份。"
        return ctx

    def check_win(self, game_state: Dict[str, Any]) -> bool:
        players = game_state.get("players", [])
        alive_gods = [p for p in players if p.get("is_god") and p.get("is_alive")]
        alive_villagers = [p for p in players if not p.get("is_god") and p.get("is_alive")]
        alive_wolves = [p for p in players if p.get("role") == RoleType.WEREWOLF.value and p.get("is_alive")]

        if not alive_wolves:
            return False
        return len(alive_gods) == 0 or len(alive_villagers) == 0