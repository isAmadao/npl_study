"""猎人角色实现。

阵营：善良阵营
胜利条件：消灭所有狼人
被动技能：死亡时可开枪带走一人，被女巫毒杀不能开枪
"""

from typing import Optional, List, Dict, Any
from roles.base import BaseRole, RoleType, Camp


class Hunter(BaseRole):
    """Hunter role - belongs to good camp.

    Win Condition: Eliminate all wolves.
    Passive: If killed (not poisoned), can shoot one player before dying.
    """

    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.can_shoot = True
        self._shot_target: Optional[int] = None

    @property
    def role_type(self) -> RoleType:
        return RoleType.HUNTER

    @property
    def camp(self) -> Camp:
        return Camp.GOOD

    def lock_shoot(self):
        self.can_shoot = False

    def on_death(self, cause: str, game_state: Dict[str, Any]) -> Optional[List[int]]:
        self._is_alive = False
        if cause == "poison":
            self.can_shoot = False
            return None
        if self.can_shoot and self.can_shoot:
            return [self._shot_target] if self._shot_target else []
        return None

    def get_private_context(self) -> Dict[str, Any]:
        ctx = super().get_private_context()
        ctx["note"] = "你是猎人。如果你死亡（不是被毒杀），你可以开枪带走一人。如果你被女巫毒杀，则不能开枪。"
        return ctx

    def check_win(self, game_state: Dict[str, Any]) -> bool:
        players = game_state.get("players", [])
        alive_wolves = [
            p for p in players
            if p.get("role") == RoleType.WEREWOLF.value and p.get("is_alive")
        ]
        return len(alive_wolves) == 0