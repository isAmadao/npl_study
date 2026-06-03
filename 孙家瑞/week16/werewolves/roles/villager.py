"""村民角色实现。

阵营：善良阵营
胜利条件：消灭所有狼人
无夜间技能：黑夜全程闭眼，白天纯靠发言找狼和投票
"""

from typing import Dict, Any
from roles.base import BaseRole, RoleType, Camp


class Villager(BaseRole):
    """Villager role - belongs to good camp.

    Win Condition: Eliminate all wolves.
    No night ability: must rely on day speech and reasoning.
    """

    @property
    def role_type(self) -> RoleType:
        return RoleType.VILLAGER

    @property
    def camp(self) -> Camp:
        return Camp.GOOD

    def get_private_context(self) -> Dict[str, Any]:
        ctx = super().get_private_context()
        ctx["note"] = "你是村民。你没有特殊技能，白天通过发言和投票找出狼人。分析其他玩家的发言逻辑，判断谁可能是狼人。"
        return ctx

    def check_win(self, game_state: Dict[str, Any]) -> bool:
        players = game_state.get("players", [])
        alive_wolves = [
            p for p in players
            if p.get("role") == RoleType.WEREWOLF.value and p.get("is_alive")
        ]
        return len(alive_wolves) == 0