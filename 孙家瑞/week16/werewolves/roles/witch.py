"""女巫角色实现。

阵营：善良阵营
胜利条件：消灭所有狼人
夜间行动：一瓶解药（救人）、一瓶毒药（杀人），每种药一局只能用一次，不能自救
"""

from typing import Optional, Dict, Any
from roles.base import BaseRole, RoleType, Camp, NightAction


class Witch(BaseRole):
    """Witch role - belongs to good camp.

    Win Condition: Eliminate all wolves.
    Night Action: One heal (resurrect tonight's death) + one poison (kill any player).
                  Each usable once per game, cannot save self.
    """

    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.has_heal = True
        self.has_poison = True
        self.used_heal = False
        self.used_poison = False

    @property
    def role_type(self) -> RoleType:
        return RoleType.WITCH

    @property
    def camp(self) -> Camp:
        return Camp.GOOD

    def is_night_actionable(self) -> bool:
        return self._is_alive and (self.has_heal or self.has_poison)

    def use_heal(self):
        self.has_heal = False
        self.used_heal = True

    def use_poison(self):
        self.has_poison = False
        self.used_poison = True

    def get_night_action(self, game_state: Dict[str, Any]) -> Optional[NightAction]:
        return NightAction(
            action_type="heal_or_poison",
            target=None,
            metadata={"has_heal": self.has_heal, "has_poison": self.has_poison}
        )

    def get_private_context(self) -> Dict[str, Any]:
        ctx = super().get_private_context()
        ctx["note"] = f"你是女巫。你有一瓶解药（{'可用' if self.has_heal else '已用'}）和一瓶毒药（{'可用' if self.has_poison else '已用'}）。解药可以救今晚被杀的人，但不能自救。毒药可以毒死任何人。"
        return ctx

    def check_win(self, game_state: Dict[str, Any]) -> bool:
        players = game_state.get("players", [])
        alive_wolves = [
            p for p in players
            if p.get("role") == RoleType.WEREWOLF.value and p.get("is_alive")
        ]
        return len(alive_wolves) == 0