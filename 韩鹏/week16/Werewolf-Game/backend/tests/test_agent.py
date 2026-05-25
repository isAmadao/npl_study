"""AI狼人杀 - Agent 单元测试 Phase 2"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from werewolf.engine import (
    Role, Team, ActionType, Action, Phase, Player, GameState,
    WerewolfGame, RuleAgent
)
from werewolf.agent import InfoFilter, LLMAgent


class TestInfoFilter:
    """信息过滤器测试"""

    def setup_method(self):
        self.state = GameState(players=[
            Player(0, "Alice", Role.WEREWOLF, True),
            Player(1, "Bob", Role.WEREWOLF, True),
            Player(2, "Carol", Role.SEER, True),
            Player(3, "Dave", Role.WITCH, True),
            Player(4, "Eve", Role.VILLAGER, True),
            Player(5, "Frank", Role.VILLAGER, True),
        ])
        self.state.round_num = 2
        self.state.phase = Phase.DAY_DISCUSS

    def test_visible_basic_info(self):
        info = InfoFilter.visible_state(self.state, self.state.players[0])
        assert info["player_name"] == "Alice"
        assert info["player_role"] == "狼人"
        assert info["round"] == 2
        assert len(info["alive_players"]) == 6

    def test_werewolf_sees_allies(self):
        info = InfoFilter.visible_state(self.state, self.state.players[0])
        assert "werewolf_allies" in info
        assert "Bob" in info["werewolf_allies"]

    def test_villager_has_no_special_info(self):
        info = InfoFilter.visible_state(self.state, self.state.players[4])
        assert "werewolf_allies" not in info
        assert "checked_players" not in info

    def test_witch_sees_potions(self):
        info = InfoFilter.visible_state(self.state, self.state.players[3])
        assert info["has_antidote"] is True
        assert info["has_poison"] is True

    def test_dead_player_marked(self):
        self.state.players[5].alive = False
        info = InfoFilter.visible_state(self.state, self.state.players[5])
        assert info["alive"] is False


class TestLLMAgentParsing:
    """LLM响应解析测试"""

    def setup_method(self):
        self.agent = LLMAgent()

    def test_parse_kill_action(self):
        resp = '{"reasoning": "kill the seer", "action": "kill", "target_id": 2}'
        action = self.agent._parse_response(resp, 0)
        assert action is not None
        assert action.action_type == ActionType.KILL
        assert action.target_id == 2

    def test_parse_speak_action(self):
        resp = '{"reasoning": "i am good", "action": "speak", "speech": "Im a villager"}'
        action = self.agent._parse_response(resp, 1)
        assert action is not None
        assert action.action_type == ActionType.SPEAK
        assert action.content == "Im a villager"

    def test_parse_invalid_json_returns_none(self):
        resp = "not json at all"
        action = self.agent._parse_response(resp, 0)
        assert action is None

    def test_parse_empty_response(self):
        action = self.agent._parse_response("", 0)
        assert action is None


class TestFallbackToRuleAgent:
    """LLM失败时回退到规则AI"""

    def test_llm_agent_fallback(self):
        agent = LLMAgent()
        state = GameState(players=[
            Player(i, f"P{i}", Role.VILLAGER, True) for i in range(6)
        ])
        state.players[0].role = Role.WEREWOLF
        state.players[1].role = Role.WEREWOLF
        state.phase = Phase.NIGHT_WEREWOLF

        # get_action should fallback to RuleAgent when LLM unavailable
        action = agent.get_action(state, state.players[0])
        assert action is not None
        assert action.action_type in (ActionType.KILL, ActionType.SKIP)


if __name__ == "__main__":
    tests = [TestInfoFilter(), TestLLMAgentParsing(), TestFallbackToRuleAgent()]
    passed = 0
    failed = 0
    for obj in tests:
        cls_name = obj.__class__.__name__
        for name in dir(obj):
            if name.startswith("test_"):
                try:
                    if hasattr(obj, "setup_method"):
                        obj.setup_method()
                    getattr(obj, name)()
                    print(f"  OK {cls_name}.{name}")
                    passed += 1
                except Exception as e:
                    print(f"  FAIL {cls_name}.{name}: {e}")
                    failed += 1
    print(f"\nPass: {passed}, Fail: {failed}")
