"""
AI狼人杀 - 引擎单元测试
Phase 1: 测试角色分配、行动合法性、投票计数、胜负判定、完整对局
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from werewolf.engine import (
    Role, Team, ActionType, Action, Phase, Player, GameState,
    RuleEngine, RuleAgent, WerewolfGame, create_default_game
)


class TestRoles:
    """角色定义测试"""

    def test_role_count(self):
        assert len(Role) == 4

    def test_role_teams(self):
        assert Role.WEREWOLF.team == Team.WEREWOLF
        assert Role.SEER.team == Team.VILLAGER
        assert Role.WITCH.team == Team.VILLAGER
        assert Role.VILLAGER.team == Team.VILLAGER

    def test_role_labels(self):
        assert Role.WEREWOLF.label == "狼人"
        assert Role.SEER.label == "预言家"
        assert Role.WITCH.label == "女巫"
        assert Role.VILLAGER.label == "村民"


class TestGameSetup:
    """游戏初始化测试"""

    def test_create_game(self):
        game = WerewolfGame(seed=42)
        assert len(game.state.players) == 6
        roles = [p.role for p in game.state.players]
        assert roles.count(Role.WEREWOLF) == 2
        assert roles.count(Role.SEER) == 1
        assert roles.count(Role.WITCH) == 1
        assert roles.count(Role.VILLAGER) == 2

    def test_phase_starts_correct(self):
        game = WerewolfGame(seed=42)
        assert game.state.phase == Phase.NIGHT_WEREWOLF
        assert game.state.round_num == 1

    def test_all_players_alive_initially(self):
        game = WerewolfGame(seed=42)
        assert all(p.alive for p in game.state.players)


class TestRuleEngine:
    """规则引擎测试"""

    def test_check_win_werewolves_win(self):
        """狼人数>=村民数 → 狼人胜"""
        state = GameState(players=[
            Player(0, "A", Role.WEREWOLF, True),
            Player(1, "B", Role.WEREWOLF, True),
            Player(2, "C", Role.VILLAGER, True),
            Player(3, "D", Role.VILLAGER, False),
            Player(4, "E", Role.SEER, False),
            Player(5, "F", Role.WITCH, False),
        ])
        assert RuleEngine.check_win(state) == Team.WEREWOLF

    def test_check_win_villagers_win(self):
        """狼人全灭 → 村民胜"""
        state = GameState(players=[
            Player(0, "A", Role.WEREWOLF, False),
            Player(1, "B", Role.WEREWOLF, False),
            Player(2, "C", Role.VILLAGER, True),
            Player(3, "D", Role.VILLAGER, True),
            Player(4, "E", Role.SEER, True),
            Player(5, "F", Role.WITCH, True),
        ])
        assert RuleEngine.check_win(state) == Team.VILLAGER

    def test_check_win_no_winner(self):
        """游戏继续"""
        state = GameState(players=[
            Player(0, "A", Role.WEREWOLF, True),
            Player(1, "B", Role.WEREWOLF, True),
            Player(2, "C", Role.VILLAGER, True),
            Player(3, "D", Role.VILLAGER, True),
            Player(4, "E", Role.SEER, True),
            Player(5, "F", Role.WITCH, True),
        ])
        assert RuleEngine.check_win(state) is None

    def test_resolve_night_kill_only(self):
        """仅狼人杀人，女巫不行动"""
        state = GameState(players=[
            Player(i, f"P{i}", Role.VILLAGER, True) for i in range(6)
        ])
        state.players[0].role = Role.WEREWOLF
        state.players[1].role = Role.WEREWOLF
        state.werewolf_target = 2
        deaths = RuleEngine.resolve_night_actions(state)
        assert deaths == [2]

    def test_resolve_night_witch_saves(self):
        """女巫救人"""
        state = GameState(players=[
            Player(i, f"P{i}", Role.VILLAGER, True) for i in range(6)
        ])
        state.werewolf_target = 2
        state.witch_save_target = 2
        deaths = RuleEngine.resolve_night_actions(state)
        assert deaths == []  # 被救活

    def test_resolve_night_witch_poison(self):
        """女巫毒人"""
        state = GameState(players=[
            Player(i, f"P{i}", Role.VILLAGER, True) for i in range(6)
        ])
        state.witch_poison_target = 3
        deaths = RuleEngine.resolve_night_actions(state)
        assert deaths == [3]

    def test_resolve_vote_simple(self):
        """简单投票"""
        state = GameState(players=[
            Player(i, f"P{i}", Role.VILLAGER, True) for i in range(6)
        ])
        state.votes = {0: 1, 1: 1, 2: 1, 3: 0, 4: 0}
        eliminated = RuleEngine.resolve_vote(state)
        assert eliminated == 1  # P1得3票

    def test_resolve_vote_tie(self):
        """平票"""
        state = GameState(players=[
            Player(i, f"P{i}", Role.VILLAGER, True) for i in range(6)
        ])
        state.votes = {0: 1, 1: 0, 2: 1, 3: 0}
        eliminated = RuleEngine.resolve_vote(state)
        assert eliminated is None  # 平票


class TestRuleAgent:
    """规则AI测试"""

    def setup_method(self):
        self.state = GameState(players=[
            Player(0, "狼A", Role.WEREWOLF, True),
            Player(1, "狼B", Role.WEREWOLF, True),
            Player(2, "预言家", Role.SEER, True),
            Player(3, "女巫", Role.WITCH, True),
            Player(4, "村民C", Role.VILLAGER, True),
            Player(5, "村民D", Role.VILLAGER, True),
        ])

    def test_werewolf_targets_non_wolf(self):
        """狼人选杀目标不能是狼同伴"""
        state = self.state
        state.phase = Phase.NIGHT_WEREWOLF
        werewolf = state.players[0]
        # 多次测试确保行为正确
        for _ in range(10):
            action = RuleAgent.get_action(state, werewolf)
            if action.target_id is not None:
                target = state.get_player(action.target_id)
                assert target.role != Role.WEREWOLF

    def test_seer_checks_other_player(self):
        """预言家查验其他玩家"""
        state = self.state
        state.phase = Phase.NIGHT_SEER
        seer = state.players[2]
        action = RuleAgent.get_action(state, seer)
        assert action.action_type == ActionType.CHECK
        assert action.target_id != seer.id

    def test_vote_returns_valid_target(self):
        """投票目标有效"""
        state = self.state
        state.phase = Phase.DAY_VOTE
        for player in state.alive_players():
            action = RuleAgent.get_action(state, player)
            if action.target_id is not None:
                assert action.target_id != player.id
                assert action.target_id in [p.id for p in state.alive_players()]


class TestFullGame:
    """完整对局集成测试"""

    def test_game_completes(self):
        """游戏能正常结束"""
        game = WerewolfGame(seed=42)
        game.run(verbose=False)
        assert game.state.phase == Phase.GAME_OVER
        assert game.state.winner is not None

    def test_game_has_logs(self):
        """游戏产生日志"""
        game = WerewolfGame(seed=42)
        game.run(verbose=False)
        assert len(game.state.event_log) > 0

    def test_game_winner_valid(self):
        """胜负结果合法"""
        game = WerewolfGame(seed=42)
        game.run(verbose=False)
        assert game.state.winner in (Team.WEREWOLF, Team.VILLAGER)

    def test_multiple_games_different(self):
        """不同种子产生不同结果"""
        game1 = WerewolfGame(seed=1)
        game1.run(verbose=False)
        game2 = WerewolfGame(seed=999)
        game2.run(verbose=False)
        # 至少角色分配不同
        roles1 = [p.role.value[0] for p in game1.state.players]
        roles2 = [p.role.value[0] for p in game2.state.players]
        assert roles1 != roles2  # 大概率不同


if __name__ == "__main__":
    # 手动运行测试
    import traceback
    tests = [
        TestRoles(), TestGameSetup(), TestRuleEngine(),
        TestRuleAgent(), TestFullGame()
    ]
    passed = 0
    failed = 0
    for obj in tests:
        cls_name = obj.__class__.__name__
        for name in dir(obj):
            if name.startswith("test_"):
                try:
                    # Call setup if exists
                    if hasattr(obj, "setup_method"):
                        obj.setup_method()
                    getattr(obj, name)()
                    print(f"  ✓ {cls_name}.{name}")
                    passed += 1
                except Exception as e:
                    print(f"  ✗ {cls_name}.{name}: {e}")
                    failed += 1
    print(f"\n通过: {passed}, 失败: {failed}")
