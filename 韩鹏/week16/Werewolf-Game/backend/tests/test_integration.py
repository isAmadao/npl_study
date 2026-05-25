"""AI狼人杀 - 集成测试 Phase 4"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from werewolf.engine import WerewolfGame, Phase, Team, RuleAgent
from werewolf.agent import LLMAgent, InfoFilter
from werewolf.evolution import EvolutionMemory


class TestEvolutionMemory:
    """自演化记忆系统测试"""

    def test_record_and_retrieve(self):
        mem = EvolutionMemory()
        game = WerewolfGame(seed=1)
        game.run(verbose=False)
        mem.record_game(game.state)
        assert len(mem.memories) >= 1
        exp = mem.get_experience_for_role(game.state.players[0].role)
        # Should have some experience text or empty string
        assert isinstance(exp, str)

    def test_multiple_games(self):
        mem = EvolutionMemory()
        for i in range(3):
            game = WerewolfGame(seed=i)
            game.run(verbose=False)
            mem.record_game(game.state)
        assert len(mem.memories) >= 3

    def test_llm_agent_with_memory(self):
        mem = EvolutionMemory()
        # Record a game first
        game = WerewolfGame(seed=1)
        game.run(verbose=False)
        mem.record_game(game.state)

        # Create agent with memory
        agent = LLMAgent(memory=mem)
        assert agent.memory is not None


class TestFullPipeline:
    """完整流水线测试"""

    def test_rule_ai_pipeline(self):
        """规则AI：端到端对局"""
        game = WerewolfGame(seed=42)
        game.run(verbose=False)
        assert game.state.phase == Phase.GAME_OVER
        assert game.state.winner is not None
        assert len(game.state.event_log) > 10

    def test_llm_fallback_pipeline(self):
        """LLM Agent（无API时回退到规则AI）：端到端对局"""
        agent = LLMAgent()
        def get_action(state, player):
            return agent.get_action(state, player)
        game = WerewolfGame(seed=42, agent=get_action)
        game.run(verbose=False)
        assert game.state.phase == Phase.GAME_OVER
        assert game.state.winner is not None

    def test_evolution_pipeline(self):
        """自演化：记录 + 注入 + 再次运行"""
        mem = EvolutionMemory()
        # Game 1
        g1 = WerewolfGame(seed=10)
        g1.run(verbose=False)
        mem.record_game(g1.state)

        # Game 2 with experience
        agent = LLMAgent(memory=mem)
        def get_action(state, player):
            return agent.get_action(state, player)
        g2 = WerewolfGame(seed=20, agent=get_action)
        g2.run(verbose=False)
        assert g2.state.phase == Phase.GAME_OVER

        # Record game 2
        mem.record_game(g2.state)
        assert len(mem.memories) >= 2


if __name__ == "__main__":
    tests = [TestEvolutionMemory(), TestFullPipeline()]
    passed = 0
    failed = 0
    for obj in tests:
        cls_name = obj.__class__.__name__
        for name in dir(obj):
            if name.startswith("test_"):
                try:
                    getattr(obj, name)()
                    print(f"  OK {cls_name}.{name}")
                    passed += 1
                except Exception as e:
                    print(f"  FAIL {cls_name}.{name}: {e}")
                    failed += 1
    print(f"\nPass: {passed}, Fail: {failed}")
