"""AI狼人杀 - LLM Agent 入口 Phase 2"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from werewolf.engine import WerewolfGame, print_roles
from werewolf.agent import LLMAgent

def main():
    print("=" * 40)
    print("  AI wolf kill Phase 2 - LLM Agent")
    print("=" * 40)

    agent = LLMAgent()

    def get_action(state, player):
        return agent.get_action(state, player)

    game = WerewolfGame(seed=42, agent=get_action)
    print_roles(game)
    print("\n--- Game Start (LLM) ---\n")
    game.run(verbose=True)
    print("\n--- Game Log ---")
    print(game.export_log())

if __name__ == "__main__":
    main()
