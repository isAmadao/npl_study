"""AI狼人杀 - 终端入口 Phase 1"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from werewolf.engine import WerewolfGame, print_roles

def main():
    print("=" * 40)
    print("  AI wolf kill Phase 1")
    print("=" * 40)
    game = WerewolfGame(seed=42)
    print_roles(game)
    print("--- Game Start ---")
    game.run(verbose=True)
    print("--- Game Log ---")
    print(game.export_log())

if __name__ == "__main__":
    main()
