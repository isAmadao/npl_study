"""狼人杀游戏控制台演示

展示完整的一局狼人杀游戏流程
"""

import asyncio
import os
from typing import Dict, Any

from config.game_config import get_role_config, shuffle_roles, ROLE_CONFIGS
from engine.game_engine import GameEngine
from agent.player_agent import create_player_agent
from memory.game_logger import GameLogger


async def create_player_agents(engine: GameEngine, player_styles: Dict[str, str] = None):
    """为每个玩家创建AI代理"""
    if player_styles is None:
        player_styles = {}

    for player in engine.game_state.players:
        agent = create_player_agent(
            player_id=player.player_id,
            role_name=player.role.name,
            private_context=player.role.get_private_context(),
            camp=player.role.camp.value,
            decision_style=player_styles.get(str(player.player_id), "balanced"),
            role_type=player.role.role_type.value,
        )
        engine.set_player_agent(player.player_id, agent)


async def run_demo():
    """运行演示"""
    print("=" * 60)
    print("狼人杀 AI 对战演示")
    print("=" * 60)

    # 检查环境变量
    if not os.environ.get("OPENAI_API_KEY"):
        print("警告: 未设置 OPENAI_API_KEY 环境变量，AI 决策可能无法正常工作")
        print("请设置: export OPENAI_API_KEY='your-api-key'")

    # 配置游戏
    config_name = "standard_6"
    role_assignment = get_role_config(config_name)
    role_assignment = shuffle_roles(role_assignment)

    player_names = [f"玩家{i+1}" for i in range(len(role_assignment))]
    print(f"\n配置: {ROLE_CONFIGS[config_name]['name']}")
    print(f"角色分配: {role_assignment}")

    # 创建游戏引擎
    logger = GameLogger()
    engine = GameEngine(player_names, logger=logger)
    await engine.initialize(role_assignment)

    # 创建玩家代理
    print("\n正在创建 AI 玩家代理...")
    await create_player_agents(engine)
    print("AI 玩家代理创建完成")

    # 运行完整游戏
    print("\n" + "=" * 60)
    print("游戏开始！")
    print("=" * 60)

    await engine.start()

    # 打印结果
    print("\n" + "=" * 60)
    print("游戏结束！")
    print(f"胜利方: {'善良阵营' if engine.game_state.get_winner() == 'good' else '邪恶阵营'}")
    print("=" * 60)

    # 打印死亡记录
    print("\n死亡记录:")
    for death in engine.death_records:
        print(f"  第{death['day']}天: {death['player_name']}({death['role']}) - {death['cause']}")

    # 打印对话摘要
    print("\n对话记录摘要:")
    for dialogue in engine.game_state.dialogues[:20]:
        phase = dialogue.get("phase", "")
        action = dialogue.get("action", "")
        if action == "speech":
            print(f"  [{phase}] {dialogue.get('player_name')}: {dialogue.get('content', '')[:50]}...")
        elif action in ("vote", "night_kill"):
            print(f"  [{phase}] {dialogue.get('player_name')} -> 玩家{dialogue.get('target')}")


async def run_step_demo():
    """逐步运行演示（用于调试）"""
    print("=" * 60)
    print("狼人杀 AI 对战演示（逐步模式）")
    print("=" * 60)

    if not os.environ.get("OPENAI_API_KEY"):
        print("警告: 未设置 OPENAI_API_KEY 环境变量")

    config_name = "standard_6"
    role_assignment = get_role_config(config_name)
    role_assignment = shuffle_roles(role_assignment)

    player_names = [f"玩家{i+1}" for i in range(len(role_assignment))]
    print(f"\n配置: {ROLE_CONFIGS[config_name]['name']}")
    print(f"角色分配: {role_assignment}")

    engine = GameEngine(player_names)
    await engine.initialize(role_assignment)

    await create_player_agents(engine)

    print("\n游戏初始化完成，开始逐步执行...")

    step = 0
    while True:
        result = await engine.step()
        print(f"\n--- Step {step}: {result['phase']} ---")
        print(f"  Day: {result['day_number']}")
        print(f"  Players: {[p['player_id'] for p in result['players'] if p['is_alive']]}")
        print(f"  Game Over: {result['is_game_over']}")

        if result['is_game_over']:
            print(f"\n胜利方: {'善良阵营' if result['winner'] == 'good' else '邪恶阵营'}")
            break

        step += 1
        if step > 20:
            print("超过最大步数，强制结束")
            break


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--step":
        asyncio.run(run_step_demo())
    else:
        asyncio.run(run_demo())