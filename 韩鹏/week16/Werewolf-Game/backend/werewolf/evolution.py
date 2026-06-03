"""AI狼人杀 - 自演化记忆系统 Phase 4"""
from __future__ import annotations

import json, os
from datetime import datetime
from typing import Optional
from werewolf.engine import Role, Team, GameState

MEMORY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "memory")

class EvolutionMemory:
    """对局经验记忆库 —— 存储每局关键决策，为下局提供经验"""

    def __init__(self):
        os.makedirs(MEMORY_DIR, exist_ok=True)
        self.memory_file = os.path.join(MEMORY_DIR, "experience.json")
        self.memories: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        return []

    def save(self):
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.memories[-100:], f, ensure_ascii=False, indent=2)

    def record_game(self, state: GameState):
        """记录一局游戏的关键信息"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "winner": state.winner.value if state.winner else "unknown",
            "rounds": state.round_num,
            "players": [],
            "key_events": [],
        }

        for p in state.players:
            summary["players"].append({
                "id": p.id, "name": p.name, "role": p.role.label,
                "alive": p.alive, "team": p.role.team.value,
            })

        # 提取关键事件
        for log in state.event_log:
            if log.get("type") in ("werewolf_kill", "seer_check", "witch_save",
                                     "witch_poison", "vote_eliminate", "game_over"):
                summary["key_events"].append(log)

        self.memories.append(summary)
        self.save()

    def get_experience_for_role(self, role: Role, max_entries: int = 5) -> str:
        """获取某角色的历史经验摘要（用于注入下一局的Prompt）"""
        relevant = [m for m in self.memories if m.get("winner")]
        if not relevant:
            return ""

        lines = ["\n[历史经验] 以下是你之前作为该角色的对局总结："]
        count = 0
        for mem in reversed(relevant):
            for p in mem.get("players", []):
                if p.get("role") == role.label:
                    outcome = "胜利" if (p["team"] == mem["winner"]) else "失败"
                    lines.append(f"- 对局结果：{outcome}（{mem['winner']}获胜），持续{mem['rounds']}轮。")
                    count += 1
                    break
            if count >= max_entries:
                break

        if count == 0:
            return ""
        lines.append("[经验结束]\n")
        return "\n".join(lines)


# 全局单例
_global_memory: Optional[EvolutionMemory] = None

def get_memory() -> EvolutionMemory:
    global _global_memory
    if _global_memory is None:
        _global_memory = EvolutionMemory()
    return _global_memory
