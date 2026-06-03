"""经验存储模块

负责保存和加载对局经验数据，用于自进化Agent
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime


DATA_DIR = Path(__file__).parent.parent / "data"
EXPERIENCE_FILE = DATA_DIR / "experience.json"


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_experiences() -> List[Dict[str, Any]]:
    """加载所有经验数据"""
    if not EXPERIENCE_FILE.exists():
        return []
    try:
        with open(EXPERIENCE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_experiences(experiences: List[Dict[str, Any]]):
    """保存经验数据"""
    ensure_data_dir()
    with open(EXPERIENCE_FILE, "w", encoding="utf-8") as f:
        json.dump(experiences, f, ensure_ascii=False, indent=2)


def save_experience(role: str, experience: Dict[str, Any]):
    """保存单条经验"""
    experiences = load_experiences()
    experiences.append({
        **experience,
        "role": role,
        "saved_at": datetime.now().isoformat(),
    })
    save_experiences(experiences)


def get_experiences_by_role(role: str) -> List[Dict[str, Any]]:
    """获取指定角色的所有经验"""
    experiences = load_experiences()
    return [e for e in experiences if e.get("role") == role]


def get_role_statistics(role: str) -> Dict[str, Any]:
    """获取指定角色的统计数据"""
    experiences = get_experiences_by_role(role)
    if not experiences:
        return {
            "total_games": 0,
            "wins": 0,
            "win_rate": 0.0,
        }

    total = len(experiences)
    wins = sum(1 for e in experiences if e.get("is_winner", False))

    return {
        "total_games": total,
        "wins": wins,
        "win_rate": wins / total if total > 0 else 0.0,
    }


def get_experience_prompt(role: str, max_experiences: int = 5) -> str:
    """生成经验提示文本"""
    experiences = get_experiences_by_role(role)
    if not experiences:
        return ""

    recent = experiences[-max_experiences:]
    lines = ["\n\n## 历史经验参考"]
    for exp in recent:
        lines.append(f"- 角色：{exp.get('role', '')}，胜率：{exp.get('win_rate', 0):.1%}")
        if exp.get("decision_history"):
            for dh in exp["decision_history"][-3:]:
                lines.append(f"  - {dh}")

    return "\n".join(lines)