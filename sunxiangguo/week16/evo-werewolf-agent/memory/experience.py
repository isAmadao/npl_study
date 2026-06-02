"""经验存储系统

按角色类型保存和加载过往游戏经验，用于跨游戏的经验共享。
增强版：支持策略标签提取、智能检索、经验上限控制。
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

EXPERIENCES_DIR = os.path.join(os.path.dirname(__file__), "experiences")
MAX_EXPERIENCES = 20  # 每个角色最多保留20条经验


def _ensure_dir() -> None:
    """确保经验存储目录存在"""
    os.makedirs(EXPERIENCES_DIR, exist_ok=True)


def _get_file_path(role_type: str) -> str:
    """获取某角色类型的经验文件路径"""
    return os.path.join(EXPERIENCES_DIR, f"{role_type}.json")


def load_experiences(role_type: str) -> List[Dict]:
    """加载某角色类型的过往经验

    Args:
        role_type: 角色类型，如 "werewolf", "seer" 等

    Returns:
        经验列表，按时间正序排列（最老的在前）
    """
    _ensure_dir()
    filepath = _get_file_path(role_type)
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_experience(role_type: str, experience: Dict) -> None:
    """保存一条新经验（带策略标签和上限控制）

    Args:
        role_type: 角色类型
        experience: 经验数据，包含 summary, strategies, mistakes, lessons 等字段
    """
    _ensure_dir()
    filepath = _get_file_path(role_type)
    experiences = load_experiences(role_type)

    # 增加策略标签和时间戳
    enriched = {
        **experience,
        "strategy_tags": _extract_strategy_tags(experience),
        "timestamp": datetime.now().isoformat(),
    }
    experiences.append(enriched)

    # 保留最近 MAX_EXPERIENCES 条经验（避免无限增长）
    if len(experiences) > MAX_EXPERIENCES:
        experiences = experiences[-MAX_EXPERIENCES:]

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(experiences, f, ensure_ascii=False, indent=2)


def _extract_strategy_tags(experience: Dict) -> List[str]:
    """从经验中提取策略标签

    Args:
        experience: 经验数据

    Returns:
        策略标签列表
    """
    tags = []

    # 胜负标签
    if experience.get("is_winner"):
        tags.append("winning_strategy")
    else:
        tags.append("failure_case")

    role = experience.get("role_type", "")
    strategies = experience.get("strategies", "")
    summary = experience.get("summary", "")
    text = strategies + summary

    if role == "werewolf":
        if "悍跳" in text:
            tags.append("jump_seer")
        if "倒钩" in text:
            tags.append("reverse_hook")
        if "隐身" in text or "低调" in text:
            tags.append("stealth")
        if "带节奏" in text or "煽动" in text:
            tags.append("manipulation")
    elif role == "seer":
        if "跳" in text and "验" in text:
            tags.append("reveal_early")
        if "暗查" in text or "隐藏" in text:
            tags.append("hide_identity")
    elif role == "witch":
        if "首夜" in text and "救" in text:
            tags.append("save_first_night")
        if "毒" in text:
            tags.append("use_poison")
    elif role == "hunter":
        if "带" in text:
            tags.append("smart_shot")
    elif role == "villager":
        if "站边" in text:
            tags.append("side_picking")
        if "带队" in text:
            tags.append("leadership")

    return tags


def get_experience_prompt(role_type: str, situation: str = "") -> str:
    """将过往经验格式化为提示词文本，供 PlayerAgent 构建指令时使用

    支持智能检索：优先选择与当前局势相关的经验。

    Args:
        role_type: 角色类型
        situation: 当前局势描述（可选），用于智能匹配相关经验

    Returns:
        格式化的经验文本，若没有经验则返回空字符串
    """
    experiences = load_experiences(role_type)
    if not experiences:
        return ""

    # 智能检索：优先选择与当前局势相关的经验
    relevant = []
    if situation:
        relevant = _find_relevant_experiences(experiences, situation)

    # 兜底：取最近3条
    if not relevant:
        relevant = experiences[-3:]

    lines = ["", "## 你的过往游戏经验"]
    lines.append("以下是你以往扮演此角色时学到的经验教训，请参考：")

    for i, exp in enumerate(relevant, 1):
        outcome = "胜利" if exp.get("is_winner", False) else "失败"
        lines.append(f"\n--- 第{i}次经验（{outcome}）---")
        if exp.get("summary"):
            lines.append(f"  总结：{exp['summary']}")
        if exp.get("strategies"):
            lines.append(f"  策略：{exp['strategies']}")
        if exp.get("mistakes"):
            lines.append(f"  教训：{exp['mistakes']}")
        if exp.get("lessons"):
            lines.append(f"  建议：{exp['lessons']}")

    return "\n".join(lines)


def _find_relevant_experiences(
    experiences: List[Dict], situation: str
) -> List[Dict]:
    """根据当前局势智能检索相关经验

    Args:
        experiences: 全部经验列表
        situation: 当前局势描述

    Returns:
        相关经验列表（最多3条）
    """
    # 简单的关键词匹配
    scored = []
    for exp in experiences:
        score = 0
        text = (
            exp.get("summary", "")
            + exp.get("strategies", "")
            + exp.get("mistakes", "")
            + exp.get("lessons", "")
        )
        # 提取局势中的关键词进行匹配
        keywords = [
            "预言家", "狼人", "女巫", "猎人", "村民",
            "悍跳", "投票", "发言", "首夜", "白天",
            "胜利", "失败",
        ]
        for kw in keywords:
            if kw in situation and kw in text:
                score += 1
        if score > 0:
            scored.append((score, exp))

    # 按相关度排序，取前3条
    scored.sort(key=lambda x: x[0], reverse=True)
    return [exp for _, exp in scored[:3]]


def get_all_role_experiences() -> Dict[str, List[Dict]]:
    """获取所有角色的经验（供进化分析使用）

    Returns:
        {role_type: [experiences]}
    """
    result = {}
    for role in ["werewolf", "seer", "witch", "hunter", "villager"]:
        exps = load_experiences(role)
        if exps:
            result[role] = exps
    return result


def clear_experiences(role_type: Optional[str] = None):
    """清空经验

    Args:
        role_type: 指定角色清空，None则清空所有角色
    """
    if role_type:
        filepath = _get_file_path(role_type)
        if os.path.exists(filepath):
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False)
    else:
        for role in ["werewolf", "seer", "witch", "hunter", "villager"]:
            clear_experiences(role)


__all__ = [
    "load_experiences",
    "save_experience",
    "get_experience_prompt",
    "get_all_role_experiences",
    "clear_experiences",
]
