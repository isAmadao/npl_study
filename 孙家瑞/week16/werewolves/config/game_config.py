"""游戏配置模块。"""

from typing import Dict

# 标准6人局角色配置
STANDARD_6P_ROLES = {
    0: "werewolf",
    1: "werewolf",
    2: "seer",
    3: "witch",
    4: "hunter",
    5: "villager",
}

ROLE_CONFIGS = {
    "standard_6": {
        "name": "标准6人局",
        "description": "2狼、1预言家、1女巫、1猎人、1村民",
        "roles": STANDARD_6P_ROLES.copy(),
    },
    "simple_4": {
        "name": "简易4人局",
        "description": "1狼、1预言家、1女巫、1村民",
        "roles": {
            0: "werewolf",
            1: "seer",
            2: "witch",
            3: "villager",
        }
    },
}


def get_role_config(config_name: str) -> Dict[int, str]:
    """获取角色配置"""
    if config_name in ROLE_CONFIGS:
        return ROLE_CONFIGS[config_name]["roles"].copy()
    else:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(ROLE_CONFIGS.keys())}")


def shuffle_roles(role_assignment: Dict[int, str]) -> Dict[int, str]:
    """随机打乱角色分配"""
    import random
    roles = list(role_assignment.values())
    random.shuffle(roles)
    return {i: r for i, r in enumerate(roles)}