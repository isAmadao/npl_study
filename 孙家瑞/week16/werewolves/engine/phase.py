"""游戏阶段定义"""

from enum import Enum


class GamePhase(Enum):
    """游戏阶段枚举"""
    NIGHT_WOLF = "狼人杀人"
    NIGHT_SEER = "预言家查验"
    NIGHT_WITCH = "女巫用药"
    NIGHT_HUNTER = "猎人开枪"
    DAY_START = "白天开始"
    ELECTION = "警长选举"
    SPEECH = "公开演讲"
    VOTE = "投票环节"
    GAME_OVER = "游戏结束"


class TurnOrder:
    """夜间行动顺序"""

    @staticmethod
    def get_night_order():
        return [
            GamePhase.NIGHT_WOLF,
            GamePhase.NIGHT_SEER,
            GamePhase.NIGHT_WITCH,
        ]

    @staticmethod
    def is_night_phase(phase: GamePhase) -> bool:
        return phase in [
            GamePhase.NIGHT_WOLF,
            GamePhase.NIGHT_SEER,
            GamePhase.NIGHT_WITCH,
            GamePhase.NIGHT_HUNTER,
        ]

    @staticmethod
    def is_day_phase(phase: GamePhase) -> bool:
        return phase in [
            GamePhase.DAY_START,
            GamePhase.ELECTION,
            GamePhase.SPEECH,
            GamePhase.VOTE,
        ]