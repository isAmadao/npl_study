"""结构化日志模块"""

from typing import List, Dict, Any, Optional
from datetime import datetime


class GameLogger:
    """游戏日志记录器

    记录游戏过程中的所有事件，用于全程可观测性分析
    """

    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.speeches: List[Dict[str, Any]] = []
        self.votes: List[Dict[str, Any]] = []
        self.deaths: List[Dict[str, Any]] = []
        self.night_actions: List[Dict[str, Any]] = []

    def log_event(self, message: str, level: str = "info"):
        self.events.append({
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
        })

    def log_speech_event(self, content: str):
        self.speeches.append({
            "timestamp": datetime.now().isoformat(),
            "content": content,
        })

    def log_vote_event(self, content: str):
        self.votes.append({
            "timestamp": datetime.now().isoformat(),
            "content": content,
        })

    def log_death_event(self, content: str):
        self.deaths.append({
            "timestamp": datetime.now().isoformat(),
            "content": content,
        })

    def log_night_action_event(self, content: str):
        self.night_actions.append({
            "timestamp": datetime.now().isoformat(),
            "content": content,
        })

    def log_action_event(self, content: str):
        self.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": "action",
            "content": content,
        })

    def get_full_log(self) -> Dict[str, Any]:
        return {
            "events": self.events,
            "speeches": self.speeches,
            "votes": self.votes,
            "deaths": self.deaths,
            "night_actions": self.night_actions,
        }

    def info(self, msg: str):
        self.log_event(msg, "info")

    def debug(self, msg: str):
        self.log_event(msg, "debug")

    def warning(self, msg: str):
        self.log_event(msg, "warning")

    def error(self, msg: str):
        self.log_event(msg, "error")