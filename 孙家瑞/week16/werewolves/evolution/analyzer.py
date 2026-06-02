"""对局分析器

分析对局结果，识别关键决策点，计算各角色表现分数
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class GameAnalysis:
    """对局分析结果"""
    game_id: str
    winner: str
    role_performance: Dict[str, RolePerformance]
    key_decisions: List[KeyDecision]
    optimization_suggestions: List[str]


@dataclass
class RolePerformance:
    """角色表现"""
    role: str
    player_id: int
    survival_days: int
    correct_kills: int
    correct_checks: int
    correct_votes: int
    speeches_quality: float
    overall_score: float


@dataclass
class KeyDecision:
    """关键决策"""
    day: int
    phase: str
    player_id: int
    role: str
    decision_type: str
    target: Optional[int]
    reasoning: str
    outcome: str
    impact_score: float


class GameAnalyzer:
    """对局分析器"""

    def __init__(self, game_record: Dict[str, Any]):
        self.game_record = game_record
        self.analysis: Optional[GameAnalysis] = None

    def analyze(self) -> GameAnalysis:
        """执行完整分析"""
        game_id = self.game_record.get("game_id", "")
        winner = self.game_record.get("winner", "")

        role_performance = self._calculate_role_performance()
        key_decisions = self._identify_key_decisions()
        suggestions = self._generate_optimization_suggestions()

        self.analysis = GameAnalysis(
            game_id=game_id,
            winner=winner,
            role_performance=role_performance,
            key_decisions=key_decisions,
            optimization_suggestions=suggestions,
        )
        return self.analysis

    def _calculate_role_performance(self) -> Dict[str, RolePerformance]:
        """计算各角色表现分数"""
        performances = {}
        dialogues = self.game_record.get("dialogues", [])
        deaths = self.game_record.get("deaths", [])

        players = self.game_record.get("players", [])
        for player in players:
            role = player.get("role", "")
            pid = player.get("player_id", 0)

            survival_days = self._calculate_survival_days(pid, deaths)
            correct_kills = self._count_correct_kills(pid, dialogues, winner=self.game_record.get("winner"))
            correct_checks = self._count_correct_checks(pid, dialogues)
            correct_votes = self._count_correct_votes(pid, dialogues)
            speeches_quality = self._evaluate_speeches_quality(pid, dialogues)

            score = self._compute_overall_score(
                survival_days, correct_kills, correct_checks, correct_votes, speeches_quality
            )

            performances[role] = RolePerformance(
                role=role,
                player_id=pid,
                survival_days=survival_days,
                correct_kills=correct_kills,
                correct_checks=correct_checks,
                correct_votes=correct_votes,
                speeches_quality=speeches_quality,
                overall_score=score,
            )

        return performances

    def _calculate_survival_days(self, player_id: int, deaths: List[Dict]) -> int:
        for death in deaths:
            if death.get("player_id") == player_id:
                return death.get("day", 1) - 1
        return self.game_record.get("day_number", 1)

    def _count_correct_kills(self, player_id: int, dialogues: List[Dict], winner: str) -> int:
        return 0

    def _count_correct_checks(self, player_id: int, dialogues: List[Dict]) -> int:
        return 0

    def _count_correct_votes(self, player_id: int, dialogues: List[Dict]) -> int:
        return 0

    def _evaluate_speeches_quality(self, player_id: int, dialogues: List[Dict]) -> float:
        return 0.5

    def _compute_overall_score(self, survival: int, kills: int, checks: int, votes: int, speeches: float) -> float:
        return min(1.0, (survival * 0.1 + kills * 0.2 + checks * 0.2 + votes * 0.2 + speeches * 0.3) / 10)

    def _identify_key_decisions(self) -> List[KeyDecision]:
        return []

    def _generate_optimization_suggestions(self) -> List[str]:
        if not self.analysis:
            return []
        suggestions = []
        for role, perf in self.analysis.role_performance.items():
            if perf.overall_score < 0.5:
                suggestions.append(f"{role}: 需要优化决策策略，当前得分 {perf.overall_score:.2f}")
        return suggestions