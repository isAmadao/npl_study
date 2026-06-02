"""游戏评测体系

多维可量化评测：结果评测、过程评测、对抗评测。
支持复盘归因和 Leaderboard 排行。
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import json
import os
from datetime import datetime


@dataclass
class PlayerEvaluation:
    """单个玩家的评测结果"""
    player_id: int
    player_name: str
    role: str
    camp: str
    is_winner: bool

    # 结果评测 (0-100)
    survival_days: int = 0          # 存活天数
    contribution_score: float = 0   # 贡献分（击杀/查验/救人等）
    result_score: float = 0         # 结果综合分

    # 过程评测 (0-100)
    speech_quality: float = 0       # 发言质量
    vote_accuracy: float = 0        # 投票准确率
    info_usage: float = 0           # 信息利用效率
    process_score: float = 0        # 过程综合分

    # 对抗评测 (0-100)
    deception_score: float = 0      # 欺骗/伪装能力（狼人）
    detection_score: float = 0      # 识别能力（好人）
    bluff_success_rate: float = 0   # 悍跳/挡刀成功率
    adversarial_score: float = 0    # 对抗综合分

    # 综合得分
    total_score: float = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "role": self.role,
            "camp": self.camp,
            "is_winner": self.is_winner,
            "result_score": round(self.result_score, 1),
            "process_score": round(self.process_score, 1),
            "adversarial_score": round(self.adversarial_score, 1),
            "total_score": round(self.total_score, 1),
            "details": {
                "survival_days": self.survival_days,
                "contribution_score": round(self.contribution_score, 1),
                "speech_quality": round(self.speech_quality, 1),
                "vote_accuracy": round(self.vote_accuracy, 1),
                "info_usage": round(self.info_usage, 1),
                "deception_score": round(self.deception_score, 1),
                "detection_score": round(self.detection_score, 1),
                "bluff_success_rate": round(self.bluff_success_rate, 1),
            },
        }


@dataclass
class GameEvaluation:
    """单局游戏评测结果"""
    game_id: str
    winner: str                     # "good" / "evil"
    total_days: int
    total_deaths: int
    player_evaluations: List[PlayerEvaluation] = field(default_factory=list)

    # 全局指标
    game_quality_score: float = 0   # 对局质量分
    balance_score: float = 0        # 阵营平衡度

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "winner": self.winner,
            "total_days": self.total_days,
            "total_deaths": self.total_deaths,
            "game_quality_score": round(self.game_quality_score, 1),
            "balance_score": round(self.balance_score, 1),
            "players": [pe.to_dict() for pe in self.player_evaluations],
        }


class GameEvaluator:
    """游戏评测器

    基于游戏引擎的完整数据（死亡记录、对话记录、投票记录）
    进行多维度量化评测。
    """

    def evaluate(
        self,
        game_id: str,
        winner: str,
        players: list,
        death_records: list,
        dialogues: list,
        vote_records: list,
        last_words: list,
    ) -> GameEvaluation:
        """对一局游戏进行全面评测"""
        total_days = max((d.get("day", 1) for d in death_records), default=1)
        total_deaths = len(death_records)

        eval_game = GameEvaluation(
            game_id=game_id,
            winner=winner,
            total_days=total_days,
            total_deaths=total_deaths,
        )

        # 逐玩家评测
        for player in players:
            pid = player.get("player_id", 0)
            pe = self._evaluate_player(
                player=player,
                winner=winner,
                death_records=death_records,
                dialogues=dialogues,
                vote_records=vote_records,
                total_days=total_days,
            )
            eval_game.player_evaluations.append(pe)

        # 全局指标
        eval_game.game_quality_score = self._calc_game_quality(
            total_days, total_deaths, len(players), eval_game.player_evaluations
        )
        eval_game.balance_score = self._calc_balance(eval_game.player_evaluations)

        return eval_game

    def _evaluate_player(
        self,
        player: dict,
        winner: str,
        death_records: list,
        dialogues: list,
        vote_records: list,
        total_days: int,
    ) -> PlayerEvaluation:
        """评测单个玩家"""
        pid = player.get("player_id", 0)
        role = player.get("role", "unknown")
        camp = player.get("camp", "good")
        is_alive = player.get("is_alive", True)
        is_winner = (camp == "good" and winner == "good") or (camp == "evil" and winner == "evil")

        pe = PlayerEvaluation(
            player_id=pid,
            player_name=player.get("name", f"玩家{pid}"),
            role=role,
            camp=camp,
            is_winner=is_winner,
        )

        # ---- 结果评测 ----
        # 存活天数
        death_day = None
        for d in death_records:
            if d.get("player_id") == pid:
                death_day = d.get("day", total_days)
                break
        pe.survival_days = death_day if death_day else total_days

        # 贡献分：基于发言数、投票准确率、角色行为
        speeches = [d for d in dialogues if d.get("player_id") == pid and d.get("action") == "speech"]
        votes = [d for d in dialogues if d.get("player_id") == pid and d.get("action") == "vote"]
        night_actions = [d for d in dialogues if d.get("player_id") == pid and d.get("phase") == "夜间行动"]

        # 发言越多贡献越高（上限30）
        speech_count = len(speeches)
        pe.contribution_score = min(30, speech_count * 6)
        # 夜间行动贡献（上限20）
        pe.contribution_score += min(20, len(night_actions) * 10)

        # 胜利加分
        if is_winner:
            pe.contribution_score += 20

        # 存活加分
        pe.contribution_score += min(30, pe.survival_days * 5)
        pe.contribution_score = min(100, pe.contribution_score)

        # 结果综合分
        pe.result_score = (
            pe.contribution_score * 0.4
            + pe.survival_days / max(total_days, 1) * 30
            + (30 if is_winner else 0)
        )

        # ---- 过程评测 ----
        # 发言质量：长度 + 信息量
        if speeches:
            avg_len = sum(len(s.get("content", "")) for s in speeches) / len(speeches)
            pe.speech_quality = min(100, avg_len * 0.8 + 20)
        else:
            pe.speech_quality = 20

        # 投票准确率：投给狼人（好人）/投给好人（狼人）
        if votes:
            correct_votes = 0
            for v in votes:
                target = v.get("target")
                if target is not None:
                    # 找target的角色
                    target_role = None
                    for d in death_records:
                        if d.get("player_id") == target:
                            target_role = d.get("role", "")
                            break
                    if camp == "good" and target_role == "werewolf":
                        correct_votes += 1
                    elif camp == "evil" and target_role != "werewolf":
                        correct_votes += 1
            pe.vote_accuracy = (correct_votes / len(votes)) * 100 if votes else 50
        else:
            pe.vote_accuracy = 50

        pe.info_usage = min(100, len(speeches) * 10 + len(votes) * 5 + 20)
        pe.process_score = pe.speech_quality * 0.35 + pe.vote_accuracy * 0.35 + pe.info_usage * 0.3

        # ---- 对抗评测 ----
        if camp == "evil":
            # 狼人：伪装能力
            pe.deception_score = self._calc_deception(pid, speeches, dialogues, is_winner)
            pe.adversarial_score = pe.deception_score * 0.6 + pe.process_score * 0.4
        else:
            # 好人：识别能力
            pe.detection_score = pe.vote_accuracy
            pe.adversarial_score = pe.detection_score * 0.6 + pe.process_score * 0.4

        pe.bluff_success_rate = self._calc_bluff_rate(pid, role, speeches, dialogues)

        # ---- 综合得分 ----
        pe.total_score = (
            pe.result_score * 0.35
            + pe.process_score * 0.35
            + pe.adversarial_score * 0.3
        )

        return pe

    def _calc_deception(self, pid: int, speeches: list, all_dialogues: list, is_winner: bool) -> float:
        """计算狼人欺骗分"""
        score = 40  # 基础分
        # 发言越多越好
        score += min(30, len(speeches) * 5)
        # 胜利加分
        if is_winner:
            score += 20
        # 没有被投票出局（伪装成功）
        was_voted = any(
            d.get("action") == "vote" and d.get("target") == pid
            for d in all_dialogues
        )
        if not was_voted:
            score += 10
        return min(100, score)

    def _calc_bluff_rate(self, pid: int, role: str, speeches: list, all_dialogues: list) -> float:
        """计算悍跳/挡刀成功率"""
        # 简化：检查是否有跳身份行为（发言中包含"我是"等关键词）
        jump_keywords = ["我是预言家", "我是女巫", "我是猎人", "我跳"]
        jump_count = 0
        for s in speeches:
            content = s.get("content", "")
            if any(kw in content for kw in jump_keywords):
                jump_count += 1
        if jump_count == 0:
            return 50  # 无跳身份行为，给中性分
        # 有跳身份行为，给高分（简化处理）
        return min(100, 60 + jump_count * 10)

    def _calc_game_quality(self, total_days: int, total_deaths: int,
                           player_count: int, evaluations: list) -> float:
        """计算对局质量分"""
        # 天数越多越精彩
        day_score = min(40, total_days * 8)
        # 死亡比例
        death_ratio = total_deaths / max(player_count, 1)
        death_score = death_ratio * 30
        # 平均得分
        avg_score = sum(e.total_score for e in evaluations) / max(len(evaluations), 1)
        return min(100, day_score + death_score + avg_score * 0.3)

    def _calc_balance(self, evaluations: list) -> float:
        """计算阵营平衡度"""
        good_scores = [e.total_score for e in evaluations if e.camp == "good"]
        evil_scores = [e.total_score for e in evaluations if e.camp == "evil"]
        if not good_scores or not evil_scores:
            return 50
        avg_good = sum(good_scores) / len(good_scores)
        avg_evil = sum(evil_scores) / len(evil_scores)
        diff = abs(avg_good - avg_evil)
        return max(0, 100 - diff)


class Leaderboard:
    """排行榜 - 记录多局游戏的评测结果"""

    def __init__(self, data_dir: str = "data/leaderboard"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.records: List[Dict[str, Any]] = []
        self._load()

    def _filepath(self) -> str:
        return os.path.join(self.data_dir, "leaderboard.json")

    def _load(self):
        fp = self._filepath()
        if os.path.exists(fp):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    self.records = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.records = []

    def _save(self):
        with open(self._filepath(), "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)

    def add_game(self, evaluation: GameEvaluation, model_name: str = "default", version: str = "v0.2.0"):
        """添加一局游戏评测结果"""
        record = {
            "game_id": evaluation.game_id,
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "version": version,
            "winner": evaluation.winner,
            "total_days": evaluation.total_days,
            "game_quality": round(evaluation.game_quality_score, 1),
            "players": [pe.to_dict() for pe in evaluation.player_evaluations],
        }
        self.records.append(record)
        self._save()

    def get_model_stats(self) -> List[Dict[str, Any]]:
        """按模型统计胜率等指标"""
        model_data: Dict[str, List] = {}
        for r in self.records:
            model = r.get("model", "default")
            if model not in model_data:
                model_data[model] = []
            model_data[model].append(r)

        stats = []
        for model, games in model_data.items():
            total = len(games)
            wins = sum(1 for g in games if g["winner"] == "good")  # 简化：好人胜率
            avg_quality = sum(g.get("game_quality", 0) for g in games) / max(total, 1)
            avg_days = sum(g.get("total_days", 1) for g in games) / max(total, 1)

            # 按角色统计
            role_scores: Dict[str, List[float]] = {}
            for g in games:
                for p in g.get("players", []):
                    role = p.get("role", "unknown")
                    score = p.get("total_score", 0)
                    if role not in role_scores:
                        role_scores[role] = []
                    role_scores[role].append(score)

            role_avg = {
                role: round(sum(scores) / len(scores), 1)
                for role, scores in role_scores.items()
            }

            stats.append({
                "model": model,
                "total_games": total,
                "win_rate": round(wins / max(total, 1) * 100, 1),
                "avg_game_quality": round(avg_quality, 1),
                "avg_days": round(avg_days, 1),
                "role_avg_scores": role_avg,
            })

        return sorted(stats, key=lambda x: x["win_rate"], reverse=True)

    def get_top_players(self, role: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最高分玩家"""
        all_players = []
        for r in self.records:
            for p in r.get("players", []):
                if role and p.get("role") != role:
                    continue
                all_players.append(p)

        all_players.sort(key=lambda x: x.get("total_score", 0), reverse=True)
        return all_players[:limit]

    def clear(self):
        """清空排行榜"""
        self.records = []
        self._save()
