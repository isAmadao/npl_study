"""测试评测与复盘模块"""

import pytest
import os
import json
import shutil
from unittest.mock import patch, AsyncMock

from evaluation.evaluator import (
    GameEvaluator,
    GameEvaluation,
    PlayerEvaluation,
    Leaderboard,
)
from evaluation.review_agent import ReviewAgent, build_timeline


# ============== PlayerEvaluation Tests ==============

class TestPlayerEvaluation:
    """测试 PlayerEvaluation 数据类"""

    def test_to_dict(self):
        pe = PlayerEvaluation(
            player_id=0, player_name="玩家1", role="werewolf",
            camp="evil", is_winner=True, result_score=85.5,
            process_score=72.3, adversarial_score=68.0, total_score=75.2,
        )
        d = pe.to_dict()
        assert d["player_id"] == 0
        assert d["total_score"] == 75.2
        assert d["is_winner"] is True
        assert "details" in d

    def test_default_values(self):
        pe = PlayerEvaluation(
            player_id=1, player_name="测试", role="villager",
            camp="good", is_winner=False,
        )
        assert pe.result_score == 0
        assert pe.total_score == 0


# ============== GameEvaluation Tests ==============

class TestGameEvaluation:
    """测试 GameEvaluation 数据类"""

    def test_to_dict(self):
        ge = GameEvaluation(
            game_id="test_001", winner="good",
            total_days=3, total_deaths=4,
            game_quality_score=75.0, balance_score=80.0,
        )
        d = ge.to_dict()
        assert d["game_id"] == "test_001"
        assert d["winner"] == "good"
        assert d["game_quality_score"] == 75.0


# ============== GameEvaluator Tests ==============

class TestGameEvaluator:
    """测试 GameEvaluator"""

    @pytest.fixture
    def sample_data(self):
        return {
            "game_id": "eval_001",
            "winner": "good",
            "players": [
                {"player_id": 0, "name": "玩家1", "role": "werewolf", "camp": "evil", "is_alive": False},
                {"player_id": 1, "name": "玩家2", "role": "seer", "camp": "good", "is_alive": True},
                {"player_id": 2, "name": "玩家3", "role": "villager", "camp": "good", "is_alive": True},
                {"player_id": 3, "name": "玩家4", "role": "werewolf", "camp": "evil", "is_alive": False},
            ],
            "death_records": [
                {"player_id": 0, "player_name": "玩家1", "role": "werewolf", "cause": "vote", "day": 2},
                {"player_id": 3, "player_name": "玩家4", "role": "werewolf", "cause": "vote", "day": 3},
            ],
            "dialogues": [
                {"day": 1, "player_id": 1, "player_name": "玩家2", "role": "seer", "action": "speech", "content": "我是预言家，查验了3号是狼人", "phase": "公开演讲"},
                {"day": 1, "player_id": 2, "player_name": "玩家3", "role": "villager", "action": "speech", "content": "我相信2号", "phase": "公开演讲"},
                {"day": 2, "player_id": 1, "player_name": "玩家2", "role": "seer", "action": "vote", "target": 0, "phase": "投票"},
                {"day": 2, "player_id": 2, "player_name": "玩家3", "role": "villager", "action": "vote", "target": 0, "phase": "投票"},
                {"day": 3, "player_id": 1, "player_name": "玩家2", "role": "seer", "action": "vote", "target": 3, "phase": "投票"},
            ],
            "vote_records": [],
            "last_words": [],
        }

    def test_evaluate_basic(self, sample_data):
        evaluator = GameEvaluator()
        result = evaluator.evaluate(**sample_data)
        assert result.game_id == "eval_001"
        assert result.winner == "good"
        assert len(result.player_evaluations) == 4

    def test_evaluate_winner_scores(self, sample_data):
        evaluator = GameEvaluator()
        result = evaluator.evaluate(**sample_data)
        # 好人应该赢
        for pe in result.player_evaluations:
            if pe.camp == "good":
                assert pe.is_winner is True
                assert pe.result_score > 0
            else:
                assert pe.is_winner is False

    def test_evaluate_total_scores_positive(self, sample_data):
        evaluator = GameEvaluator()
        result = evaluator.evaluate(**sample_data)
        for pe in result.player_evaluations:
            assert pe.total_score >= 0

    def test_evaluate_game_quality(self, sample_data):
        evaluator = GameEvaluator()
        result = evaluator.evaluate(**sample_data)
        assert result.game_quality_score >= 0
        assert result.game_quality_score <= 100

    def test_evaluate_balance(self, sample_data):
        evaluator = GameEvaluator()
        result = evaluator.evaluate(**sample_data)
        assert result.balance_score >= 0
        assert result.balance_score <= 100

    def test_evaluate_empty_data(self):
        evaluator = GameEvaluator()
        result = evaluator.evaluate(
            game_id="empty", winner="good",
            players=[], death_records=[], dialogues=[],
            vote_records=[], last_words=[],
        )
        assert result.game_id == "empty"
        assert len(result.player_evaluations) == 0


# ============== Leaderboard Tests ==============

class TestLeaderboard:
    """测试 Leaderboard"""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        return str(tmp_path / "test_leaderboard")

    def test_init_creates_dir(self, temp_dir):
        lb = Leaderboard(data_dir=temp_dir)
        assert os.path.exists(temp_dir)

    def test_add_and_get_stats(self, temp_dir):
        lb = Leaderboard(data_dir=temp_dir)
        # 创建模拟评测
        eval_game = GameEvaluation(
            game_id="lb_001", winner="good",
            total_days=3, total_deaths=2,
            game_quality_score=70.0,
            player_evaluations=[
                PlayerEvaluation(
                    player_id=0, player_name="P0", role="werewolf",
                    camp="evil", is_winner=False, total_score=50,
                ),
                PlayerEvaluation(
                    player_id=1, player_name="P1", role="seer",
                    camp="good", is_winner=True, total_score=80,
                ),
            ],
        )
        lb.add_game(eval_game, model_name="qwen-flash")
        stats = lb.get_model_stats()
        assert len(stats) == 1
        assert stats[0]["model"] == "qwen-flash"
        assert stats[0]["total_games"] == 1

    def test_get_top_players(self, temp_dir):
        lb = Leaderboard(data_dir=temp_dir)
        eval_game = GameEvaluation(
            game_id="lb_002", winner="good",
            total_days=2, total_deaths=1,
            player_evaluations=[
                PlayerEvaluation(
                    player_id=0, player_name="P0", role="werewolf",
                    camp="evil", is_winner=False, total_score=60,
                ),
                PlayerEvaluation(
                    player_id=1, player_name="P1", role="villager",
                    camp="good", is_winner=True, total_score=90,
                ),
            ],
        )
        lb.add_game(eval_game)
        top = lb.get_top_players(limit=5)
        assert len(top) >= 1
        assert top[0]["total_score"] >= top[-1]["total_score"]

    def test_clear(self, temp_dir):
        lb = Leaderboard(data_dir=temp_dir)
        eval_game = GameEvaluation(
            game_id="lb_003", winner="evil",
            total_days=1, total_deaths=3,
        )
        lb.add_game(eval_game)
        assert len(lb.records) > 0
        lb.clear()
        assert len(lb.records) == 0

    def test_persistence(self, temp_dir):
        lb1 = Leaderboard(data_dir=temp_dir)
        eval_game = GameEvaluation(
            game_id="lb_004", winner="good",
            total_days=4, total_deaths=5,
        )
        lb1.add_game(eval_game)

        lb2 = Leaderboard(data_dir=temp_dir)
        assert len(lb2.records) == 1
        assert lb2.records[0]["game_id"] == "lb_004"


# ============== build_timeline Tests ==============

class TestBuildTimeline:
    """测试 build_timeline"""

    def test_basic_timeline(self):
        dialogues = [
            {"day": 1, "player_id": 0, "action": "vote", "target": 2, "phase": "投票"},
            {"day": 2, "player_id": 1, "action": "hunter_shot", "target": 3, "phase": "猎人开枪"},
        ]
        death_records = [
            {"player_id": 2, "player_name": "玩家3", "role": "villager", "cause": "vote", "day": 1},
        ]
        timeline = build_timeline(dialogues, death_records)
        assert len(timeline) >= 2
        # 应该按天数排序
        days = [e["day"] for e in timeline]
        assert days == sorted(days)

    def test_empty_data(self):
        assert build_timeline([], []) == []

    def test_death_priority(self):
        dialogues = []
        death_records = [
            {"player_id": 0, "player_name": "玩家1", "role": "werewolf", "cause": "night_kill", "day": 1},
        ]
        timeline = build_timeline(dialogues, death_records)
        assert len(timeline) == 1
        assert "死亡" in timeline[0]["description"]


# ============== ReviewAgent Tests ==============

class TestReviewAgent:
    """测试 ReviewAgent"""

    def test_format_player_stats(self):
        players = [
            {"player_id": 0, "player_name": "玩家1", "role": "werewolf", "camp": "evil",
             "is_winner": False, "result_score": 50, "process_score": 60,
             "adversarial_score": 70, "total_score": 55},
        ]
        result = ReviewAgent._format_player_stats(players)
        assert "玩家1" in result
        assert "werewolf" in result

    def test_format_timeline(self):
        timeline = [
            {"day": 1, "phase": "死亡", "description": "玩家1死亡"},
            {"day": 2, "phase": "投票", "description": "玩家2投票"},
        ]
        result = ReviewAgent._format_timeline(timeline)
        assert "第1天" in result
        assert "玩家1死亡" in result

    def test_parse_json_valid(self):
        output = '{"key_turning_points": ["test"], "overall_review": "good"}'
        result = ReviewAgent._parse_json(output)
        assert result["overall_review"] == "good"

    def test_parse_json_invalid(self):
        result = ReviewAgent._parse_json("not json at all")
        assert "overall_review" in result
        assert result["key_turning_points"] == []
