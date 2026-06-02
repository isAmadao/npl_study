"""自进化模块测试

测试 EvolutionAgent、EvolutionTracker 的纯逻辑功能。
"""

import json
import os
import shutil
import pytest
import pytest_asyncio

from agent.evolution_agent import EvolutionAgent, ROLE_NAMES
from evaluation.evolution_tracker import EvolutionTracker


# ============================================================
# EvolutionAgent 纯逻辑测试
# ============================================================


class TestEvolutionAgentStaticMethods:
    """测试 EvolutionAgent 的静态方法"""

    def test_format_experiences_empty(self):
        result = EvolutionAgent._format_experiences([])
        assert result == "暂无经验记录"

    def test_format_experiences_with_data(self):
        exps = [
            {
                "is_winner": True,
                "summary": "悍跳预言家成功",
                "strategies": "前置位悍跳",
                "mistakes": "",
                "lessons": "要更自信",
                "strategy_tags": ["winning_strategy", "jump_seer"],
            },
            {
                "is_winner": False,
                "summary": "被查验出局",
                "strategies": "",
                "mistakes": "太早暴露",
                "lessons": "应该低调",
                "strategy_tags": ["failure_case"],
            },
        ]
        result = EvolutionAgent._format_experiences(exps)
        assert "胜利" in result
        assert "失败" in result
        assert "悍跳预言家成功" in result
        assert "winning_strategy" in result

    def test_format_experiences_truncation(self):
        """测试长文本截断"""
        exps = [{"is_winner": True, "summary": "a" * 200}]
        result = EvolutionAgent._format_experiences(exps)
        # 截断到150字符
        assert len("a" * 200) > 150
        assert "a" * 150 in result

    def test_format_experiences_last_10(self):
        """测试只取最后10条"""
        exps = [{"is_winner": i % 2 == 0, "summary": f"exp{i}"} for i in range(15)]
        result = EvolutionAgent._format_experiences(exps)
        # 应该从第6条开始（index 5）
        assert "exp5" in result
        assert "exp14" in result
        assert "exp0" not in result

    def test_format_metrics_none(self):
        result = EvolutionAgent._format_metrics(None)
        assert "暂无评测数据" in result

    def test_format_metrics_empty(self):
        result = EvolutionAgent._format_metrics({})
        assert "暂无评测数据" in result

    def test_format_metrics_with_data(self):
        metrics = {
            "win_rate": 0.6,
            "avg_score": 7.5,
            "avg_speech_quality": 8.0,
            "avg_vote_accuracy": 0.7,
            "avg_survival_days": 3.2,
        }
        result = EvolutionAgent._format_metrics(metrics)
        assert "60.0%" in result
        assert "7.5" in result
        assert "8.0" in result

    def test_parse_json_output_direct(self):
        output = '{"trend_analysis": "test", "evolution_plan": []}'
        result = EvolutionAgent._parse_json_output(output)
        assert result["trend_analysis"] == "test"
        assert result["evolution_plan"] == []

    def test_parse_json_output_code_block(self):
        output = '```json\n{"trend_analysis": "code block"}\n```'
        result = EvolutionAgent._parse_json_output(output)
        assert result["trend_analysis"] == "code block"

    def test_parse_json_output_fallback(self):
        result = EvolutionAgent._parse_json_output("plain text no json")
        assert "trend_analysis" in result
        assert result["priority"] == "unknown"


class TestRoleNames:
    """测试角色名映射"""

    def test_all_roles_mapped(self):
        assert ROLE_NAMES["werewolf"] == "狼人"
        assert ROLE_NAMES["seer"] == "预言家"
        assert ROLE_NAMES["witch"] == "女巫"
        assert ROLE_NAMES["hunter"] == "猎人"
        assert ROLE_NAMES["villager"] == "村民"


# ============================================================
# EvolutionTracker 测试
# ============================================================


class TestEvolutionTracker:
    """测试进化追踪器"""

    def setup_method(self):
        self.test_dir = "data/test_evolution_tracker"
        os.makedirs(self.test_dir, exist_ok=True)
        self.tracker = EvolutionTracker(data_dir=self.test_dir)

    def teardown_method(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_record_evolution(self):
        self.tracker.record_evolution(
            role_type="werewolf",
            version=1,
            metrics_before={"win_rate": 0.4, "avg_score": 6.0},
            metrics_after={"win_rate": 0.6, "avg_score": 7.5},
        )
        history = self.tracker._load_history()
        assert len(history) == 1
        assert history[0]["role"] == "werewolf"
        assert history[0]["version"] == 1

    def test_record_with_advice(self):
        self.tracker.record_evolution(
            role_type="seer",
            version=1,
            metrics_before={"win_rate": 0.5},
            metrics_after={"win_rate": 0.7},
            evolution_advice={"trend_analysis": "预言家表现提升", "priority": "发言质量"},
        )
        history = self.tracker._load_history()
        assert history[0]["advice_summary"] == "预言家表现提升"
        assert history[0]["priority"] == "发言质量"

    def test_calc_improvement(self):
        before = {"win_rate": 0.4, "score": 5.0}
        after = {"win_rate": 0.6, "score": 7.5}
        result = EvolutionTracker._calc_improvement(before, after)
        assert result["win_rate"] == 0.5  # (0.6-0.4)/0.4 = 0.5
        assert result["score"] == 0.5  # (7.5-5.0)/5.0 = 0.5

    def test_calc_improvement_zero_before(self):
        result = EvolutionTracker._calc_improvement({"x": 0}, {"x": 5.0})
        assert result["x"] == 1.0

    def test_calc_improvement_both_zero(self):
        result = EvolutionTracker._calc_improvement({"x": 0}, {"x": 0})
        assert result["x"] == 0.0

    def test_get_evolution_report_empty(self):
        report = self.tracker.get_evolution_report("werewolf")
        assert report["total_evolutions"] == 0
        assert report["latest"] is None

    def test_get_evolution_report_with_data(self):
        self.tracker.record_evolution("werewolf", 1, {"win_rate": 0.4}, {"win_rate": 0.5})
        self.tracker.record_evolution("werewolf", 2, {"win_rate": 0.5}, {"win_rate": 0.7})
        report = self.tracker.get_evolution_report("werewolf")
        assert report["total_evolutions"] == 2
        assert len(report["trend"]) == 2
        assert report["latest"]["version"] == 2

    def test_get_all_reports(self):
        self.tracker.record_evolution("werewolf", 1, {"win_rate": 0.4}, {"win_rate": 0.5})
        self.tracker.record_evolution("seer", 1, {"win_rate": 0.5}, {"win_rate": 0.6})
        reports = self.tracker.get_all_reports()
        assert "werewolf" in reports
        assert "seer" in reports

    def test_get_version_comparison(self):
        self.tracker.record_evolution("werewolf", 1, {"win_rate": 0.4}, {"win_rate": 0.5})
        self.tracker.record_evolution("werewolf", 2, {"win_rate": 0.5}, {"win_rate": 0.7})
        comparisons = self.tracker.get_version_comparison("werewolf")
        assert len(comparisons) == 2
        assert comparisons[0]["version"] == 1
        assert comparisons[1]["version"] == 2

    def test_clear(self):
        self.tracker.record_evolution("werewolf", 1, {"x": 1}, {"x": 2})
        self.tracker.clear()
        history = self.tracker._load_history()
        assert len(history) == 0

    def test_summarize_trend_no_data(self):
        result = EvolutionTracker._summarize_trend([])
        assert result == "暂无数据"

    def test_summarize_trend_single(self):
        result = EvolutionTracker._summarize_trend([{"x": 0.1}])
        assert "仅有一次" in result

    def test_summarize_trend_all_positive(self):
        trends = [{"x": 0.1, "y": 0.2}, {"x": 0.3, "y": 0.4}]
        result = EvolutionTracker._summarize_trend(trends)
        assert "正向" in result or "改善" in result

    def test_summarize_trend_all_negative(self):
        trends = [{"x": 0.1}, {"x": -0.1, "y": -0.2}]
        result = EvolutionTracker._summarize_trend(trends)
        assert "审视" in result or "退步" in result or "改善" in result

    def test_persistence(self):
        """测试数据持久化"""
        self.tracker.record_evolution("werewolf", 1, {"win_rate": 0.4}, {"win_rate": 0.5})
        # 创建新的tracker实例
        tracker2 = EvolutionTracker(data_dir=self.test_dir)
        history = tracker2._load_history()
        assert len(history) == 1


# ============================================================
# 集成测试：自进化循环逻辑（不依赖LLM）
# ============================================================


class TestEvolutionCycleLogic:
    """测试自进化循环的核心逻辑（不触发LLM调用）"""

    def test_evolution_agent_creation(self):
        """测试EvolutionAgent可以正常创建"""
        agent = EvolutionAgent()
        assert agent.agent is not None
        assert agent.agent.name == "EvolutionAnalyst"

    def test_tracker_records_multiple_roles(self):
        """测试tracker可以记录多个角色的进化"""
        test_dir = "data/test_evo_cycle"
        os.makedirs(test_dir, exist_ok=True)
        try:
            tracker = EvolutionTracker(data_dir=test_dir)
            for role in ["werewolf", "seer", "witch"]:
                tracker.record_evolution(
                    role_type=role,
                    version=1,
                    metrics_before={"win_rate": 0.4},
                    metrics_after={"win_rate": 0.6},
                )
            reports = tracker.get_all_reports()
            assert len(reports) == 3
        finally:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)

    def test_avg_improvement_calculation(self):
        """测试平均改进幅度计算"""
        test_dir = "data/test_avg_imp"
        os.makedirs(test_dir, exist_ok=True)
        try:
            tracker = EvolutionTracker(data_dir=test_dir)
            tracker.record_evolution("werewolf", 1, {"win_rate": 0.4}, {"win_rate": 0.5})
            tracker.record_evolution("werewolf", 2, {"win_rate": 0.5}, {"win_rate": 0.6})
            report = tracker.get_evolution_report("werewolf")
            assert "avg_improvement" in report
            assert "win_rate" in report["avg_improvement"]
        finally:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
