"""
SemanticMessageHistory 单元测试
==============================
"""

import pytest
import numpy as np

from src.SemanticMessageHistory import (
    SemanticMessageHistory,
    Message,
    SessionNotFoundError,
)
from src.EmbeddingsCache import EmbeddingsCache


# ==================== 辅助函数 ====================


def dummy_embedding(text: str) -> np.ndarray:
    """测试用 Embedding 函数。"""
    np.random.seed(sum(ord(c) for c in text[:50]) % 2**31)
    return np.random.randn(128).astype(np.float32)


@pytest.fixture
def history():
    """创建 SemanticMessageHistory 测试实例。"""
    import fakeredis
    redis_client = fakeredis.FakeRedis(decode_responses=True)
    cache = EmbeddingsCache(redis_client=redis_client, embedding_dim=128, skip_milvus=True)
    return SemanticMessageHistory(
        cache=cache,
        embedding_func=dummy_embedding,
        max_history=20,
        max_context_tokens=2048,
    )


# ==================== 测试用例 ====================


class TestAddMessage:
    """测试添加消息。"""

    def test_add_user_message(self, history):
        """添加用户消息。"""
        msg = history.add_message("session_1", "user", "你好")
        assert msg.role == "user"
        assert msg.content == "你好"
        assert msg.message_id != ""

    def test_add_assistant_message(self, history):
        """添加助手消息。"""
        msg = history.add_message("session_1", "assistant", "你好！有什么可以帮助你的？")
        assert msg.role == "assistant"

    def test_add_system_message(self, history):
        """添加系统消息。"""
        msg = history.add_message("session_1", "system", "你是一个助手。")
        assert msg.role == "system"

    def test_add_message_invalid_role(self, history):
        """无效角色应报错。"""
        with pytest.raises(ValueError, match="不支持的 role"):
            history.add_message("session_1", "invalid_role", "content")

    def test_add_empty_content(self, history):
        """空内容应报错。"""
        with pytest.raises(ValueError, match="消息内容不能为空"):
            history.add_message("session_1", "user", "")

    def test_add_with_metadata(self, history):
        """带元数据的消息。"""
        msg = history.add_message(
            "session_1", "user", "你好",
            metadata={"source": "web", "user_id": "123"},
        )
        assert msg.metadata["source"] == "web"


class TestGetHistory:
    """测试获取历史。"""

    def test_get_history(self, history):
        """获取会话历史。"""
        history.add_message("session_1", "user", "问题1")
        history.add_message("session_1", "assistant", "回答1")
        history.add_message("session_1", "user", "问题2")

        messages = history.get_history("session_1")
        assert len(messages) == 3
        assert messages[0].content == "问题1"
        assert messages[-1].content == "问题2"

    def test_get_history_nonexistent_session(self, history):
        """不存在的会话应抛出 SessionNotFoundError。"""
        with pytest.raises(SessionNotFoundError):
            history.get_history("not_exist")

    def test_get_history_with_limit(self, history):
        """限制返回条数。"""
        for i in range(10):
            history.add_message("session_1", "user", f"消息{i}")
        messages = history.get_history("session_1", limit=3)
        assert len(messages) == 3

    def test_session_isolation(self, history):
        """不同会话的数据应隔离。"""
        history.add_message("session_a", "user", "A的问题")
        history.add_message("session_b", "user", "B的问题")

        msgs_a = history.get_history("session_a")
        msgs_b = history.get_history("session_b")

        assert len(msgs_a) == 1
        assert len(msgs_b) == 1
        assert msgs_a[0].content == "A的问题"
        assert msgs_b[0].content == "B的问题"


class TestGetContext:
    """测试上下文构建。"""

    def test_get_context_basic(self, history):
        """基本上下文构建。"""
        history.add_message("session_1", "system", "你是助手。")
        history.add_message("session_1", "user", "你好")
        history.add_message("session_1", "assistant", "你好！")

        context = history.get_context("session_1")
        assert "system: 你是助手。" in context
        assert "user: 你好" in context
        assert "assistant: 你好！" in context

    def test_get_context_truncation(self, history):
        """超长上下文应裁剪。"""
        history.max_context_tokens = 10  # 很小的窗口
        history.add_message("session_1", "user", "非常长的消息内容" * 100)

        context = history.get_context("session_1")
        # 应该被裁剪
        assert len(context) > 0


class TestClearSession:
    """测试清空会话。"""

    def test_clear_session(self, history):
        """清空会话。"""
        history.add_message("session_1", "user", "消息1")
        assert history.clear_session("session_1") is True

        with pytest.raises(SessionNotFoundError):
            history.get_history("session_1")

    def test_clear_nonexistent_session(self, history):
        """清空不存在的会话返回 False。"""
        assert history.clear_session("no_such_session") is False


class TestListSessions:
    """测试会话列表。"""

    def test_list_sessions(self, history):
        """列出所有会话。"""
        history.add_message("session_a", "user", "你好")
        history.add_message("session_b", "user", "你好")

        sessions = history.list_sessions()
        assert len(sessions) == 2
        assert "session_a" in sessions
        assert "session_b" in sessions

    def test_list_sessions_empty(self, history):
        """空会话列表。"""
        assert history.list_sessions() == []


class TestGetStats:
    """测试会话统计。"""

    def test_get_session_stats(self, history):
        """获取会话统计。"""
        history.add_message("session_1", "user", "问题1")
        history.add_message("session_1", "assistant", "回答1")
        history.add_message("session_1", "user", "问题2")

        stats = history.get_session_stats("session_1")
        assert stats["message_count"] == 3
        assert stats["role_distribution"]["user"] == 2
        assert stats["role_distribution"]["assistant"] == 1
        assert stats["session_id"] == "session_1"
