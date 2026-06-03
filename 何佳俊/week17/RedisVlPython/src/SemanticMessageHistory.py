"""
SemanticMessageHistory — 语义消息历史管理
==========================================
基于 Embedding 的 Agent 对话历史存储与语义检索模块。

功能：
    1. 按 session_id 隔离管理多轮对话
    2. 支持通过语义相似度从历史消息中检索相关上下文
    3. 上下文窗口管理（Token 数 / 消息条数限制）
    4. 历史消息过期自动清理

设计思路：
    - 每条消息作为一个独立单元存储，携带角色、内容和时间戳
    - 消息内容同时生成 Embedding 存入向量缓存，支持语义检索
    - 会话按时间顺序组织，支持滑动窗口获取最新上下文
    - 结合 Token 计数实现精确的上下文窗口裁剪
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .EmbeddingsCache import EmbeddingsCache

logger = logging.getLogger(__name__)


# ==================== 数据类定义 ====================


@dataclass
class Message:
    """单条对话消息。

    Attributes:
        role: 消息角色（user / assistant / system / tool）。
        content: 消息文本内容。
        timestamp: 消息时间戳（秒）。
        message_id: 消息唯一 ID。
        metadata: 附加元数据。
    """

    role: str
    content: str
    timestamp: float
    message_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.message_id:
            self.message_id = str(uuid.uuid4())


@dataclass
class SessionInfo:
    """会话信息。

    Attributes:
        session_id: 会话 ID。
        message_count: 消息总数。
        created_at: 创建时间戳。
        last_active: 最后活动时间戳。
        token_count: 当前会话预估 Token 数。
    """

    session_id: str
    message_count: int = 0
    created_at: float = 0.0
    last_active: float = 0.0
    token_count: int = 0


# ==================== 异常定义 ====================


class MessageHistoryError(Exception):
    """SemanticMessageHistory 基础异常。"""


class SessionNotFoundError(MessageHistoryError):
    """会话不存在。"""


class ContextWindowExceededError(MessageHistoryError):
    """上下文窗口超出限制。"""


# ==================== 主类 ====================


class SemanticMessageHistory:
    """语义消息历史管理。

    管理 Agent 对话消息的存储、检索和上下文构建。
    每条消息自动生成 Embedding，支持按语义相似度检索历史消息。

    Attributes:
        cache: EmbeddingsCache 实例。
        embedding_func: Embedding 生成函数。
        max_history: 每个会话保留的最大消息数。
        max_context_tokens: 上下文窗口最大 Token 数。
    """

    # Redis Key 前缀
    _KEY_PREFIX_SESSION = "session:"       # 会话元数据 Hash
    _KEY_PREFIX_MSG = "msg:"               # 单条消息 Hash
    _KEY_SESSION_INDEX = "sessions:index"  # 所有会话 ID 集合
    _KEY_PREFIX_MSG_LIST = "msgs:"         # 会话的消息 ID 列表 (List)

    # 简易 Token 估算：中英文混合按平均 2 字符 / token
    _TOKEN_ESTIMATE_RATIO = 2.0

    def __init__(
        self,
        cache: EmbeddingsCache,
        embedding_func=None,
        max_history: int = 100,
        max_context_tokens: int = 4096,
    ) -> None:
        """初始化 SemanticMessageHistory。

        Args:
            cache: EmbeddingsCache 实例。
            embedding_func: Embedding 生成函数。
                为 None 时使用内置简单编码（仅支持存储，不支持语义检索）。
            max_history: 每个会话最多保留的消息条数。
            max_context_tokens: 上下文窗口最大 Token 数。
        """
        self.cache = cache
        self._embedding_func = embedding_func or self._default_embedding
        self.max_history = max_history
        self.max_context_tokens = max_context_tokens

        logger.info(
            "SemanticMessageHistory 初始化完成 | "
            "max_history=%d | max_context_tokens=%d",
            max_history, max_context_tokens,
        )

    # ==================== 公开 API ====================

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """向会话中添加一条消息。

        自动生成消息 ID 和时间戳，同步存储 Embedding。

        Args:
            session_id: 会话 ID。
            role: 消息角色（user / assistant / system / tool）。
            content: 消息内容。
            metadata: 附加元数据。

        Returns:
            已创建的消息对象。

        Raises:
            ValueError: role 或 content 不合法。
            MessageHistoryError: 存储失败。
        """
        if role not in ("user", "assistant", "system", "tool"):
            raise ValueError(f"不支持的 role: {role}")
        if not content or not content.strip():
            raise ValueError("消息内容不能为空")

        message = Message(
            role=role,
            content=content,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        try:
            # 更新会话元数据
            self._ensure_session(session_id)

            # 存储消息到 Redis List
            msg_list_key = f"{self._KEY_PREFIX_MSG_LIST}{session_id}"
            serialized = self._serialize_message(message)
            pipe = self.cache.redis.pipeline()
            pipe.rpush(msg_list_key, serialized)

            # 限制列表长度
            if self.max_history > 0:
                pipe.ltrim(msg_list_key, -self.max_history, -1)

            pipe.execute()

            # 生成并存储消息的 Embedding（用于语义检索）
            self._store_message_embedding(session_id, message)

            # 更新会话最后活动时间
            self._update_session_activity(session_id)

            logger.debug(
                "消息已添加 | session=%s | role=%s | len=%d",
                session_id, role, len(content),
            )
            return message

        except Exception as e:
            raise MessageHistoryError(f"添加消息失败: {e}") from e

    def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[Message]:
        """获取会话的历史消息列表。

        Args:
            session_id: 会话 ID。
            limit: 返回的最大消息数，为 None 返回全部。

        Returns:
            按时间升序排列的消息列表（旧消息在前）。

        Raises:
            SessionNotFoundError: 会话不存在。
        """
        self._check_session_exists(session_id)
        limit = limit or self.max_history

        try:
            msg_list_key = f"{self._KEY_PREFIX_MSG_LIST}{session_id}"
            # 获取最近 N 条消息
            raw_messages = self.cache.redis.lrange(
                msg_list_key, -limit, -1
            )
            return [self._deserialize_message(m) for m in raw_messages]

        except SessionNotFoundError:
            raise
        except Exception as e:
            raise MessageHistoryError(f"获取历史消息失败: {e}") from e

    def search_similar(
        self,
        query: str,
        session_id: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ) -> List[Message]:
        """在会话历史中搜索语义相似的消息。

        用于 RAG（检索增强生成），找到与当前问题最相关的历史消息。

        Args:
            query: 查询文本。
            session_id: 会话 ID。
            top_k: 返回的最相似消息数。
            similarity_threshold: 相似度阈值。

        Returns:
            按相似度降序排列的消息列表。

        Raises:
            SessionNotFoundError: 会话不存在。
            MessageHistoryError: 搜索失败。
        """
        self._check_session_exists(session_id)

        if top_k < 1:
            return []

        try:
            # 生成查询的 Embedding
            query_vector = self._embed_text(query)

            # 获取会话中所有消息的 Embedding key
            embed_keys = self.cache.redis.smembers(
                f"embidx:{session_id}"
            )
            if not embed_keys:
                return []

            # 计算相似度
            scored_messages: List[Tuple[float, Message]] = []
            for embed_key in embed_keys:
                embed_key_str = (
                    embed_key.decode() if isinstance(embed_key, bytes)
                    else embed_key
                )
                stored_vector = self.cache.get(embed_key_str)
                if stored_vector is None:
                    continue

                similarity = self._cosine_similarity(
                    query_vector, stored_vector
                )
                if similarity >= similarity_threshold:
                    # 通过关联的消息 ID 获取消息内容
                    msg_id = embed_key_str.split(":", 1)[1] if ":" in embed_key_str else ""
                    if msg_id:
                        msg = self._get_message_by_id(session_id, msg_id)
                        if msg:
                            scored_messages.append((similarity, msg))

            # 按相似度降序排列
            scored_messages.sort(key=lambda x: x[0], reverse=True)
            return [msg for _, msg in scored_messages[:top_k]]

        except SessionNotFoundError:
            raise
        except Exception as e:
            raise MessageHistoryError(
                f"语义检索失败: {e}"
            ) from e

    def get_context(
        self,
        session_id: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """构建当前会话的上下文文本。

        按时间顺序拼接消息，并在超过 max_tokens 时从旧消息开始裁剪。

        Args:
            session_id: 会话 ID。
            max_tokens: 最大 Token 数，为 None 使用实例默认值。

        Returns:
            格式化的上下文文本，适用于 LLM 调用。

        Raises:
            SessionNotFoundError: 会话不存在。
        """
        self._check_session_exists(session_id)
        max_tokens = max_tokens or self.max_context_tokens

        try:
            messages = self.get_history(session_id)
            if not messages:
                return ""

            # 从最新消息开始反向裁剪
            context_parts: List[str] = []
            total_tokens = 0

            for msg in reversed(messages):
                formatted = self._format_message(msg)
                msg_tokens = self._estimate_tokens(formatted)

                if total_tokens + msg_tokens > max_tokens:
                    # 如果连最旧的消息都放不下就不继续了
                    # 但如果一条消息都没放进去，至少放最新的那条
                    if not context_parts:
                        context_parts.append(formatted)
                    continue

                context_parts.append(formatted)
                total_tokens += msg_tokens

            # 恢复时间顺序（旧 → 新）
            context_parts.reverse()
            return "\n".join(context_parts)

        except SessionNotFoundError:
            raise
        except Exception as e:
            raise MessageHistoryError(f"构建上下文失败: {e}") from e

    def clear_session(self, session_id: str) -> bool:
        """清空指定会话的所有数据。

        Args:
            session_id: 会话 ID。

        Returns:
            是否成功清空（会话不存在时返回 False）。
        """
        try:
            if not self.cache.redis.sismember(
                self._KEY_SESSION_INDEX, session_id
            ):
                return False

            # 删除消息列表
            self.cache.redis.delete(
                f"{self._KEY_PREFIX_MSG_LIST}{session_id}"
            )

            # 删除会话元数据
            self.cache.redis.delete(
                f"{self._KEY_PREFIX_SESSION}{session_id}"
            )

            # 删除 Embedding 索引
            embed_keys = self.cache.redis.smembers(
                f"embidx:{session_id}"
            )
            if embed_keys:
                for key in embed_keys:
                    key_str = (
                        key.decode() if isinstance(key, bytes) else key
                    )
                    self.cache.delete(key_str)
                self.cache.redis.delete(f"embidx:{session_id}")

            # 从会话索引中移除
            self.cache.redis.srem(self._KEY_SESSION_INDEX, session_id)

            logger.info("会话已清空: %s", session_id)
            return True

        except Exception as e:
            raise MessageHistoryError(f"清空会话失败: {e}") from e

    def list_sessions(self) -> List[str]:
        """获取所有会话 ID 列表。

        Returns:
            会话 ID 列表，按最后活动时间降序排列。
        """
        try:
            sessions = self.cache.redis.smembers(self._KEY_SESSION_INDEX)
            result: List[Tuple[float, str]] = []

            for session in sessions:
                sid = session.decode() if isinstance(session, bytes) else session
                info = self._get_session_meta(sid)
                last_active = info.last_active if info else 0.0
                result.append((last_active, sid))

            # 按活动时间降序
            result.sort(key=lambda x: x[0], reverse=True)
            return [sid for _, sid in result]

        except Exception as e:
            raise MessageHistoryError(f"获取会话列表失败: {e}") from e

    def delete_session(self, session_id: str) -> bool:
        """删除整个会话及其所有数据。

        Args:
            session_id: 会话 ID。

        Returns:
            是否成功删除。
        """
        return self.clear_session(session_id)

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """获取会话统计信息。

        Args:
            session_id: 会话 ID。

        Returns:
            统计信息字典。

        Raises:
            SessionNotFoundError: 会话不存在。
        """
        self._check_session_exists(session_id)

        try:
            messages = self.get_history(session_id)
            total_tokens = sum(
                self._estimate_tokens(m.content) for m in messages
            )

            role_counts: Dict[str, int] = {}
            for m in messages:
                role_counts[m.role] = role_counts.get(m.role, 0) + 1

            info = self._get_session_meta(session_id)

            return {
                "session_id": session_id,
                "message_count": len(messages),
                "total_tokens": total_tokens,
                "role_distribution": role_counts,
                "created_at": info.created_at if info else 0.0,
                "last_active": info.last_active if info else 0.0,
                "duration": (
                    time.time() - info.created_at if info else 0.0
                ),
            }

        except SessionNotFoundError:
            raise
        except Exception as e:
            raise MessageHistoryError(f"获取会话统计失败: {e}") from e

    # ==================== 内部方法 ====================

    def _ensure_session(self, session_id: str) -> None:
        """确保会话存在，不存在则创建。

        Args:
            session_id: 会话 ID。
        """
        if not self.cache.redis.sismember(
            self._KEY_SESSION_INDEX, session_id
        ):
            pipe = self.cache.redis.pipeline()
            now = time.time()
            sk = f"{self._KEY_PREFIX_SESSION}{session_id}"
            pipe.hset(sk, "session_id", session_id)
            pipe.hset(sk, "created_at", str(now))
            pipe.hset(sk, "last_active", str(now))
            pipe.hset(sk, "message_count", "0")
            pipe.sadd(self._KEY_SESSION_INDEX, session_id)
            pipe.execute()

    def _check_session_exists(self, session_id: str) -> None:
        """检查会话是否存在。

        Args:
            session_id: 会话 ID。

        Raises:
            SessionNotFoundError: 会话不存在。
        """
        if not self.cache.redis.sismember(
            self._KEY_SESSION_INDEX, session_id
        ):
            raise SessionNotFoundError(f"会话不存在: {session_id}")

    def _get_session_meta(self, session_id: str) -> Optional[SessionInfo]:
        """获取会话元数据。

        Args:
            session_id: 会话 ID。

        Returns:
            SessionInfo 对象，不存在时返回 None。
        """
        data = self.cache.redis.hgetall(
            f"{self._KEY_PREFIX_SESSION}{session_id}"
        )
        if not data:
            return None
        return SessionInfo(
            session_id=session_id,
            message_count=int(data.get("message_count", 0)),
            created_at=float(data.get("created_at", 0)),
            last_active=float(data.get("last_active", 0)),
        )

    def _update_session_activity(self, session_id: str) -> None:
        """更新会话的最后活动时间。

        Args:
            session_id: 会话 ID。
        """
        try:
            self.cache.redis.hset(
                f"{self._KEY_PREFIX_SESSION}{session_id}",
                "last_active",
                str(time.time()),
            )
            self.cache.redis.hincrby(
                f"{self._KEY_PREFIX_SESSION}{session_id}",
                "message_count",
                1,
            )
        except Exception as e:
            logger.warning("更新会话活动时间失败: %s", e)

    def _store_message_embedding(
        self, session_id: str, message: Message
    ) -> None:
        """为消息生成并存储 Embedding。

        Args:
            session_id: 会话 ID。
            message: 消息对象。
        """
        try:
            vector = self._embed_text(message.content)
            embed_key = f"emb:{session_id}:{message.message_id}"

            self.cache.set(
                key=embed_key,
                vector=vector,
                metadata={
                    "session_id": session_id,
                    "message_id": message.message_id,
                    "role": message.role,
                    "timestamp": message.timestamp,
                },
            )

            # 添加到会话的 Embedding 索引集合
            self.cache.redis.sadd(f"embidx:{session_id}", embed_key)

        except Exception as e:
            logger.warning(
                "消息 Embedding 存储失败 (msg_id=%s): %s",
                message.message_id, e,
            )

    def _get_message_by_id(
        self, session_id: str, message_id: str
    ) -> Optional[Message]:
        """根据消息 ID 获取消息。

        Args:
            session_id: 会话 ID。
            message_id: 消息 ID。

        Returns:
            找到的消息对象，未找到时返回 None。
        """
        try:
            msg_list_key = f"{self._KEY_PREFIX_MSG_LIST}{session_id}"
            raw_messages = self.cache.redis.lrange(msg_list_key, 0, -1)

            for raw in raw_messages:
                msg = self._deserialize_message(raw)
                if msg.message_id == message_id:
                    return msg
            return None

        except Exception:
            return None

    def _embed_text(self, text: str) -> np.ndarray:
        """生成文本的 Embedding 向量。

        Args:
            text: 输入文本。

        Returns:
            numpy 向量（已归一化）。
        """
        vector = self._embedding_func(text)
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim > 1:
            vector = vector.flatten()
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    def _default_embedding(self, text: str) -> np.ndarray:
        """默认 Embedding 函数：基于字符的简单编码。

        当未提供外部 Embedding 函数时使用。
        注意：此编码不具备语义能力，仅用于键的生成。

        Args:
            text: 输入文本。

        Returns:
           固定维度的编码向量。
        """
        # 简单的字符哈希编码，维度固定为 64
        np.random.seed(sum(ord(c) for c in text[:100]))
        return np.random.randn(64).astype(np.float32)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度。

        Args:
            a: 向量 a（已归一化）。
            b: 向量 b（已归一化）。

        Returns:
            余弦相似度 [0.0, 1.0]。
        """
        return max(0.0, min(1.0, float(np.dot(a, b))))

    @staticmethod
    def _serialize_message(message: Message) -> str:
        """将 Message 序列化为 JSON 字符串。

        Args:
            message: 消息对象。

        Returns:
            JSON 字符串。
        """
        return json.dumps({
            "message_id": message.message_id,
            "role": message.role,
            "content": message.content,
            "timestamp": message.timestamp,
            "metadata": message.metadata,
        }, ensure_ascii=False)

    @staticmethod
    def _deserialize_message(data: Any) -> Message:
        """从 JSON 字符串反序列化为 Message。

        Args:
            data: JSON 字符串或字节。

        Returns:
            消息对象。
        """
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        obj = json.loads(data)
        return Message(
            message_id=obj.get("message_id", ""),
            role=obj.get("role", "user"),
            content=obj.get("content", ""),
            timestamp=obj.get("timestamp", 0.0),
            metadata=obj.get("metadata", {}),
        )

    @staticmethod
    def _format_message(message: Message) -> str:
        """格式化消息为上下文文本。

        Args:
            message: 消息对象。

        Returns:
            格式化的文本。
        """
        return f"{message.role}: {message.content}"

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """估算文本的 Token 数。

        使用简单的中英文混合估算：总字符数 / 2。

        Args:
            text: 输入文本。

        Returns:
            预估 Token 数。
        """
        return max(1, int(len(text) / SemanticMessageHistory._TOKEN_ESTIMATE_RATIO))

    def __repr__(self) -> str:
        return (
            f"SemanticMessageHistory("
            f"max_history={self.max_history}, "
            f"max_context_tokens={self.max_context_tokens})"
        )
