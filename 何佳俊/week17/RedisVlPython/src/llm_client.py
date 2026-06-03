"""
LLM Client — 大语言模型调用客户端
===================================
封装 Qwen（通义千问）DashScope 对话 API，提供统一的 LLM 调用接口。

功能：
    1. 非流式 / 流式响应
    2. 多轮对话（system / user / assistant / tool）
    3. 自动从 .env / 环境变量读取 API Key
    4. Token 计数和成本统计
    5. 自动重试（网络抖动容错）
    6. 兼容 OpenAI API（DashScope 兼容模式 / 纯 OpenAI）

设计原则：
    - DashScope SDK 优先，失败时降级到 OpenAI 兼容接口
    - 所有 Token 消耗可追踪可统计
    - 线程安全（不共享可变状态）

运行方式：
    pip install dashscope
    # 设置 DASHSCOPE_API_KEY 或 LLM_API_KEY
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ==================== 配置常量 ====================

# Qwen 模型定价（元 / 1K tokens，中国站价格）
# 参考 https://help.aliyun.com/zh/model-studio/getting-started/models
QWEN_PRICING: Dict[str, Dict[str, float]] = {
    "qwen-turbo": {"input": 0.0003, "output": 0.0006},
    "qwen-plus": {"input": 0.0008, "output": 0.002},
    "qwen-max": {"input": 0.02, "output": 0.06},
    "qwen-max-longcontext": {"input": 0.02, "output": 0.06},
    "qwen2.5-72b-instruct": {"input": 0.004, "output": 0.012},
    "qwen2.5-32b-instruct": {"input": 0.002, "output": 0.006},
    "qwen2.5-14b-instruct": {"input": 0.001, "output": 0.002},
    "qwen2.5-7b-instruct": {"input": 0.0005, "output": 0.001},
    "default": {"input": 0.001, "output": 0.002},  # 兜底价格
}

# OpenAI 定价（USD / 1K tokens）
OPENAI_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "default": {"input": 0.001, "output": 0.002},
}

# 重试配置
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_DELAY = 1.0  # 秒
_DEFAULT_TIMEOUT = 60  # 秒


# ==================== 数据类 ====================


class MessageRole(str, Enum):
    """消息角色枚举。"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class LLMProvider(str, Enum):
    """LLM 提供方枚举。"""

    DASHSCOPE = "dashscope"  # Qwen 原生 DashScope SDK
    OPENAI_COMPAT = "openai_compat"  # OpenAI 兼容接口（也可用 Qwen）
    OPENAI = "openai"  # 原生 OpenAI


@dataclass
class Message:
    """单条对话消息。

    Attributes:
        role: 消息角色。
        content: 消息文本内容。
        name: 可选的发送者名称（工具调用时使用）。
    """

    role: Union[MessageRole, str]
    content: str
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, str]:
        """转换为 API 请求的字典格式。

        Returns:
            包含 role 和 content 的字典。
        """
        d: Dict[str, str] = {"role": self.role.value
                             if isinstance(self.role, MessageRole)
                             else self.role,
                             "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class LLMResponse:
    """LLM 调用响应。

    Attributes:
        text: 生成的文本内容。
        finish_reason: 结束原因（stop / length / content_filter 等）。
        input_tokens: 输入 Token 数。
        output_tokens: 输出 Token 数。
        total_tokens: 总 Token 数。
        cost: 估算成本（元 / USD）。
        latency: 调用耗时（秒）。
        model: 使用的模型名称。
        provider: LLM 提供方。
        raw: 原始 API 响应（调试用）。
    """

    text: str = ""
    finish_reason: str = "stop"
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    latency: float = 0.0
    model: str = ""
    provider: LLMProvider = LLMProvider.DASHSCOPE
    raw: Any = None


@dataclass
class LLMUsageStats:
    """LLM 调用统计汇总。

    Attributes:
        total_calls: 总调用次数。
        total_input_tokens: 总输入 Token 数。
        total_output_tokens: 总输出 Token 数。
        total_cost: 总成本。
        total_latency: 总耗时（秒）。
        model_breakdown: 按模型的调用次数统计。
    """

    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    total_latency: float = 0.0
    model_breakdown: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": round(self.total_cost, 6),
            "total_latency": round(self.total_latency, 2),
            "avg_latency": round(self.total_latency / self.total_calls, 2)
            if self.total_calls > 0 else 0.0,
            "model_breakdown": dict(self.model_breakdown),
        }


# ==================== 异常定义 ====================


class LLMClientError(Exception):
    """LLM Client 基础异常。"""


class LLMAuthenticationError(LLMClientError):
    """API 认证失败（无效的 API Key）。"""


class LLMRateLimitError(LLMClientError):
    """API 限流（请求过频繁）。"""


class LLMServiceError(LLMClientError):
    """LLM 服务端错误。"""


class LLMConfigurationError(LLMClientError):
    """LLM 客户端配置错误。"""


# ==================== 主类 ====================


class LLMClient:
    """大语言模型调用客户端。

    统一封装 Qwen（DashScope）和 OpenAI API 的调用，
    提供流式 / 非流式、多轮对话、Token 追踪、成本统计等功能。

    Attributes:
        model: 模型名称（如 qwen-turbo, qwen-plus, gpt-4o）。
        provider: LLM 提供方。
        api_key: API Key。
        api_base: API 基础地址（仅 OpenAI 兼容模式）。
        max_retries: 最大重试次数。
        timeout: 请求超时（秒）。
        stats: 调用统计（全局累加）。
    """

    def __init__(
        self,
        model: str = "qwen-turbo",
        provider: Optional[Union[str, LLMProvider]] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        timeout: int = _DEFAULT_TIMEOUT,
        enable_stats: bool = True,
    ) -> None:
        """初始化 LLMClient。

        自动检测可用的 LLM 提供方和 API Key，优先级：
          1. 显式传入的 provider 和 api_key
          2. 环境变量 LLM_PROVIDER / DASHSCOPE_API_KEY / OPENAI_API_KEY
          3. .env 文件中的配置

        Args:
            model: 模型名称。默认 "qwen-turbo"。
            provider: LLM 提供方。为 None 时自动检测：
                dashscope → 优先使用 DashScope SDK
                openai_compat → OpenAI 兼容接口
                openai → 原生 OpenAI
            api_key: API Key。为 None 时从环境变量读取。
            api_base: API 基础地址（仅 OpenAI 兼容模式有效）。
            max_retries: 最大重试次数。
            timeout: 请求超时秒数。
            enable_stats: 是否启用调用统计。

        Raises:
            LLMConfigurationError: 无法找到可用的 API Key 或 SDK 时抛出。
        """
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.enable_stats = enable_stats
        self.stats = LLMUsageStats()

        # 确定 provider
        self.provider = self._resolve_provider(provider)

        # 确定 API Key 和 Base
        # 自动检测优先级: DASHSCOPE_API_KEY > LLM_API_KEY > EMBEDDING_API_KEY
        def _find_api_key(*names: str) -> Optional[str]:
            for name in names:
                val = os.environ.get(name)
                if val:
                    return val
            return None

        if self.provider == LLMProvider.DASHSCOPE:
            self.api_key = (api_key or _find_api_key("DASHSCOPE_API_KEY", "LLM_API_KEY", "EMBEDDING_API_KEY"))
            self.api_base = None  # DashScope SDK 使用默认端点
            if not self.api_key:
                raise LLMConfigurationError(
                    "DashScope API Key 未设置。请设置 DASHSCOPE_API_KEY 或 LLM_API_KEY"
                )
            self._check_dashscope_sdk()
        elif self.provider == LLMProvider.OPENAI_COMPAT:
            self.api_key = api_key or _find_api_key("LLM_API_KEY", "OPENAI_API_KEY", "EMBEDDING_API_KEY")
            self.api_base = api_base or os.environ.get("LLM_API_BASE") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            if not self.api_key:
                raise LLMConfigurationError(
                    "API Key 未设置。请设置 LLM_API_KEY 或 OPENAI_API_KEY"
                )
            self._check_openai_sdk()
        else:  # OpenAI
            self.api_key = api_key or _find_api_key("OPENAI_API_KEY", "LLM_API_KEY")
            self.api_base = api_base or os.environ.get("OPENAI_API_BASE") or "https://api.openai.com/v1"
            if not self.api_key:
                raise LLMConfigurationError(
                    "OpenAI API Key 未设置。请设置 OPENAI_API_KEY"
                )
            self._check_openai_sdk()

        logger.info(
            "LLMClient 初始化 | model=%s | provider=%s",
            self.model, self.provider.value,
        )

    # ==================== 公开 API ====================

    def chat(
        self,
        messages: List[Union[Message, Dict[str, str]]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9,
        stream: bool = False,
    ) -> LLMResponse:
        """执行一次对话补全调用。

        Args:
            messages: 对话消息列表。可以是 Message 对象或 {"role": ..., "content": ...} 字典。
            system_prompt: 系统提示词。会作为 system 消息插入到消息列表最前面。
            temperature: 采样温度 (0.0 ~ 2.0)，越高越随机。
            max_tokens: 最大生成 Token 数。为 None 时不限制。
            top_p: 核采样参数 (0.0 ~ 1.0)。
            stream: 是否使用流式响应（默认 False）。

        Returns:
            LLMResponse 包含生成的文本和 Token 使用统计。

        Raises:
            LLMAuthenticationError: API Key 无效。
            LLMRateLimitError: 请求过频繁被限流。
            LLMServiceError: 服务端错误。
            LLMClientError: 其他客户端错误。
        """
        # 1. 转换消息格式
        api_messages = self._prepare_messages(messages, system_prompt)

        # 2. 根据 provider 选择调用方式
        start_time = time.time()
        try:
            if stream:
                response = self._chat_stream(api_messages, temperature, max_tokens, top_p, start_time)
            else:
                response = self._chat_non_stream(api_messages, temperature, max_tokens, top_p, start_time)
            if self.enable_stats:
                self._update_global_stats(response)
            return response
        except LLMClientError:
            raise
        except Exception as e:
            raise LLMClientError(f"LLM 调用失败: {e}") from e

    def chat_stream(
        self,
        messages: List[Union[Message, Dict[str, str]]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9,
    ) -> Generator[str, None, LLMResponse]:
        """流式对话补全（生成器）。

        每次 yield 一个文本块，最终返回完整的 LLMResponse（含 Token 统计）。

        Args:
            messages: 对话消息列表。
            system_prompt: 系统提示词。
            temperature: 采样温度。
            max_tokens: 最大生成 Token 数。
            top_p: 核采样参数。

        Yields:
            文本块（str）。

        Returns:
            完整的 LLMResponse（含 Token 统计）。
        """
        api_messages = self._prepare_messages(messages, system_prompt)
        start_time = time.time()

        collected_text = ""
        response = LLMResponse(model=self.model, provider=self.provider)

        try:
            if self.provider == LLMProvider.DASHSCOPE:
                yield from self._stream_dashscope(api_messages, temperature,
                                                  max_tokens, top_p,
                                                  collected_text, response)
            else:
                yield from self._stream_openai(api_messages, temperature,
                                               max_tokens, top_p,
                                               collected_text, response)
        except Exception as e:
            raise LLMClientError(f"流式调用失败: {e}") from e
        finally:
            response.latency = time.time() - start_time
            response.cost = self._estimate_cost(response)
            if self.enable_stats:
                self._update_global_stats(response)

        return response

    def count_tokens(self, text: str) -> int:
        """估算文本的 Token 数。

        使用简单的中英文混合估算方法：
            - 中文字符：约 1.5 token/字
            - 英文字符：约 0.25 token/字母
            - 标点和数字：约 0.5 token/字符

        Args:
            text: 输入文本。

        Returns:
            估算的 Token 数。
        """
        if not text:
            return 0

        token_count = 0
        for char in text:
            if '一' <= char <= '鿿' or '㐀' <= char <= '䶿':
                token_count += 2  # 中文字符约 2 token
            elif char.isalpha():
                token_count += 0.25  # 英文字母约 0.25 token
            elif char.isdigit():
                token_count += 0.5  # 数字约 0.5 token
            else:
                token_count += 0.5  # 标点和其他字符

        return max(1, int(token_count))

    def get_stats(self) -> LLMUsageStats:
        """获取当前调用统计。

        Returns:
            LLMUsageStats 统计对象。
        """
        return self.stats

    def reset_stats(self) -> None:
        """重置调用统计。"""
        self.stats = LLMUsageStats()

    # ==================== 流式调用 ====================

    def _chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        top_p: float,
        start_time: float,
    ) -> LLMResponse:
        """流式对话补全。

        在后台消费流式生成器，收集所有文本块后返回完整响应。

        Args:
            messages: API 格式的消息列表。
            temperature: 采样温度。
            max_tokens: 最大生成 Token 数。
            top_p: 核采样参数。
            start_time: 开始时间戳。

        Returns:
            LLMResponse 对象。
        """
        response = LLMResponse(model=self.model, provider=self.provider)
        collected_text = ""

        try:
            if self.provider == LLMProvider.DASHSCOPE:
                for chunk in self._stream_dashscope(
                    messages, temperature, max_tokens, top_p, "", response
                ):
                    collected_text += chunk
            else:
                for chunk in self._stream_openai(
                    messages, temperature, max_tokens, top_p, "", response
                ):
                    collected_text += chunk
        except Exception as e:
            raise LLMClientError(f"流式调用失败: {e}") from e

        response.text = collected_text
        response.latency = time.time() - start_time
        response.cost = self._estimate_cost(response)
        return response

    # ==================== 非流式调用 ====================

    def _chat_non_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        top_p: float,
        start_time: float,
    ) -> LLMResponse:
        """非流式对话补全。

        Args:
            messages: API 格式的消息列表。
            temperature: 采样温度。
            max_tokens: 最大生成 Token 数。
            top_p: 核采样参数。
            start_time: 开始时间戳。

        Returns:
            LLMResponse 对象。
        """
        response = LLMResponse(model=self.model, provider=self.provider)

        for attempt in range(self.max_retries):
            try:
                if self.provider == LLMProvider.DASHSCOPE:
                    result = self._call_dashscope(messages, temperature,
                                                  max_tokens, top_p)
                else:
                    result = self._call_openai(messages, temperature,
                                               max_tokens, top_p)

                response.text = result["text"]
                response.finish_reason = result.get("finish_reason", "stop")
                response.input_tokens = result.get("input_tokens", 0)
                response.output_tokens = result.get("output_tokens", 0)
                response.total_tokens = response.input_tokens + response.output_tokens
                response.raw = result.get("raw")
                break

            except (LLMRateLimitError, LLMServiceError) as e:
                if attempt < self.max_retries - 1:
                    delay = _DEFAULT_RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        "LLM 调用重试 (%d/%d): %s, 等待 %.1fs",
                        attempt + 1, self.max_retries, e, delay,
                    )
                    time.sleep(delay)
                else:
                    raise
            except Exception as e:
                raise LLMClientError(f"LLM 调用失败: {e}") from e

        response.latency = time.time() - start_time
        response.cost = self._estimate_cost(response)

        logger.debug(
            "非流式调用完成 | model=%s | tokens=%d/%d | cost=%.6f | latency=%.2fs",
            self.model, response.input_tokens, response.output_tokens,
            response.cost, response.latency,
        )
        return response

    # ==================== DashScope 原生调用 ====================

    def _call_dashscope(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        top_p: float,
    ) -> Dict[str, Any]:
        """通过 DashScope SDK 调用 Qwen 模型。

        Args:
            messages: 消息列表。
            temperature: 采样温度。
            max_tokens: 最大生成 Token 数。
            top_p: 核采样参数。

        Returns:
            包含 text, finish_reason, input_tokens, output_tokens 的字典。

        Raises:
            LLMAuthenticationError: API Key 无效。
            LLMRateLimitError: 请求限流。
            LLMServiceError: 服务端错误。
        """
        from dashscope import Generation

        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "api_key": self.api_key,
            "result_format": "message",
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        resp = Generation.call(**params)

        if resp.status_code == 401:
            raise LLMAuthenticationError(f"DashScope 认证失败: {resp.message}")
        elif resp.status_code == 429:
            raise LLMRateLimitError(f"DashScope 限流: {resp.message}")
        elif resp.status_code != 200:
            raise LLMServiceError(
                f"DashScope API 错误 (status={resp.status_code}): {resp.message}"
            )

        output = resp.output
        text = ""
        finish_reason = "stop"
        if output and output.get("choices"):
            choice = output["choices"][0]
            text = choice.get("message", {}).get("content", "")
            finish_reason = choice.get("finish_reason", "stop")

        usage = resp.usage or {}
        return {
            "text": text,
            "finish_reason": finish_reason,
            "input_tokens": usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0) or usage.get("completion_tokens", 0),
            "raw": resp,
        }

    def _stream_dashscope(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        top_p: float,
        collected_text: str,
        response: LLMResponse,
    ) -> Generator[str, None, None]:
        """DashScope 流式调用。

        使用 SSE (Server-Sent Events) 流式接收生成结果。

        Args:
            messages: 消息列表。
            temperature: 采样温度。
            max_tokens: 最大生成 Token 数。
            top_p: 核采样参数。
            collected_text: 已收集的文本（初始为空）。
            response: 响应对象（用于填充统计）。

        Yields:
            每个文本块。
        """
        from dashscope import Generation

        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "api_key": self.api_key,
            "stream": True,
            "result_format": "message",
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        full_text = collected_text
        input_tokens = 0
        output_tokens = 0

        try:
            for chunk in Generation.call(**params):
                if chunk.status_code == 401:
                    raise LLMAuthenticationError(f"DashScope 认证失败: {chunk.message}")
                elif chunk.status_code == 429:
                    raise LLMRateLimitError(f"DashScope 限流: {chunk.message}")
                elif chunk.status_code != 200:
                    raise LLMServiceError(
                        f"DashScope API 错误 (status={chunk.status_code}): {chunk.message}"
                    )

                output = chunk.output
                if output:
                    choices = output.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full_text += content
                            yield content

                        finish_reason = choices[0].get("finish_reason", "")
                        if finish_reason:
                            response.finish_reason = finish_reason

                usage = chunk.usage
                if usage:
                    input_tokens = usage.get("input_tokens", input_tokens) or usage.get("prompt_tokens", input_tokens)
                    output_tokens = usage.get("output_tokens", output_tokens) or usage.get("completion_tokens", output_tokens)

        except GeneratorExit:
            pass
        finally:
            response.text = full_text
            response.input_tokens = input_tokens
            response.output_tokens = output_tokens
            response.total_tokens = input_tokens + output_tokens

    # ==================== OpenAI 兼容调用 ====================

    def _call_openai(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        top_p: float,
    ) -> Dict[str, Any]:
        """通过 OpenAI SDK 调用模型（含 DashScope 兼容模式）。

        Args:
            messages: 消息列表。
            temperature: 采样温度。
            max_tokens: 最大生成 Token 数。
            top_p: 核采样参数。

        Returns:
            包含 text, finish_reason, input_tokens, output_tokens 的字典。

        Raises:
            LLMAuthenticationError: API Key 无效。
            LLMRateLimitError: 请求限流。
            LLMServiceError: 服务端错误。
        """
        from openai import OpenAI, AuthenticationError, RateLimitError, APIError

        client_kwargs: Dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout,
        }
        if self.api_base:
            client_kwargs["base_url"] = self.api_base

        client = OpenAI(**client_kwargs)

        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        try:
            resp = client.chat.completions.create(**params)
        except AuthenticationError as e:
            raise LLMAuthenticationError(f"OpenAI 认证失败: {e}") from e
        except RateLimitError as e:
            raise LLMRateLimitError(f"OpenAI 限流: {e}") from e
        except APIError as e:
            raise LLMServiceError(f"OpenAI API 错误: {e}") from e

        choice = resp.choices[0] if resp.choices else None
        text = choice.message.content if choice and choice.message else ""
        finish_reason = choice.finish_reason if choice else "stop"

        usage = resp.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        return {
            "text": text or "",
            "finish_reason": finish_reason or "stop",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "raw": resp,
        }

    def _stream_openai(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        top_p: float,
        collected_text: str,
        response: LLMResponse,
    ) -> Generator[str, None, None]:
        """OpenAI 流式调用。

        Args:
            messages: 消息列表。
            temperature: 采样温度。
            max_tokens: 最大生成 Token 数。
            top_p: 核采样参数。
            collected_text: 已收集的文本。
            response: 响应对象。

        Yields:
            每个文本块。
        """
        from openai import OpenAI

        client_kwargs: Dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout,
        }
        if self.api_base:
            client_kwargs["base_url"] = self.api_base

        client = OpenAI(**client_kwargs)

        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        full_text = collected_text
        input_tokens = 0
        output_tokens = 0

        try:
            stream = client.chat.completions.create(**params)
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_text += content
                    yield content

                if chunk.choices and chunk.choices[0].finish_reason:
                    response.finish_reason = chunk.choices[0].finish_reason

                if hasattr(chunk, 'usage') and chunk.usage:
                    if chunk.usage.prompt_tokens:
                        input_tokens = chunk.usage.prompt_tokens
                    if chunk.usage.completion_tokens:
                        output_tokens = chunk.usage.completion_tokens

        except GeneratorExit:
            pass
        finally:
            response.text = full_text
            response.input_tokens = input_tokens
            response.output_tokens = output_tokens
            response.total_tokens = input_tokens + output_tokens

    # ==================== 内部方法 ====================

    def _resolve_provider(
        self,
        provider: Optional[Union[str, LLMProvider]],
    ) -> LLMProvider:
        """解析 LLM 提供方。

        Args:
            provider: 用户传入的 provider 或 None。

        Returns:
            确定的 LLMProvider 枚举值。

        Raises:
            LLMConfigurationError: 无法确定 provider。
        """
        if provider is not None:
            if isinstance(provider, LLMProvider):
                return provider
            provider_lower = provider.lower().replace("-", "_")
            for p in LLMProvider:
                if p.value == provider_lower:
                    return p
            raise LLMConfigurationError(f"不支持的 LLM provider: {provider}")

        # 自动检测：DashScope SDK 优先
        try:
            import dashscope  # noqa: F401
            # 检查是否有 DashScope API Key
            key = (os.environ.get("DASHSCOPE_API_KEY")
                   or os.environ.get("LLM_API_KEY")
                   or os.environ.get("EMBEDDING_API_KEY"))
            if key:
                return LLMProvider.DASHSCOPE
        except ImportError:
            pass

        # 降级到 OpenAI 兼容模式
        try:
            import openai  # noqa: F401
            key = (os.environ.get("LLM_API_KEY")
                   or os.environ.get("OPENAI_API_KEY")
                   or os.environ.get("EMBEDDING_API_KEY"))
            if key:
                # 检查是否有显式的 OpenAI base URL
                base = os.environ.get("LLM_API_BASE") or os.environ.get("OPENAI_API_BASE")
                if base and "openai.com" in base.lower():
                    return LLMProvider.OPENAI
                return LLMProvider.OPENAI_COMPAT
        except ImportError:
            pass

        raise LLMConfigurationError(
            "无法确定 LLM provider。请安装 dashscope 或 openai 库，"
            "并设置对应的 API Key。"
        )

    def _check_dashscope_sdk(self) -> None:
        """检查 DashScope SDK 是否可用。"""
        try:
            import dashscope  # noqa: F401
        except ImportError:
            raise LLMConfigurationError(
                "dashscope 库未安装。请执行: pip install dashscope"
            )

    def _check_openai_sdk(self) -> None:
        """检查 OpenAI SDK 是否可用。"""
        try:
            import openai  # noqa: F401
        except ImportError:
            raise LLMConfigurationError(
                "openai 库未安装。请执行: pip install openai>=1.0.0"
            )

    def _prepare_messages(
        self,
        messages: List[Union[Message, Dict[str, str]]],
        system_prompt: Optional[str],
    ) -> List[Dict[str, str]]:
        """准备 API 调用的消息列表。

        将各种格式的消息统一转换为 API 所需的字典列表格式。

        Args:
            messages: 原始消息列表。
            system_prompt: 可选的系统提示词。

        Returns:
            API 格式的消息字典列表。
        """
        api_messages: List[Dict[str, str]] = []

        # 插入系统提示词
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            if isinstance(msg, dict):
                # 已经是字典格式，确保有 role 和 content
                if "role" not in msg or "content" not in msg:
                    raise LLMClientError(f"消息字典必须包含 role 和 content: {msg}")
                api_messages.append(msg)
            elif isinstance(msg, Message):
                api_messages.append(msg.to_dict())
            else:
                raise LLMClientError(f"不支持的消息类型: {type(msg)}")

        return api_messages

    def _estimate_cost(self, response: LLMResponse) -> float:
        """估算调用成本。

        Args:
            response: LLM 响应对象。

        Returns:
            估算的成本（DashScope 为人民币，OpenAI 为美元）。
        """
        if self.provider == LLMProvider.OPENAI:
            pricing_table = OPENAI_PRICING
        else:
            pricing_table = QWEN_PRICING

        model_pricing = pricing_table.get(self.model, pricing_table["default"])
        input_cost = response.input_tokens * model_pricing["input"] / 1000
        output_cost = response.output_tokens * model_pricing["output"] / 1000
        return round(input_cost + output_cost, 8)

    def _update_global_stats(self, response: Optional[LLMResponse] = None) -> None:
        """更新全局调用统计。

        Args:
            response: 最近一次调用的响应（为 None 时仅增加调用计数）。
        """
        self.stats.total_calls += 1
        self.stats.model_breakdown[self.model] = \
            self.stats.model_breakdown.get(self.model, 0) + 1

        if response:
            self.stats.total_input_tokens += response.input_tokens
            self.stats.total_output_tokens += response.output_tokens
            self.stats.total_cost += response.cost
            self.stats.total_latency += response.latency

    def __repr__(self) -> str:
        return (
            f"LLMClient(model={self.model}, "
            f"provider={self.provider.value}, "
            f"calls={self.stats.total_calls})"
        )
