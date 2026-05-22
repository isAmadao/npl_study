"""Qwen-VL answer generation via OpenAI-compatible API (blocking + streaming)."""
from __future__ import annotations

import base64
import json
from collections.abc import AsyncGenerator

from openai import AsyncOpenAI

from app.core.config import settings

_client: AsyncOpenAI | None = None

_NO_CONTEXT_REPLY = "未在知识库中检索到相关内容，无法回答该问题。"

_SYSTEM_PROMPT = (
    "你是一个专业的知识库问答助手。"
    "请仅根据用户提供的参考材料回答问题，不要凭空捏造信息。"
    "回答结束后，必须在单独一行用以下格式注明信息来源：\n"
    "【来源】文件名 第N页"
)


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.qwen_vl_api_key,
            base_url=settings.qwen_vl_base_url,
        )
    return _client


async def generate_answer(query: str, context: list[dict]) -> str:
    """Blocking call — returns the complete answer string."""
    if not context:
        return _NO_CONTEXT_REPLY

    messages = _build_messages(query, context)
    resp = await _get_client().chat.completions.create(
        model=settings.qwen_vl_model,
        messages=messages,
    )
    return resp.choices[0].message.content


async def stream_answer(query: str, context: list[dict]) -> AsyncGenerator[str, None]:
    """
    SSE generator — yields lines in the format:
        data: {"text": "..."}\n\n
        data: [DONE]\n\n
    """
    if not context:
        yield f"data: {json.dumps({'text': _NO_CONTEXT_REPLY}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        return

    messages = _build_messages(query, context)
    stream = await _get_client().chat.completions.create(
        model=settings.qwen_vl_model,
        messages=messages,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield f"data: {json.dumps({'text': delta}, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"


# ── message builder ───────────────────────────────────────────────────────────

def _build_messages(query: str, context: list[dict]) -> list[dict]:
    content: list[dict] = []

    text_chunks = [c for c in context if c["chunk_type"] == "text" and c.get("content")]
    if text_chunks:
        refs = "\n\n".join(
            f"[{c['filename']} 第{c['page_num']}页]\n{c['content']}"
            for c in text_chunks
        )
        content.append({"type": "text", "text": f"参考文本：\n{refs}\n\n"})

    for c in context:
        if c["chunk_type"] != "image" or not c.get("image_path"):
            continue
        try:
            with open(c["image_path"], "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            ext = c["image_path"].rsplit(".", 1)[-1].lower()
            mime = "image/png" if ext == "png" else "image/jpeg"
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
            content.append({
                "type": "text",
                "text": f"[图片来源: {c['filename']} 第{c['page_num']}页]\n",
            })
        except OSError:
            pass  # image file missing — skip silently

    content.append({
        "type": "text",
        "text": f"问题：{query}",
    })

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
