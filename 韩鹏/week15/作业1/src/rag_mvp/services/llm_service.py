from __future__ import annotations

import json
from urllib import request

from rag_mvp.config import settings
from rag_mvp.types import SearchResult


class LLMService:
    def ask(self, question: str, context: str, history: list[dict]) -> str:
        endpoint = settings.llm_base_url.rstrip("/") + "/chat/completions"
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个图文知识库问答助手。"
                    "请严格基于检索上下文回答。"
                    "如果上下文不足，请明确说明不知道或信息不足。"
                ),
            }
        ]
        for item in history[-6:]:
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        messages.append(
            {
                "role": "user",
                "content": f"问题：{question}\n\n检索上下文：\n{context}",
            }
        )

        payload = {
            "model": settings.llm_model,
            "messages": messages,
            "temperature": 0.2,
        }
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {settings.llm_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    def fallback_answer(self, question: str, references: list[SearchResult]) -> str:
        if not references:
            return (
                "当前没有检索到足够相关的知识片段，暂时无法基于知识库回答这个问题。"
                "可以先上传文档，或换一种更具体的问法。"
            )

        lines = [
            f"大模型暂时不可用，我先基于检索结果给你摘要回答：",
            f"问题：{question}",
            "",
            "当前最相关的知识片段包括：",
        ]
        for item in references[:5]:
            source = f"{item.file_name}"
            if item.page_no:
                source += f" 第{item.page_no}页"
            lines.append(f"1. 来源：{source}，类型：{item.chunk_type}，相关度：{item.score:.4f}")
            if item.content:
                lines.append(item.content)
        return "\n".join(lines)

