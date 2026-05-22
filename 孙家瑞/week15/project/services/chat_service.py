"""问答编排服务 —— 检索相关上下文 + 调用 LLM 生成答案。"""

from pymilvus import MilvusClient as _MilvusClient

from ..core.model_loader import get_chat_model
from .retriever_service import retrieve

RAG_PROMPT = """根据给定资料回答用户问题。回答要客观、有逻辑，只能基于提供的资料。
如果资料中包含图片链接，保留原始链接并放在合适的内容位置。

用户问题: {question}

相关资料:
{context}"""


def chat(question: str, milvus_client: _MilvusClient) -> dict:
    """执行一次 RAG 问答，返回答案和来源上下文。"""
    context = retrieve(question, milvus_client)

    model = get_chat_model("qwen")
    answer = model.chat(
        system="你是一个基于文档知识库的AI助手，根据提供的资料准确回答问题。",
        user=RAG_PROMPT.format(question=question, context=context),
    )

    return {"question": question, "answer": answer, "context": context}
