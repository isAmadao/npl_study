from __future__ import annotations

from rag_mvp.config import settings
from rag_mvp.services.embedding_service import MultimodalEmbeddingService, TextEmbeddingService
from rag_mvp.services.llm_service import LLMService
from rag_mvp.services.vector_store import HybridVectorStore
from rag_mvp.types import ChatReply, SearchResult
from rag_mvp.utils import pad_embedding


class ChatService:
    def __init__(self) -> None:
        self.text_embedder = TextEmbeddingService()
        self.multimodal_embedder = MultimodalEmbeddingService(
            text_fallback=self.text_embedder
        )
        self.vector_store = HybridVectorStore()
        self.llm = LLMService()

    def ask(self, question: str, history: list[dict]) -> ChatReply:
        text_query_embedding = self.text_embedder.encode([question])[0]

        text_results: list[SearchResult] = []
        if self.vector_store.has_chunk_type("text"):
            text_results = self.vector_store.search(
                query_vector=pad_embedding(list(text_query_embedding), settings.target_vector_dim),
                chunk_type="text",
                top_k=settings.query_top_k,
            )
        image_results: list[SearchResult] = []
        if self.vector_store.has_chunk_type("image"):
            image_query_embedding = self.multimodal_embedder.encode_text([question])[0]
            image_results = self.vector_store.search(
                query_vector=pad_embedding(list(image_query_embedding), settings.target_vector_dim),
                chunk_type="image",
                top_k=settings.query_top_k,
            )

        references = self._merge_results(text_results, image_results)
        context = self._build_context(references)

        try:
            answer = self.llm.ask(question=question, context=context, history=history)
        except Exception:
            answer = self.llm.fallback_answer(question=question, references=references)
        return ChatReply(answer=answer, references=references)

    def _merge_results(
        self,
        text_results: list[SearchResult],
        image_results: list[SearchResult],
    ) -> list[SearchResult]:
        merged = {}
        for item in text_results + image_results:
            current = merged.get(item.record_id)
            if current is None or item.score > current.score:
                merged[item.record_id] = item
        return sorted(merged.values(), key=lambda item: item.score, reverse=True)[: settings.query_top_k * 2]

    def _build_context(self, references: list[SearchResult]) -> str:
        if not references:
            return "当前知识库中没有检索到相关内容。"

        lines = []
        for item in references:
            source = f"来源文件：{item.file_name}"
            if item.page_no:
                source += f"，页码：{item.page_no}"
            lines.append(source)
            lines.append(f"内容类型：{item.chunk_type}")
            if item.content:
                lines.append(f"内容：{item.content}")
            if item.image_path:
                lines.append(f"图片路径：{item.image_path}")
            lines.append("")
        return "\n".join(lines).strip()
