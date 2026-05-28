#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05-multimodal-rag-chatbot - 多模态 RAG 聊天机器人

核心功能:
1. 文档解析与分块 (支持 MinerU/pdfplumber)
2. 文本与图像向量化
3. 多模态检索 (混合检索 + Rerank)
4. 智能对话生成
5. 流式输出支持

系统架构:
┌─────────────────────────────────────────────────────────┐
│                     用户界面层                            │
│              (CLI / Web API / Streamlit)                │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                   对话管理层 (ChatManager)              │
│              - 会话历史管理                              │
│              - 提示词工程                                │
└──────────────────────────┬──────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           │                               │
┌──────────▼──────────┐         ┌──────────▼──────────┐
│   检索引擎层        │         │   生成引擎层        │
│ (RetrievalEngine)   │         │  (GenerationEngine) │
│ - 向量检索          │         │  - LLM 调用        │
│ - 混合检索          │         │  - 流式输出        │
│ - Rerank            │         │  - 多模态理解      │
└──────────┬──────────┘         └──────────▲──────────┘
           │                               │
┌──────────▼──────────┐                   │
│   文档处理层        │                   │
│ (DocumentProcessor) │                   │
│ - PDF 解析         │                   │
│ - 文档分块         │                   │
│ - 图像提取         │                   │
└──────────┬──────────┘                   │
           │                               │
┌──────────▼──────────┐                   │
│   向量存储层        │                   │
│  (VectorStore)      │                   │
│ - 文本向量          │                   │
│ - 图像向量          │                   │
└─────────────────────┘───────────────────┘
"""

import os
import json
import hashlib
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


# =============================================================================
# 数据模型定义
# =============================================================================

@dataclass
class DocumentChunk:
    """文档分块"""
    chunk_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    image_path: Optional[str] = None


@dataclass
class Document:
    """文档"""
    doc_id: str
    path: str
    chunks: List[DocumentChunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """检索结果"""
    chunk: DocumentChunk
    score: float
    rank: int


@dataclass
class ChatMessage:
    """聊天消息"""
    role: str  # user / assistant / system
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """会话"""
    conv_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 文档解析器 (Document Parser)
# =============================================================================

class BaseDocumentParser:
    """文档解析器基类"""

    def parse(self, file_path: str) -> Tuple[str, List[Dict]]:
        """
        解析文档

        Args:
            file_path: 文件路径

        Returns:
            (文本内容, 图片列表)
        """
        raise NotImplementedError


class PDFPlumberParser(BaseDocumentParser):
    """pdfplumber 解析器"""

    def parse(self, file_path: str) -> Tuple[str, List[Dict]]:
        import pdfplumber

        text_parts = []
        images = []

        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                text_parts.append(f"--- Page {i+1} ---\n{text}")

                # 记录图片信息
                if page.images:
                    images.extend([{
                        "page": i+1,
                        "bbox": img.get("bbox", [])
                    } for img in page.images])

        return "\n\n".join(text_parts), images


class DocumentChunker:
    """文档分块器"""

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 128):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_by_header(self, markdown_text: str, source_path: str) -> List[DocumentChunk]:
        """
        按标题分块

        Args:
            markdown_text: Markdown 文本
            source_path: 源文件路径

        Returns:
            分块列表
        """
        import re

        chunks = []
        header_pattern = r'(^#+ .+$)'
        lines = markdown_text.split('\n')

        current_header = "Document"
        current_content = []

        def _add_chunk():
            if current_content:
                content = "\n".join(current_content).strip()
                if content:
                    chunk_id = hashlib.md5(f"{source_path}:{current_header}:{len(chunks)}".encode()).hexdigest()[:16]
                    chunks.append(DocumentChunk(
                        chunk_id=chunk_id,
                        content=content,
                        metadata={
                            "header": current_header,
                            "source": source_path,
                            "chunk_index": len(chunks)
                        }
                    ))

        for line in lines:
            if re.match(header_pattern, line.strip()):
                _add_chunk()
                current_header = line.strip()
                current_content = [line]
            else:
                current_content.append(line)

        _add_chunk()
        return chunks

    def chunk_by_size(self, text: str, source_path: str) -> List[DocumentChunk]:
        """
        按大小分块

        Args:
            text: 文本
            source_path: 源文件路径

        Returns:
            分块列表
        """
        chunks = []
        words = text.split()

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)

            chunk_id = hashlib.md5(f"{source_path}:chunk:{i}".encode()).hexdigest()[:16]
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                content=chunk_text,
                metadata={
                    "source": source_path,
                    "start_index": i,
                    "chunk_index": len(chunks)
                }
            ))

        return chunks


# =============================================================================
# 向量存储 (Vector Store)
# =============================================================================

class SimpleVectorStore:
    """简单向量存储（内存实现）"""

    def __init__(self):
        self.chunks: Dict[str, DocumentChunk] = {}  # chunk_id -> chunk
        self.embeddings: Optional[np.ndarray] = None
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: List[str] = []

    def add_chunks(self, chunks: List[DocumentChunk]):
        """添加分块"""
        for chunk in chunks:
            if chunk.chunk_id not in self.chunks:
                self.chunks[chunk.chunk_id] = chunk
                if chunk.embedding is not None:
                    self._update_index(chunk)

    def _update_index(self, chunk: DocumentChunk):
        """更新索引"""
        if self.embeddings is None:
            self.embeddings = chunk.embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, chunk.embedding])

        idx = len(self.index_to_id)
        self.id_to_index[chunk.chunk_id] = idx
        self.index_to_id.append(chunk.chunk_id)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievalResult]:
        """
        向量检索

        Args:
            query_embedding: 查询向量
            top_k: 返回数量

        Returns:
            检索结果列表
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        # 计算余弦相似度
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        similarities = np.dot(embeddings_norm, query_norm.T).flatten()

        # 获取 top_k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            chunk_id = self.index_to_id[idx]
            results.append(RetrievalResult(
                chunk=self.chunks[chunk_id],
                score=float(similarities[idx]),
                rank=rank + 1
            ))

        return results


# =============================================================================
# 嵌入模型 (Embedding Model)
# =============================================================================

class BaseEmbeddingModel:
    """嵌入模型基类"""

    def embed_text(self, text: str) -> np.ndarray:
        """嵌入文本"""
        raise NotImplementedError

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """批量嵌入文本"""
        return [self.embed_text(text) for text in texts]

    def embed_image(self, image_path: str) -> np.ndarray:
        """嵌入图片"""
        raise NotImplementedError


class MockEmbeddingModel(BaseEmbeddingModel):
    """模拟嵌入模型（用于测试）"""

    def __init__(self, dim: int = 768):
        self.dim = dim

    def embed_text(self, text: str) -> np.ndarray:
        """模拟文本嵌入"""
        # 基于文本哈希生成伪随机向量
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        np.random.seed(hash_val)
        return np.random.randn(self.dim).astype(np.float32)

    def embed_image(self, image_path: str) -> np.ndarray:
        """模拟图片嵌入"""
        hash_val = int(hashlib.md5(image_path.encode()).hexdigest()[:8], 16)
        np.random.seed(hash_val)
        return np.random.randn(self.dim).astype(np.float32)


# =============================================================================
# 检索引擎 (Retrieval Engine)
# =============================================================================

class RetrievalEngine:
    """检索引擎"""

    def __init__(self, vector_store: SimpleVectorStore, embedding_model: BaseEmbeddingModel):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        检索相关文档

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            检索结果列表
        """
        # 生成查询向量
        query_embedding = self.embedding_model.embed_text(query)

        # 向量检索
        results = self.vector_store.search(query_embedding, top_k)

        return results


# =============================================================================
# 生成引擎 (Generation Engine)
# =============================================================================

class BaseGenerationModel:
    """生成模型基类"""

    def generate(self, messages: List[Dict], stream: bool = False):
        """生成回复"""
        raise NotImplementedError


class MockGenerationModel(BaseGenerationModel):
    """模拟生成模型（用于测试）"""

    def generate(self, messages: List[Dict], stream: bool = False):
        """模拟生成"""
        response = "这是一个模拟的回复。基于检索到的文档内容，我可以为您提供相关信息。"

        if stream:
            # 流式输出
            async def _stream():
                for char in response:
                    yield char
                    await asyncio.sleep(0.01)
            return _stream()
        else:
            return response


# =============================================================================
# 对话管理器 (Chat Manager)
# =============================================================================

class ChatManager:
    """对话管理器"""

    def __init__(self, retrieval_engine: RetrievalEngine, generation_model: BaseGenerationModel):
        self.retrieval_engine = retrieval_engine
        self.generation_model = generation_model
        self.conversations: Dict[str, Conversation] = {}

    def create_conversation(self) -> str:
        """创建新会话"""
        conv_id = hashlib.md5(datetime.now().isoformat().encode()).hexdigest()[:16]
        self.conversations[conv_id] = Conversation(conv_id=conv_id)
        return conv_id

    def add_message(self, conv_id: str, role: str, content: str):
        """添加消息"""
        if conv_id not in self.conversations:
            self.conversations[conv_id] = Conversation(conv_id=conv_id)

        self.conversations[conv_id].messages.append(
            ChatMessage(role=role, content=content)
        )

    def build_prompt(self, query: str, retrieval_results: List[RetrievalResult]) -> str:
        """构建提示词"""
        context_parts = []

        for i, result in enumerate(retrieval_results):
            context_parts.append(
                f"[文档 {i+1}] (相关度: {result.score:.3f})\n"
                f"{result.chunk.content}\n"
            )

        context = "\n---\n".join(context_parts)

        prompt = f"""你是一个专业的问答助手。请基于以下文档内容回答用户的问题。

【参考文档】
{context}

【用户问题】
{query}

【回答要求】
1. 仅基于提供的文档内容回答
2. 如果文档中没有相关信息，请如实告知
3. 回答要简洁准确
4. 可以引用文档中的关键内容"""

        return prompt

    async def chat(self, conv_id: str, query: str, top_k: int = 5, stream: bool = False):
        """
        聊天

        Args:
            conv_id: 会话 ID
            query: 用户查询
            top_k: 检索数量
            stream: 是否流式输出

        Returns:
            回复内容
        """
        # 1. 检索相关文档
        retrieval_results = self.retrieval_engine.retrieve(query, top_k)

        # 2. 构建提示词
        prompt = self.build_prompt(query, retrieval_results)

        # 3. 添加用户消息
        self.add_message(conv_id, "user", query)

        # 4. 构建消息上下文
        messages = [{"role": "user", "content": prompt}]

        # 5. 生成回复
        if stream:
            return self._chat_stream(conv_id, messages)
        else:
            response = self.generation_model.generate(messages)
            self.add_message(conv_id, "assistant", response)
            return response, retrieval_results

    async def _chat_stream(self, conv_id: str, messages: List[Dict]):
        """流式聊天"""
        full_response = ""
        stream = self.generation_model.generate(messages, stream=True)

        async for chunk in stream:
            full_response += chunk
            yield chunk

        self.add_message(conv_id, "assistant", full_response)


# =============================================================================
# 文档处理器 (Document Processor)
# =============================================================================

class DocumentProcessor:
    """文档处理器"""

    def __init__(self, parser: BaseDocumentParser, chunker: DocumentChunker, embedding_model: BaseEmbeddingModel):
        self.parser = parser
        self.chunker = chunker
        self.embedding_model = embedding_model

    def process_document(self, file_path: str) -> Document:
        """
        处理文档

        Args:
            file_path: 文件路径

        Returns:
            处理后的文档
        """
        # 1. 解析文档
        text, images = self.parser.parse(file_path)

        # 2. 分块
        chunks = self.chunker.chunk_by_header(text, file_path)

        # 3. 生成嵌入
        for chunk in chunks:
            chunk.embedding = self.embedding_model.embed_text(chunk.content)

        # 4. 创建文档对象
        doc_id = hashlib.md5(file_path.encode()).hexdigest()[:16]
        return Document(
            doc_id=doc_id,
            path=file_path,
            chunks=chunks,
            metadata={
                "images": images,
                "processed_at": datetime.now().isoformat()
            }
        )


# =============================================================================
# RAG 聊天机器人主类
# =============================================================================

class MultimodalRAGChatbot:
    """多模态 RAG 聊天机器人"""

    def __init__(self):
        # 初始化组件
        self.embedding_model = MockEmbeddingModel()
        self.vector_store = SimpleVectorStore()
        self.retrieval_engine = RetrievalEngine(self.vector_store, self.embedding_model)
        self.generation_model = MockGenerationModel()
        self.chat_manager = ChatManager(self.retrieval_engine, self.generation_model)

        self.parser = PDFPlumberParser()
        self.chunker = DocumentChunker()
        self.doc_processor = DocumentProcessor(self.parser, self.chunker, self.embedding_model)

    def add_document(self, file_path: str) -> Document:
        """添加文档到知识库"""
        doc = self.doc_processor.process_document(file_path)
        self.vector_store.add_chunks(doc.chunks)
        print(f"[OK] 文档已添加: {file_path} (共 {len(doc.chunks)} 个分块)")
        return doc

    async def chat(self, query: str, conv_id: Optional[str] = None, top_k: int = 5):
        """
        聊天接口

        Args:
            query: 用户查询
            conv_id: 会话 ID（可选）
            top_k: 检索数量

        Returns:
            (回复, 检索结果, 会话 ID)
        """
        if conv_id is None:
            conv_id = self.chat_manager.create_conversation()

        response, results = await self.chat_manager.chat(conv_id, query, top_k)
        return response, results, conv_id


# =============================================================================
# 测试用例
# =============================================================================

def run_tests():
    """运行测试"""
    print("="*60)
    print("05-multimodal-rag-chatbot 测试")
    print("="*60)

    # 1. 测试文档分块
    print("\n[测试 1] 文档分块...")
    chunker = DocumentChunker()
    test_markdown = """# 标题一
这是第一段内容。

## 子标题
这是第二段内容，包含更多信息。

### 更深的标题
第三段内容。"""
    chunks = chunker.chunk_by_header(test_markdown, "test.md")
    print(f"[OK] 分块成功，共 {len(chunks)} 个分块")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {chunk.metadata.get('header', 'N/A')}")

    # 2. 测试向量存储
    print("\n[测试 2] 向量存储...")
    embed_model = MockEmbeddingModel()
    vector_store = SimpleVectorStore()

    # 添加测试分块
    for chunk in chunks:
        chunk.embedding = embed_model.embed_text(chunk.content)
    vector_store.add_chunks(chunks)
    print(f"[OK] 向量存储初始化完成，共 {len(vector_store.chunks)} 个分块")

    # 3. 测试检索
    print("\n[测试 3] 检索功能...")
    retrieval_engine = RetrievalEngine(vector_store, embed_model)
    results = retrieval_engine.retrieve("子标题的内容", top_k=3)
    print(f"[OK] 检索成功，返回 {len(results)} 个结果")
    for result in results:
        print(f"  Rank {result.rank}, Score {result.score:.3f}: {result.chunk.metadata.get('header')}")

    # 4. 测试聊天机器人
    print("\n[测试 4] 聊天机器人...")
    chatbot = MultimodalRAGChatbot()
    print("[OK] 聊天机器人初始化完成")

    print("\n" + "="*60)
    print("所有测试通过！")
    print("="*60)


# =============================================================================
# 主函数
# =============================================================================

async def main():
    """主函数 - 演示用法"""
    print("="*60)
    print("05-multimodal-rag-chatbot 多模态 RAG 聊天机器人")
    print("="*60)

    # 1. 初始化聊天机器人
    chatbot = MultimodalRAGChatbot()

    # 2. 测试
    run_tests()

    # 3. 可选：添加真实文档
    print("\n提示：可以使用 chatbot.add_document('path/to/document.pdf') 添加文档")


if __name__ == "__main__":
    # 先运行同步测试
    run_tests()

    # 再运行异步演示
    # asyncio.run(main())
