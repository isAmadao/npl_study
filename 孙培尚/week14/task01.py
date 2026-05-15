# main.py

import os
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ========== 配置 ==========
class Config:
    MODEL_NAME = "qwen-flash"
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    API_KEY = "sk-4fedee4ece6541d3b17a7173f0b3c16f"
    KNOWLEDGE_BASE_DIR = "./knowledge_base"
    VECTOR_STORE_DIR = "./vector_store"
    CHUNK_SIZE = 500
    TOP_K_RESULTS = 5


config = Config()


# ========== 请求/响应模型 ==========
class AskRequest(BaseModel):
    question: str
    stream: bool = False


class AskResponse(BaseModel):
    answer: str


class UploadResponse(BaseModel):
    message: str
    chunk_count: int


# ========== 知识库服务 ==========
class KnowledgeBaseService:
    def __init__(self):
        self.vector_store = None
        self.agent = None
        self.llm = None
        self._init_llm()
        self._init_vector_store()
        self._init_agent()

    def _init_llm(self):
        self.llm = ChatOpenAI(
            model=config.MODEL_NAME,
            base_url=config.BASE_URL,
            api_key=config.API_KEY,
            temperature=0.7,
        )

    def _init_vector_store(self):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        if os.path.exists(config.VECTOR_STORE_DIR):
            self.vector_store = FAISS.load_local(
                config.VECTOR_STORE_DIR,
                embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            Path(config.KNOWLEDGE_BASE_DIR).mkdir(exist_ok=True)
            self.vector_store = FAISS.from_documents([], embeddings)

    def _init_agent(self):
        @tool
        def search_knowledge_base(query: str) -> str:
            """在知识库中搜索相关信息"""
            results = self.vector_store.similarity_search(query, k=config.TOP_K_RESULTS)
            if not results:
                return "未找到相关信息"
            return "\n\n---\n\n".join([doc.page_content for doc in results])

        self.agent = create_agent(
            model=self.llm,
            tools=[search_knowledge_base],
            system_prompt="你是知识库助手，基于检索到的内容回答问题。如果找不到相关信息，请告知用户。"
        )

    def add_documents(self, file_paths: List[str]) -> int:
        """添加文档到知识库"""
        documents = []
        for file_path in file_paths:
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(documents)

        self.vector_store.add_documents(chunks)
        self.vector_store.save_local(config.VECTOR_STORE_DIR)

        return len(chunks)

    def ask(self, question: str) -> str:
        """同步问答"""
        result = self.agent.invoke({
            "messages": [{"role": "user", "content": question}]
        })
        return result["messages"][-1].content

    async def ask_stream(self, question: str):
        """流式问答"""
        messages = [{"role": "user", "content": question}]

        for chunk in self.agent.stream(
                {"messages": messages},
                stream_mode="messages",
                version="v2",
        ):
            if hasattr(chunk, 'content') and chunk.content:
                yield chunk.content


# ========== 创建FastAPI应用 ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化知识库服务
    app.state.qa_service = KnowledgeBaseService()
    yield
    # 关闭时清理资源
    pass


app = FastAPI(title="本地知识库问答系统", lifespan=lifespan)


# ========== API接口 ==========
@app.get("/")
async def root():
    return {"message": "本地知识库问答系统", "status": "running"}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """同步问答接口"""
    try:
        answer = app.state.qa_service.ask(request.question)
        return AskResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/stream")
async def ask_stream(request: AskRequest):
    """流式问答接口"""
    try:
        return StreamingResponse(
            app.state.qa_service.ask_stream(request.question),
            media_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_document(file_paths: List[str]):
    """上传文档到知识库"""
    try:
        chunk_count = app.state.qa_service.add_documents(file_paths)
        return UploadResponse(
            message=f"成功添加 {len(file_paths)} 个文档",
            chunk_count=chunk_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}
