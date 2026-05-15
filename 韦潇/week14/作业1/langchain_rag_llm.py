from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os


class LangChainRAG:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 LangChain RAG 问答系统

        Args:
            config: 配置字典，包含以下键：
                - llm_model: LLM 模型名称
                - llm_base_url: LLM API 基础地址
                - llm_api_key: LLM API 密钥
                - embedding_model: 嵌入模型名称或路径
                - chunk_size: 文本分割大小
                - chunk_overlap: 文本分割重叠大小
                - top_k: 检索返回的文档数量
        """
        self.config = config

        # 初始化 LLM
        self.llm = self._init_llm()

        # 初始化嵌入模型
        self.embeddings = self._init_embeddings()

        # 文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.get("chunk_size", 500),
            chunk_overlap=config.get("chunk_overlap", 50),
            length_function=len
        )

        # 向量数据库
        self.vector_store = None

        # 检索问答链
        self.qa_chain = None

        # 构建提示模板
        self.prompt = self._build_prompt()

    def _init_llm(self) -> ChatOpenAI:
        """初始化大语言模型"""
        return ChatOpenAI(
            model=self.config["llm_model"],
            base_url=self.config["llm_base_url"],
            api_key=self.config["llm_api_key"],
            temperature=self.config.get("temperature", 0.1)
        )

    def _init_embeddings(self) -> HuggingFaceEmbeddings:
        """初始化嵌入模型"""
        model_name = self.config.get("embedding_model", "BAAI/bge-small-zh-v1.5")
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": self.config.get("device", "cpu")},
            encode_kwargs={"normalize_embeddings": True}
        )

    def _build_prompt(self) -> PromptTemplate:
        """构建 RAG 问答提示模板"""
        template = """
你是一个专业的问答助手，请根据提供的上下文信息回答用户的问题。

上下文信息：
{context}

问题：{question}

请根据上下文信息准确回答问题。如果上下文信息不足以回答问题，请明确说明"无法从知识库中找到相关信息"。
"""
        return PromptTemplate(
            input_variables=["context", "question"],
            template=template.strip()
        )

    def load_documents(self, directory_path: str) -> List:
        """
        从目录加载文档

        Args:
            directory_path: 文档目录路径

        Returns:
            加载的文档列表
        """
        # 支持的文件类型加载器
        loaders = [
            DirectoryLoader(
                directory_path,
                glob="*.txt",
                loader_cls=TextLoader
            ),
            DirectoryLoader(
                directory_path,
                glob="*.pdf",
                loader_cls=PyPDFLoader
            ),
            DirectoryLoader(
                directory_path,
                glob="*.md",
                loader_cls=TextLoader
            )
        ]

        documents = []
        for loader in loaders:
            try:
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                print(f"加载文件时出错: {e}")

        return documents

    def build_vector_store(self, documents: List):
        """
        构建向量数据库

        Args:
            documents: 文档列表
        """
        # 分割文档
        split_docs = self.text_splitter.split_documents(documents)
        print(f"文档分割完成，共 {len(split_docs)} 个片段")

        # 创建向量数据库
        self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
        print("向量数据库构建完成")

    def save_vector_store(self, save_path: str):
        """
        保存向量数据库到本地

        Args:
            save_path: 保存路径
        """
        if self.vector_store:
            self.vector_store.save_local(save_path)
            print(f"向量数据库已保存到: {save_path}")
        else:
            print("请先构建向量数据库")

    def load_vector_store(self, load_path: str):
        """
        从本地加载向量数据库

        Args:
            load_path: 加载路径
        """
        self.vector_store = FAISS.load_local(load_path, self.embeddings)
        print(f"向量数据库已从 {load_path} 加载")

    def init_qa_chain(self):
        """初始化问答链"""
        if not self.vector_store:
            raise ValueError("请先构建或加载向量数据库")

        # 创建检索器
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.config.get("top_k", 5)}
        )

        # 创建问答链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )
        print("问答链初始化完成")

    def query(self, question: str) -> Dict[str, Any]:
        """
        执行问答查询

        Args:
            question: 用户问题

        Returns:
            包含答案和来源文档的字典
        """
        if not self.qa_chain:
            self.init_qa_chain()

        result = self.qa_chain({"query": question})

        # 整理来源文档信息
        source_docs = []
        if "source_documents" in result:
            for doc in result["source_documents"]:
                source_docs.append({
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", "unknown"),
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })

        return {
            "answer": result["result"],
            "source_documents": source_docs,
            "question": question
        }


# 使用示例
if __name__ == "__main__":
    # 配置参数
    config = {
        "llm_model": "qwen-flash",
        "llm_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "llm_api_key": "your-api-key-here",  # 替换为API Key
        "embedding_model": "BAAI/bge-small-zh-v1.5",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "top_k": 5,
        "temperature": 0.1,
        "device": "cpu"
    }

    # 创建 RAG 实例
    rag = LangChainRAG(config)

    # 示例1: 从目录加载文档并构建向量库
    # documents = rag.load_documents("./knowledge_base")
    # rag.build_vector_store(documents)
    # rag.save_vector_store("./vector_store")

    # 示例2: 加载已有的向量库
    rag.load_vector_store("./vector_store")

    # 初始化问答链
    rag.init_qa_chain()

    # 测试问答
    while True:
        question = input("请输入你的问题（输入 'exit' 退出）：")
        if question.lower() == "exit":
            break

        result = rag.query(question)
        print("\n=== 回答 ===")
        print(result["answer"])

        print("\n=== 来源文档 ===")
        for i, doc in enumerate(result["source_documents"], 1):
            print(f"【{i}】来源: {doc['source']}, 页码: {doc['page']}")
            print(f"内容摘要: {doc['content']}\n")
