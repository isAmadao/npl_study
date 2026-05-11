"""
作业1：基于讲解的Langchain框架，开发对本地知识库进行问答的逻辑，只需要包括文档检索+llm问答流程
"""

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 初始化模型和 Embedding
llm = ChatOpenAI(
    model="qwen-flash",  # 模型的代号
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key="API_KEY"
)

# 使用免费的本地 Embedding 模型（首次运行会自动下载）
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5"
)

# --- 1. 准备知识文档 ---
# 实际项目中，这些文档可以从文件、网页、数据库等加载
documents = [
    "LangChain 是一个用于构建大语言模型应用的开源框架，由 Harrison Chase 于 2022 年创建。",
    "LangChain 的核心组件包括：模型接口、提示词模板、链、记忆、检索和代理。",
    "LCEL（LangChain Expression Language）是 LangChain 的新一代链构建语法，使用管道符 | 连接各组件。",
    "RAG（检索增强生成）通过在生成前检索相关文档，让 LLM 能回答训练数据之外的问题。",
    "LangGraph 是 LangChain 团队推出的新框架，专门用于构建复杂的多步骤 AI 代理工作流。",
    "LangSmith 是 LangChain 的可观测性平台，用于调试、测试和监控 LLM 应用。",
]

# --- 2. 文本切片 ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)
# 这里文档已经较短，实际场景中长文档会被切成多个片段
texts = text_splitter.create_documents(documents)

# --- 3. 向量化并存入 FAISS ---
vectorstore = FAISS.from_documents(texts, embeddings)

# --- 4. 创建检索器 ---
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# --- 5. 构建 RAG 链 ---
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个知识助手。根据以下检索到的上下文来回答问题。如果上下文中没有答案，就说你不知道。\n\n上下文：{context}"),
    ("human", "{question}")
])

# 辅助函数：将检索到的文档拼接为字符串
def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# --- 6. 测试 RAG ---
question = "LangChain 的核心组件有哪些？"
print(f"问题：{question}")
print(f"回答：{rag_chain.invoke(question)}")