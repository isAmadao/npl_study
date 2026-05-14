

import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent
from langchain.tools import tool

# ==================== 配置区 ====================
LLM_CONFIG = {
    "model": "qwen-flash",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key": "sk-4fedee4ece6541d3b17a7173f0b3c16f"
}

# 本地知识库路径（可修改为你的文档目录）
KNOWLEDGE_BASE_PATH = "./knowledge_base"
VECTOR_STORE_PATH = "./vector_store"

# ==================== 1. 文档加载 ====================

def load_documents(directory_path: str):
    """加载本地文档目录中的所有文本文件"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"已创建知识库目录: {directory_path}")
        print("请将文档放入该目录后重新运行")
        return []

    loader = DirectoryLoader(
        directory_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    print(f"加载了 {len(documents)} 个文档")
    return documents


# ==================== 2. 文档分割 ====================

def split_documents(documents):
    """将文档分割成适合检索的块"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"分割为 {len(chunks)} 个文本块")
    return chunks


# ==================== 3. 向量存储 ====================

def build_vector_store(chunks):
    """构建向量数据库"""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-v3",
        base_url=LLM_CONFIG["base_url"],
        api_key=LLM_CONFIG["api_key"]
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"向量库已保存到: {VECTOR_STORE_PATH}")
    return vector_store


def load_vector_store():
    """加载已存在的向量数据库"""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-v3",
        base_url=LLM_CONFIG["base_url"],
        api_key=LLM_CONFIG["api_key"]
    )
    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store


# ==================== 4. 检索问答链 ====================

def create_qa_chain():
    """创建检索增强生成（RAG）问答链"""
    llm = ChatOpenAI(**LLM_CONFIG)

    # 加载向量库
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # 构建提示模板
    system_prompt = (
        "你是一个专业的文档问答助手。请基于以下提供的上下文内容来回答用户的问题。\n"
        "如果上下文中没有足够的信息来回答问题，请明确告知用户。\n"
        "请用中文回答，并且引用相关的原文内容来支持你的回答。\n\n"
        "上下文内容：\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 创建文档组合链
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    # 创建检索链
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return retrieval_chain


# ==================== 5. Agent方式（参考教程中的Agent模式） ====================

@tool
def query_knowledge_base(query: str) -> str:
    """查询本地知识库，获取与问题相关的文档内容"""
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


def create_qa_agent():
    """使用Agent方式创建问答系统（参考03_Agent基础使用.py）"""
    llm = ChatOpenAI(**LLM_CONFIG)

    agent = create_agent(
        model=llm,
        tools=[query_knowledge_base],
        system_prompt=(
            "你是一个本地知识库问答助手。\n"
            "当用户提问时，使用 query_knowledge_base 工具检索本地文档中的相关内容，"
            "然后基于检索到的内容回答用户的问题。\n"
            "如果检索到的内容不足以回答问题，请如实告知用户。"
        ),
    )
    return agent


# ==================== 6. 主流程 ====================

def initialize_knowledge_base():
    """初始化知识库（加载文档 -> 分割 -> 构建向量库）"""
    print("=" * 50)
    print("初始化本地知识库...")
    print("=" * 50)

    documents = load_documents(KNOWLEDGE_BASE_PATH)
    if not documents:
        return False

    chunks = split_documents(documents)
    build_vector_store(chunks)
    print("知识库初始化完成！\n")
    return True


def main():
    """主程序入口"""
    print("=" * 50)
    print("本地知识库问答系统 (RAG)")
    print("=" * 50)

    # 检查向量库是否存在，不存在则初始化
    if not os.path.exists(VECTOR_STORE_PATH):
        print("未检测到向量库，开始初始化...")
        success = initialize_knowledge_base()
        if not success:
            print("初始化失败，请确保知识库目录中有文档")
            return

    # 创建问答链
    qa_chain = create_qa_chain()

    print("\n输入你的问题（输入 'quit' 退出，输入 'reload' 重新加载知识库）：")
    print("-" * 50)

    while True:
        question = input("\n问题: ").strip()

        if question.lower() == "quit":
            print("再见！")
            break
        elif question.lower() == "reload":
            initialize_knowledge_base()
            qa_chain = create_qa_chain()
            print("知识库已重新加载")
            continue
        elif not question:
            continue

        # 执行检索问答
        result = qa_chain.invoke({"input": question})

        print(f"\n回答: {result['answer']}")
        print(f"\n参考文档: {[doc.metadata.get('source', '未知') for doc in result['context']]}")


if __name__ == "__main__":
    main()
