"""
作业1：基于LangChain的本地知识库问答系统
使用 LangGraph StateGraph 模式实现 文档检索 + LLM 回答流程

参考: week14/code/02_langgraph教程/01_graph_api基础.py

运行方式: python 作业1_知识库问答.py
"""

import os
from pathlib import Path
from langchain.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.messages import AnyMessage, ToolMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from typing import Literal
import operator

import config

# ========== 配置 ==========
BASE_URL = config.OPENAI_BASE_URL
API_KEY = config.OPENAI_API_KEY
MODEL_NAME = config.DEFAULT_MODEL
EMBEDDING_MODEL = "text-embedding-v3"

KB_DIR = Path(__file__).parent / "knowledge_base"
VECTOR_STORE_DIR = Path(__file__).parent / "vector_store"


def build_knowledge_base():
    """加载知识库文档，切分后构建 FAISS 向量存储"""
    loader = DirectoryLoader(
        str(KB_DIR),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    print(f"  加载了 {len(documents)} 个文档")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"  切分为 {len(chunks)} 个文本块")

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL, base_url=BASE_URL, api_key=API_KEY
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(VECTOR_STORE_DIR))
    print(f"  向量存储已保存到 {VECTOR_STORE_DIR}")
    return vector_store


def load_or_build_kb():
    """加载已有向量存储，不存在则重建"""
    if (VECTOR_STORE_DIR / "index.faiss").exists() and (
        VECTOR_STORE_DIR / "index.pkl"
    ).exists():
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL, base_url=BASE_URL, api_key=API_KEY
        )
        vector_store = FAISS.load_local(
            str(VECTOR_STORE_DIR), embeddings, allow_dangerous_deserialization=True
        )
        print("  从本地加载了已有向量存储")
        return vector_store
    else:
        return build_knowledge_base()


# 全局向量存储（延迟加载）
_vector_store = None


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = load_or_build_kb()
    return _vector_store


# ========== 检索工具 ==========
@tool
def search_knowledge_base(query: str) -> str:
    """搜索本地知识库，获取与查询最相关的文档内容。
    在回答任何需要知识库信息的问题之前，必须先调用此工具。

    Args:
        query: 搜索关键词或问题
    """
    vs = get_vector_store()
    docs = vs.similarity_search(query, k=3)

    results = []
    for i, doc in enumerate(docs):
        src = Path(doc.metadata.get("source", "未知")).name
        results.append(f"[文档{i+1} 来源: {src}]\n{doc.page_content}")

    return "\n\n---\n\n".join(results)


# ========== LangGraph Agent ==========
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


def create_agent():
    model = ChatOpenAI(model=MODEL_NAME, base_url=BASE_URL, api_key=API_KEY)
    tools = [search_knowledge_base]
    tools_by_name = {t.name: t for t in tools}
    model_with_tools = model.bind_tools(tools)

    def llm_call(state: dict):
        return {
            "messages": [
                model_with_tools.invoke(
                    [
                        SystemMessage(
                            content=(
                                "你是一个知识库问答助手。你的任务是基于本地知识库的内容回答用户问题。"
                                "规则：\n"
                                "1. 必须先调用 search_knowledge_base 工具检索相关文档\n"
                                "2. 基于检索到的文档内容回答，不要编造信息\n"
                                "3. 如果知识库中没有相关信息，请如实告知用户\n"
                                "4. 回答时注明信息来源"
                            )
                        )
                    ]
                    + state["messages"]
                )
            ],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def tool_node(state: dict):
        last_msg = state["messages"][-1]
        result = []
        for tc in last_msg.tool_calls:
            tool = tools_by_name[tc["name"]]
            observation = tool.invoke(tc["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tc["id"]))
        return {"messages": result}

    def should_continue(state: MessagesState) -> Literal["tool_node", END]:
        last_msg = state["messages"][-1]
        if last_msg.tool_calls:
            return "tool_node"
        return END

    builder = StateGraph(MessagesState)
    builder.add_node("llm_call", llm_call)
    builder.add_node("tool_node", tool_node)
    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    builder.add_edge("tool_node", "llm_call")

    return builder.compile()


# ========== 交互式问答 ==========
def main():
    print("=" * 50)
    print("  本地知识库问答系统 (LangChain + LangGraph)")
    print("  知识库目录:", KB_DIR)
    print("=" * 50)
    print("  正在初始化...")

    agent = create_agent()
    get_vector_store()  # 预加载向量存储

    print("\n  知识库就绪！输入问题开始对话，输入 quit 退出\n")

    while True:
        user_input = input("  用户: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("  再见！")
            break
        if not user_input:
            continue

        result = agent.invoke({"messages": [HumanMessage(content=user_input)]})
        last_msg = result["messages"][-1]
        print(f"\n  助手: {last_msg.content}")
        print(f"  [LLM调用: {result.get('llm_calls', 0)}次]")


if __name__ == "__main__":
    main()
