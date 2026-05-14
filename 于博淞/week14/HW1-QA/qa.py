"""
在线问答：加载 FAISS 索引 -> 检索相关 chunk -> LLM 生成回答

流程：
  用户问题
    ↓ 向量化
  FAISS 检索 top-k chunk
    ↓ 拼接成 context
  ChatPromptTemplate 构造 prompt
    ↓
  ChatOpenAI 生成回答
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ---- 配置 ---------------------------------------------------------------
INDEX_DIR = "./faiss_index"
TOP_K = 3                          # 检索返回最相似的 chunk 数量

EMBEDDING_MODEL = "text-embedding-v3"
LLM_MODEL = "qwen3.6-plus"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "sk-02e847ab13a543798c4860e15d459293"
# -------------------------------------------------------------------------


def load_chain():
    # 1. 加载本地向量索引
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=BASE_URL,
        api_key=API_KEY,
        check_embedding_ctx_length=False,  # dashscope 只接受原始字符串，禁用 tokenize
    )
    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,  # 加载本地自己构建的索引，安全
    )

    # 2. 转成 retriever，每次检索返回 top-k 个 chunk
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # 3. Prompt 模板：把检索到的 context 和用户问题一起传给 LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是公司内部知识库助手，请严格根据下面提供的上下文内容回答用户问题。
如果上下文中没有相关信息，请直接回答"抱歉，知识库中没有相关内容"，不要编造答案。

上下文：
{context}"""),
        ("human", "{question}"),
    ])

    # 4. LLM
    llm = ChatOpenAI(
        model=LLM_MODEL,
        base_url=BASE_URL,
        api_key=API_KEY,
    )

    # 5. 把多个 chunk 的文本拼接成一段 context
    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    # 6. LCEL 管道：retriever → prompt → llm → 解析纯文本
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def ask(chain, question: str):
    print(f"\n问：{question}")
    print("答：", end="", flush=True)
    # 流式输出，逐 token 打印
    for chunk in chain.stream(question):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    chain = load_chain()

    # 示例问题
    questions = [
        "员工迟到了怎么处理？",
        "出差住宿费用怎么报销？",
        "年假有多少天？",
        "周末加班怎么计算工资？",
    ]
    for q in questions:
        ask(chain, q)
