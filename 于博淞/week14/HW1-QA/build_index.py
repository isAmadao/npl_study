"""
离线预处理：加载本地文档 -> 切块 -> 向量化 -> 保存到 FAISS 索引
运行一次即可，后续 qa.py 直接加载索引
"""
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ---- 配置 ---------------------------------------------------------------
DOCS_DIR = "./docs"           # 本地知识库文件夹
INDEX_DIR = "./faiss_index"   # 向量索引保存目录
CHUNK_SIZE = 500              # 每个 chunk 的字符数
CHUNK_OVERLAP = 50            # chunk 之间的重叠字符数（保留上下文衔接）

EMBEDDING_MODEL = "text-embedding-v3"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "sk-02e847ab13a543798c4860e15d459293"
# -------------------------------------------------------------------------


def build_index():
    # 1. 加载 docs/ 下所有 .txt 文件
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    print(f"加载文档数: {len(docs)}")

    # 2. 切块
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"切块数: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"  chunk[{i}]: {chunk.page_content[:50].replace(chr(10), ' ')}...")

    # 3. 向量化并存入 FAISS
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=BASE_URL,
        api_key=API_KEY,
        check_embedding_ctx_length=False,
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 4. 保存索引到本地
    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(INDEX_DIR)
    print(f"索引已保存到: {INDEX_DIR}")


if __name__ == "__main__":
    build_index()
