import os, sys, re
import glob
import requests
from typing import List, Dict, Optional
import numpy as np
from sklearn.preprocessing import normalize
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI


def split_markdown_by_headers_simple(markdown_text: str, path: str, max_length: Optional[int] = 1024) -> List[Dict]:
    """
    按 Markdown 标题分割文本的简化版本

    Args:
        markdown_text: Markdown 文本内容
        max_length: 每个块的最大长度（字符数），如果为None则不限制

    Returns:
        分割后的块列表
    """
    # 先按标题分割
    header_pattern = r'(^#+\s+.+$)'
    lines = markdown_text.split('\n')
    chunks = []
    current_chunk = []
    current_header = "Document"

    for i, line in enumerate(lines):
        if re.match(header_pattern, line.strip()):
            if current_chunk:
                chunk_text = '\n'.join(current_chunk).strip()
                if chunk_text:
                    chunks.append({
                        'text': chunk_text,
                        'header': current_header,
                        "path": path
                    })
            current_chunk = [line]
            current_header = line.strip()
        else:
            current_chunk.append(line)

    if current_chunk:
        chunk_text = '\n'.join(current_chunk).strip()
        if chunk_text:
            chunks.append({
                'text': chunk_text,
                'header': current_header,
                "path": path
            })

    # 如果设置了最大长度，进一步分割大的块
    if max_length:
        final_chunks = []
        for chunk in chunks:
            text = chunk['text']
            header = chunk['header']

            if len(text) <= max_length:
                final_chunks.append(chunk)
            else:
                # 按最大长度简单分割
                for i in range(0, len(text), max_length):
                    sub_text = text[i:i + max_length]
                    final_chunks.append({
                        'text': sub_text,
                        'header': f"{header} (Part {i // max_length + 1})",
                        "path": path
                    })

    return final_chunks


def get_embeddings(input_texts, model="Qwen/Qwen3-Embedding-0.6B/", base_url="http://localhost:8081"):
    """
    Get embeddings for input texts.

    Args:
        input_texts: List of strings to get embeddings for
        model: Model to use for embeddings
        base_url: Base URL of the embeddings server

    Returns:
        dict: Response JSON from the embeddings API
    """
    url = f"{base_url}/v1/embeddings"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "input": input_texts
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()


def get_reranks(query, documents, model="BAAI/bge-reranker-base/", base_url="http://localhost:8082"):
    """
    Rerank documents based on relevance to a query.

    Args:
        query: Search query string
        documents: List of documents to rerank
        model: Reranker model to use
        base_url: Base URL of the rerank server

    Returns:
        dict: Response JSON from the rerank API
    """
    url = f"{base_url}/v1/rerank"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "query": query,
        "documents": documents
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()


markdown_paths = glob.glob("./documents/**/**.md", recursive=True)
print(markdown_paths)

markdown_chunks = []
for path in markdown_paths:
    content = "\n".join(open(path).readlines())
    markdown_chunks += split_markdown_by_headers_simple(content, path)

print("Total chunks", len(markdown_chunks))

embedding_vector = get_embeddings([x["text"] for x in markdown_chunks])
embedding_vector = np.array([x["embedding"] for x in embedding_vector["data"]])
embedding_vector = normalize(embedding_vector)
print("Total chunks", embedding_vector.shape)

query_text = "什么是VLLM的Memory layout？"
query_vector = get_embeddings(query_text)
query_vector = np.array([x["embedding"] for x in query_vector["data"]])
query_vector = normalize(query_vector)
print("Query chunks", query_vector.shape)

ids = np.dot(query_vector, embedding_vector.T)
idx = np.argsort(ids[0])[::-1][:5]

related_text = ""
for chunk in [markdown_chunks[i] for i in idx]:
    related_text += chunk["text"].replace("![](", "![](" + os.path.dirname(chunk["path"]) + "/")

print(related_text)


messages = [
            {"role": "user", "content": f"已有资料：{related_text}\n 用户提问：{query_text}\n请基于已有资料回答问题，并结合资料中的图片进行图文排版。"}
]

model = ChatOpenAI(
    model="qwen-flash", # 模型的代号
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-9c6195bf91f7435d88ea4b819073c92c"
)
ai_response = model.invoke(messages)
print(ai_response)