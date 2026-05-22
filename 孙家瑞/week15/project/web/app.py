"""Streamlit 可选前端 —— 调用 FastAPI 后端服务。"""

import re

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"


def clear_chat():
    st.session_state.messages = [
        {"role": "system", "content": "你好，我是多模态RAG助手，可以对上传的PDF文档进行图文问答。"}
    ]


if "messages" not in st.session_state:
    clear_chat()


def render_answer(markdown_text: str):
    """渲染含图片的 Markdown 文本。"""
    pattern = re.compile(r"!\[.*?\]\((.*?)\)")
    last = 0
    for m in pattern.finditer(markdown_text):
        st.markdown(markdown_text[last : m.start()])
        st.image(m.group(1))
        last = m.end()
    st.markdown(markdown_text[last:])


# --- 页面导航 ---
st.set_page_config(page_title="多模态 RAG Chatbot", layout="wide")
st.title("多模态 RAG Chatbot")

tab1, tab2 = st.tabs(["图文对话", "文件管理"])

# === 图文对话 ===
with tab1:
    with st.sidebar:
        st.button("清空对话", on_click=clear_chat, use_container_width=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("输入你的问题"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            try:
                resp = requests.post(f"{API_BASE}/chat", json={"question": prompt}, timeout=120)
                data = resp.json()
                answer = data.get("answer", "获取答案失败")
                render_answer(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                err = f"后端服务不可用: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

# === 文件管理 ===
with tab2:
    st.subheader("文件列表")

    try:
        docs_resp = requests.get(f"{API_BASE}/documents", timeout=10)
        if docs_resp.ok:
            docs = docs_resp.json()
            if docs:
                for doc in docs:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    col1.write(doc["original_name"])
                    col2.caption(doc["filestate"])
                    if col3.button("删除", key=f"del_{doc['id']}"):
                        requests.delete(f"{API_BASE}/documents/{doc['id']}")
                        st.rerun()
            else:
                st.info("暂无文件，请上传一个 PDF 开始使用")
        else:
            st.warning("无法连接后端，请先启动 FastAPI 服务")
    except Exception:
        st.warning("后端服务未启动，请运行: uvicorn project.main:app --host 0.0.0.0 --port 8000")

    st.subheader("上传文件")
    uploaded = st.file_uploader("选择 PDF 文件", type=["pdf"], key="uploader")
    if uploaded:
        with st.spinner("上传中..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/upload/document",
                    files={"file": (uploaded.name, uploaded.getvalue())},
                    timeout=30,
                )
                if resp.ok:
                    st.success(f"{uploaded.name} 上传成功，后台正在解析...")
                else:
                    st.error(f"上传失败: {resp.text}")
            except Exception as e:
                st.error(f"后端不可用: {e}")
