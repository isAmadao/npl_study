from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st

from rag_mvp.db import init_db
from rag_mvp.services.chat_service import ChatService
from rag_mvp.services.file_service import FileService


@st.cache_resource
def get_file_service() -> FileService:
    init_db()
    return FileService()


@st.cache_resource
def get_chat_service() -> ChatService:
    init_db()
    return ChatService()


def render_file_management() -> None:
    st.title("文件管理")
    st.caption("支持上传 PDF / DOCX，系统会记录状态并投递异步处理任务。")

    uploader = st.file_uploader(
        "选择文件",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )
    file_service = get_file_service()

    if st.button("上传并创建处理任务", type="primary"):
        if not uploader:
            st.warning("请先选择至少一个 PDF 或 DOCX 文件。")
        else:
            for uploaded_file in uploader:
                try:
                    record, dispatch = file_service.save_uploaded_file(uploaded_file)
                    st.success(
                        f"{record.original_name} 上传成功，当前状态：{record.status}，"
                        f"任务投递方式：{dispatch.backend}"
                    )
                    if dispatch.message:
                        st.info(dispatch.message)
                except Exception as exc:
                    st.error(f"{uploaded_file.name} 上传失败：{exc}")

    st.divider()
    st.subheader("已上传文件")

    records = file_service.list_files()
    if not records:
        st.info("当前还没有上传的文件。")
        return

    header_cols = st.columns([3, 1.2, 1.5, 2, 1])
    header_cols[0].markdown("**文件名**")
    header_cols[1].markdown("**类型**")
    header_cols[2].markdown("**状态**")
    header_cols[3].markdown("**创建时间**")
    header_cols[4].markdown("**操作**")

    for record in records:
        cols = st.columns([3, 1.2, 1.5, 2, 1])
        cols[0].write(record.original_name)
        cols[1].write(record.document_type)
        cols[2].write(record.status)
        cols[3].write(record.created_at.strftime("%Y-%m-%d %H:%M:%S"))
        if cols[4].button("删除", key=f"delete-{record.id}"):
            try:
                file_service.delete_file(record.id)
                st.success(f"已删除：{record.original_name}")
                st.rerun()
            except Exception as exc:
                st.error(f"删除失败：{exc}")
        if record.status_detail:
            st.caption(f"{record.original_name}：{record.status_detail}")


def render_chat() -> None:
    st.title("图文对话")
    st.caption("提问后系统会先检索知识库，再将上下文传给大模型生成回答。")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "你好，我已经准备好从知识库中帮你检索并回答问题。",
            }
        ]

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            references = message.get("references")
            if references:
                with st.expander("参考片段"):
                    for ref in references:
                        source = f"{ref.file_name} / {ref.chunk_type}"
                        if ref.page_no:
                            source += f" / page {ref.page_no}"
                        st.markdown(f"- `{source}` 分数={ref.score:.4f}")
                        if ref.content:
                            st.write(ref.content)
                        if ref.image_path:
                            st.write(ref.image_path)

    prompt = st.chat_input("请输入问题")
    if not prompt:
        return

    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("正在检索并组织答案..."):
            reply = get_chat_service().ask(
                question=prompt,
                history=st.session_state.chat_messages[:-1],
            )
            st.markdown(reply.answer)
            if reply.references:
                with st.expander("参考片段"):
                    for ref in reply.references:
                        source = f"{ref.file_name} / {ref.chunk_type}"
                        if ref.page_no:
                            source += f" / page {ref.page_no}"
                        st.markdown(f"- `{source}` 分数={ref.score:.4f}")
                        if ref.content:
                            st.write(ref.content)
                        if ref.image_path:
                            st.write(ref.image_path)

    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": reply.answer,
            "references": reply.references,
        }
    )


def main() -> None:
    st.set_page_config(page_title="图文知识库 MVP", page_icon="📚", layout="wide")
    init_db()

    menu = st.sidebar.radio("菜单", ["文件管理", "图文对话"])
    if menu == "文件管理":
        render_file_management()
    else:
        render_chat()


if __name__ == "__main__":
    main()

