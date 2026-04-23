import os
import threading
import traceback
from datetime import datetime

import streamlit as st
import asyncio
from agents.mcp.server import MCPServerSse
import requests

from fastmcp import FastMCP
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, SQLiteSession, RunConfig, ModelSettings
from openai.types.responses import ResponseTextDeltaEvent, ResponseOutputItemDoneEvent, ResponseFunctionToolCall
from agents import set_default_openai_api, set_tracing_disabled

# OpenAI-agent settings
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

# ===============================
#  MCP Server with 3 custom tools
# ===============================
mcp = FastMCP(
    name="FileTools-MCP-Server",
    instructions="This server provides file system tools: ls, cat, rename.",
)

@mcp.tool
def ls(directory: str) -> str:
    """Lists all files and directories in the specified directory path.

    Args:
        directory: The absolute or relative directory path to list.

    Returns:
        A string listing all files and subdirectories in the given path.
    """
    try:
        if not os.path.exists(directory):
            return f"Error: Directory '{directory}' does not exist."
        if not os.path.isdir(directory):
            return f"Error: '{directory}' is not a directory."
        entries = os.listdir(directory)
        if not entries:
            return f"Directory '{directory}' is empty."
        result_lines = [f"Contents of '{directory}':"]
        for entry in entries:
            full_path = os.path.join(directory, entry)
            if os.path.isdir(full_path):
                result_lines.append(f"  [DIR]  {entry}/")
            else:
                size = os.path.getsize(full_path)
                result_lines.append(f"  [FILE] {entry} ({size} bytes)")
        return "\n".join(result_lines)
    except PermissionError:
        return f"Error: Permission denied to access '{directory}'."
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool
def cat(file_path: str) -> str:
    """Reads and returns the text content of the specified file.

    Args:
        file_path: The absolute or relative path to the file to read.

    Returns:
        The text content of the file, or an error message if reading fails.
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' does not exist."
        if not os.path.isfile(file_path):
            return f"Error: '{file_path}' is not a regular file."
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except PermissionError:
        return f"Error: Permission denied to read '{file_path}'."
    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="gbk") as f:
                content = f.read()
            return content
        except Exception as e:
            return f"Error: Unable to decode file '{file_path}' as UTF-8 or GBK: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool
def rename(old_name: str, new_name: str) -> str:
    """Renames (or moves) a file or directory.

    Args:
        old_name: The current file or directory path.
        new_name: The new name or target path.

    Returns:
        A success or error message.
    """
    try:
        if not os.path.exists(old_name):
            return f"Error: Path '{old_name}' does not exist."
        if os.path.exists(new_name):
            return f"Error: Target path '{new_name}' already exists."
        os.rename(old_name, new_name)
        return f"Success: Renamed '{old_name}' to '{new_name}'."
    except PermissionError:
        return f"Error: Permission denied to rename '{old_name}'."
    except Exception as e:
        return f"Error: {str(e)}"


# ===============================
#  Start MCP server in background
# ===============================
def run_mcp_server():
    asyncio.run(mcp.run(transport="sse", port=8900))


mcp_thread = threading.Thread(target=run_mcp_server, daemon=True)
mcp_thread.start()


# ===============================
#  Streamlit UI
# ===============================
st.set_page_config(page_title="企业职能机器人 - 文件工具助手")

session = SQLiteSession("conversation_file_tools")

with st.sidebar:
    st.title("文件工具助手")
    env_key = os.environ.get("MINIMAX_KEY", "")
    default_key = env_key if env_key else ""
    if "API_TOKEN" in st.session_state and len(st.session_state.get("API_TOKEN", "")) > 1:
        st.success("API Token already configured", icon="✅")
        key = st.session_state["API_TOKEN"]
    else:
        key = st.text_input("Input API Token:", type="password", value=default_key)

    st.session_state["API_TOKEN"] = key

    model_name = "MiniMax-M2.7"
    use_tool = st.checkbox("Enable Tools", value=True)


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "你好，我是企业文件工具助手，可以帮你完成文件操作。目前提供以下3个工具：\n"
                       "1. ls - 列出指定目录下的所有文件\n"
                       "2. cat - 读取文件的文本内容\n"
                       "3. rename - 将文件重命名\n"
                       "请用自然语言描述你的需求，我会帮你选择合适的工具并执行。",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [
        {
            "role": "system",
            "content": "你好，我是企业文件工具助手，可以帮你完成文件操作。目前提供以下3个工具：\n"
                       "1. ls - 列出指定目录下的所有文件\n"
                       "2. cat - 读取文件的文本内容\n"
                       "3. rename - 将文件重命名\n"
                       "请用自然语言描述你的需求，我会帮你选择合适的工具并执行。",
        }
    ]
    global session
    session = SQLiteSession("conversation_file_tools")


st.sidebar.button("Clear Chat", on_click=clear_chat_history)


async def get_model_response(prompt, model_name, use_tool):
    async with MCPServerSse(
        name="FileTools MCP Server",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        client_session_timeout_seconds=20,
    ) as mcp_server:
        external_client = AsyncOpenAI(
            api_key=key,
            base_url="https://api.minimaxi.com/v1",
        )

        if use_tool:
            agent = Agent(
                name="FileTools Assistant",
                instructions=(
                    "你是一个文件系统助手。用户用自然语言描述他们想要进行的文件操作。\n"
                    "可用的工具：\n"
                    "- ls(directory): 列出指定目录下的所有文件和子目录。\n"
                    "- cat(file_path): 读取文件的内容。\n"
                    "- rename(old_name, new_name): 给文件或目录重命名。\n"
                    "根据用户的自然语言请求，选择合适的工具，填充正确的参数，并将工具执行结果返回给用户。\n"
                    "请始终使用中文回答用户。"
                ),
                mcp_servers=[mcp_server],
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                ),
            )
        else:
            agent = Agent(
                name="FileTools Assistant",
                instructions="你是一个有用的助手，请始终用中文回答用户。",
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                ),
            )

        result = Runner.run_streamed(agent, input=prompt, session=session)

        async for event in result.stream_events():
            print(datetime.now(), event)

            if (
                event.type == "raw_response_event"
                and hasattr(event, "data")
                and isinstance(event.data, ResponseOutputItemDoneEvent)
            ):
                if isinstance(event.data.item, ResponseFunctionToolCall):
                    yield "argument", event.data.item

            if (
                event.type == "run_item_stream_event"
                and hasattr(event, "name")
                and event.name == "tool_output"
            ):
                yield "raw", event.item.raw_item["output"]

            if (
                event.type == "raw_response_event"
                and hasattr(event, "data")
                and isinstance(event.data, ResponseTextDeltaEvent)
            ):
                yield "content", event.data.delta


if len(key) > 1:
    if prompt := st.chat_input("请用自然语言描述你要进行的文件操作..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()

            with st.spinner("Processing..."):
                try:
                    async def stream_output():
                        accumulated_text = ""
                        response_generator = get_model_response(prompt, model_name, use_tool)

                        async for event_type, chunk in response_generator:
                            if event_type == "argument":
                                formatted_raw = f"\n\n```json\n[ToolCall]\n{str(chunk)}\n```\n"
                                accumulated_text += formatted_raw
                                placeholder.markdown(accumulated_text + "▌")

                            elif event_type == "raw":
                                formatted_raw = f"\n\n```json\n[ToolResult]\n{str(chunk)}\n```\n"
                                accumulated_text += formatted_raw
                                placeholder.markdown(accumulated_text + "▌")

                            elif event_type == "content":
                                accumulated_text += chunk
                                placeholder.markdown(accumulated_text + "▌")

                        return accumulated_text

                    final_text = asyncio.run(stream_output())
                    placeholder.markdown(final_text)

                except Exception as e:
                    error_msg = f"Error: {e}"
                    placeholder.error(error_msg)
                    traceback.print_exc()
                    final_text = error_msg

            st.session_state.messages.append({"role": "assistant", "content": final_text})
