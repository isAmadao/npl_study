import re
from fastmcp import FastMCP
from datetime import datetime
from typing import Literal

mcp = FastMCP(
    name="Tools-MCP-Server",
    instructions="""This server contains some api of tools.""",
)

@mcp.tool
def get_current_time(
    format: Literal["full", "date", "time"] = "full"
) -> str:
    """获取当前本地时间，支持多种格式"""
    now = datetime.now()
    if format == "full":
        return now.strftime("%Y-%m-%d %H:%M:%S")
    elif format == "date":
        return now.strftime("%Y-%m-%d")
    else:
        return now.strftime("%H:%M:%S")

@mcp.tool
def validate_email(email: str) -> bool:
    """验证邮箱格式是否合法"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

@mcp.tool
def text_stats(text: str) -> dict:
    """统计文本中的字符数（含空格）、单词数、行数"""
    lines = text.splitlines()
    return {
        "char_count": len(text),
        "word_count": len(text.split()),
        "line_count": len(lines)
    }