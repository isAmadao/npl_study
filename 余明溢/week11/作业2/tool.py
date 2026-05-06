import re
from datetime import datetime
from typing import Annotated, Union

import httpx
import requests
import pytz
import jieba

TOKEN = "6d997a997fbf"

from fastmcp import FastMCP

mcp = FastMCP(
    name="Tools-MCP-Server",
    instructions="""This server contains some api of tools.""",
)


@mcp.tool
def get_city_weather(city_name: Annotated[str, "The Pinyin of the city name (e.g., 'beijing' or 'shanghai')"]):
    """Retrieves the current weather data using the city's Pinyin name."""
    try:
        return requests.get(f"https://whyta.cn/api/tianqi?key={TOKEN}&city={city_name}").json()["data"]
    except:
        return []


@mcp.tool
def get_address_detail(address_text: Annotated[str, "City Name"]):
    """Parses a raw address string to extract detailed components (province, city, district, etc.)."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/addressparse?key={TOKEN}&text={address_text}").json()["result"]
    except:
        return []


@mcp.tool
def get_tel_info(tel_no: Annotated[str, "Tel phone number"]):
    """Retrieves basic information (location, carrier) for a given telephone number."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/mobilelocal?key={TOKEN}&phone={tel_no}").json()["result"]
    except:
        return []


@mcp.tool
def get_scenic_info(scenic_name: Annotated[str, "Scenic/tourist place name"]):
    """Searches for and retrieves information about a specific scenic spot or tourist attraction."""
    # https://apis.whyta.cn/docs/tx-scenic.html
    try:
        return requests.get(f"https://whyta.cn/api/tx/scenic?key={TOKEN}&word={scenic_name}").json()["result"]["list"]
    except:
        return []


@mcp.tool
def get_flower_info(flower_name: Annotated[str, "Flower name"]):
    """Retrieves the flower language (花语) and details for a given flower name."""
    # https://apis.whyta.cn/docs/tx-huayu.html
    try:
        return requests.get(f"https://whyta.cn/api/tx/huayu?key={TOKEN}&word={flower_name}").json()["result"]
    except:
        return []


@mcp.tool
def get_rate_transform(
        source_coin: Annotated[str, "The three-letter code (e.g., USD, CNY) for the source currency."],
        aim_coin: Annotated[str, "The three-letter code (e.g., EUR, JPY) for the target currency."],
        money: Annotated[Union[int, float], "The amount of money to convert."]
):
    """Calculates the currency exchange conversion amount between two specified coins."""
    try:
        return requests.get(
            f"https://whyta.cn/api/tx/fxrate?key={TOKEN}&fromcoin={source_coin}&tocoin={aim_coin}&money={money}").json()[
            "result"]["money"]
    except:
        return []


@mcp.tool
def sentiment_classification(text: Annotated[str, "The text to analyze"]):
    """Classifies the sentiment of a given text."""
    positive_keywords_zh = ['喜欢', '赞', '棒', '优秀', '精彩', '完美', '开心', '满意']
    negative_keywords_zh = ['差', '烂', '坏', '糟糕', '失望', '垃圾', '厌恶', '敷衍']

    positive_pattern = '(' + '|'.join(positive_keywords_zh) + ')'
    negative_pattern = '(' + '|'.join(negative_keywords_zh) + ')'

    positive_matches = re.findall(positive_pattern, text)
    negative_matches = re.findall(negative_pattern, text)

    count_positive = len(positive_matches)
    count_negative = len(negative_matches)

    if count_positive > count_negative:
        return "积极 (Positive)"
    elif count_negative > count_positive:
        return "消极 (Negative)"
    else:
        return "中性 (Neutral)"


@mcp.tool
def query_salary_info(user_name: Annotated[str, "用户名"]):
    """Query user salary baed on the username."""

    # TODO 基于用户名，在数据库中查询，返回数据库查询结果

    if len(user_name) == 2:
        return 1000
    elif len(user_name) == 3:
        return 2000
    else:
        return 3000


@mcp.tool
def query_current_time(timezone: Annotated[str, "Timezone"]):
    """
    Queries the current time and returns the current time.
    :param timezone:
    :return:
    """
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
        return now.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception as e:
        return f"时区无效: {e}"


@mcp.tool
def count_text_length(text: str) -> str:
    """
    Counts the number of characters in a given text.
    :param text:
    :return:
    """
    char_count = len(text)
    words = jieba.lcut(text)
    words = [w for w in words if w.strip() != '']
    word_count = len(words)
    return f"字符数: {char_count}, 单词数: {word_count}"


@mcp.tool
async def translate(text: str, target_lang: str = "auto") -> str:
    """
    中英文双向翻译工具（自动检测源语言）。

    Args:
        text: 需要翻译的文本（支持中文或英文）
        target_lang: 目标语言，可选值: "zh"（中文）、"en"（英文）、"auto"（自动判断）

    Returns:
        翻译后的文本
    """
    # 如果目标语言为 auto，则根据文本内容自动判断
    if target_lang == "auto":
        # 简单判断文本是否包含中文字符
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            target_lang = "en"  # 中文 -> 英文
        else:
            target_lang = "zh"  # 英文 -> 中文

    # 使用 MyMemory 免费翻译 API
    url = "https://api.mymemory.translated.net/get"
    params = {
        "q": text,
        "langpair": f"auto|{target_lang}"
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            translated_text = data.get("responseData", {}).get("translatedText", "")
            if translated_text:
                return translated_text
            else:
                return f"翻译失败，请检查文本或稍后重试。原始文本：{text}"
    except Exception as e:
        return f"翻译服务出错: {str(e)}"


@mcp.tool
async def chinese_to_english(text: str) -> str:
    """将中文文本翻译成英文。"""
    return await translate(text, target_lang="en")


@mcp.tool
async def english_to_chinese(text: str) -> str:
    """将英文文本翻译成中文。"""
    return await translate(text, target_lang="zh")
