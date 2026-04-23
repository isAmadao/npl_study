import re
from typing import Annotated, Union
import requests


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
        return requests.get(f"https://whyta.cn/api/tx/fxrate?key={TOKEN}&fromcoin={source_coin}&tocoin={aim_coin}&money={money}").json()["result"]["money"]
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
def query_leave_balance(employee_name: Annotated[str, "The name of the employee"]) -> str:
    """Query the remaining annual leave days for an employee."""
    import random
    mock_db = {"Alice": 12, "Bob": 5, "Charlie": 0, "David": 8}
    days = mock_db.get(employee_name, random.randint(0, 15))
    return f"{employee_name} has {days} annual leave days remaining."


@mcp.tool
def find_available_meeting_times(
    room_name: Annotated[str, "The meeting room name, e.g. 'Room A', 'Room B', 'Conference Hall'"],
    date: Annotated[str, "Date in YYYY-MM-DD format"],
) -> str:
    """Find available time slots for a meeting room on a given date."""
    available_rooms = ["Room A", "Room B", "Conference Hall"]
    if room_name not in available_rooms:
        return f"Room '{room_name}' not found. Available: {', '.join(available_rooms)}."

    return "8:00am -12:00pm"


@mcp.tool
def submit_time_off_request(
    employee_name: Annotated[str, "The name of the employee submitting the request"],
    start_date: Annotated[str, "Start date of leave in YYYY-MM-DD format"],
    end_date: Annotated[str, "End date of leave in YYYY-MM-DD format"],
    reason: Annotated[str, "Reason for the time off request"],
) -> str:
    """Submit a time off / leave request for an employee."""
    import random
    request_id = f"LV-{random.randint(10000, 99999)}"
    return (
        f"Time off request submitted!\n"
        f"Request ID: {request_id}\n"
        f"Employee: {employee_name} | {start_date} to {end_date}\n"
        f"Reason: {reason}\n"
        f"Status: Pending manager approval"
    )
