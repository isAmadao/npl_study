import datetime
from typing import Annotated
import requests

from fastmcp import FastMCP
mcp = FastMCP(
    name="Tools-MCP-Server",
    instructions="""This server contains some api of tools.""",
)


@mcp.tool
def get_ip_info(
    ip_address: Annotated[str, "IPv4/IPv6 address to look up, e.g. '8.8.8.8'. Leave empty string to query the caller's own public IP."]
):
    """
    Looks up geolocation and ISP info for an IP address (country, region, city, ISP, timezone).
    Uses ip-api.com — free, no token required.
    Pass an empty string '' to query the current machine's public IP.
    """
    try:
        target = ip_address.strip() if ip_address.strip() else ""
        url = f"http://ip-api.com/json/{target}?lang=zh-CN&fields=status,message,country,regionName,city,isp,org,lat,lon,timezone,query"
        resp = requests.get(url, timeout=5).json()
        if resp.get("status") != "success":
            return {"error": resp.get("message", "查询失败，请检查IP地址格式")}
        return {
            "IP地址":  resp.get("query"),
            "国家":    resp.get("country"),
            "省份/地区": resp.get("regionName"),
            "城市":    resp.get("city"),
            "运营商":  resp.get("isp"),
            "组织":    resp.get("org"),
            "时区":    resp.get("timezone"),
            "纬度":    resp.get("lat"),
            "经度":    resp.get("lon"),
        }
    except Exception as e:
        return {"error": f"请求失败: {str(e)}"}


@mcp.tool
def get_random_joke(
    category: Annotated[str, "Joke category: 'general', 'programming', 'knock-knock', or 'random' for any category."]
):
    """
    Fetches a random English joke from the Official Joke API.
    Returns the setup (question) and punchline (answer).
    Uses official-joke-api.appspot.com — free, no token required.
    """
    try:
        valid_categories = {"general", "programming", "knock-knock"}
        if category.lower() in valid_categories:
            url = f"https://official-joke-api.appspot.com/jokes/{category.lower()}/random"
        else:
            url = "https://official-joke-api.appspot.com/random_joke"

        resp = requests.get(url, timeout=5).json()
        # 单个joke时是dict，按category取时是list
        joke = resp[0] if isinstance(resp, list) else resp
        return {
            "类型":    joke.get("type"),
            "问":      joke.get("setup"),
            "答（笑点）": joke.get("punchline"),
        }
    except Exception as e:
        return {"error": f"请求失败: {str(e)}"}


@mcp.tool
def get_date_info(
    date_str: Annotated[str, "Date string in 'YYYY-MM-DD' format, e.g. '2025-10-01'. Pass 'today' to query today."]
):
    """
    Calculates detailed information about a given date using Python's standard library (no network needed).
    Returns weekday, day of year, week number, whether it's a weekend, days until/since today, and zodiac sign.
    """
    try:
        if date_str.strip().lower() == "today":
            target = datetime.date.today()
        else:
            target = datetime.date.fromisoformat(date_str.strip())

        today = datetime.date.today()
        delta = (target - today).days

        weekdays_zh = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        weekday_zh = weekdays_zh[target.weekday()]
        is_weekend = target.weekday() >= 5

        # 计算星座
        def get_zodiac(month, day):
            zodiacs = [
                ((1, 20), (2, 18),  "水瓶座"),
                ((2, 19), (3, 20),  "双鱼座"),
                ((3, 21), (4, 19),  "白羊座"),
                ((4, 20), (5, 20),  "金牛座"),
                ((5, 21), (6, 20),  "双子座"),
                ((6, 21), (7, 22),  "巨蟹座"),
                ((7, 23), (8, 22),  "狮子座"),
                ((8, 23), (9, 22),  "处女座"),
                ((9, 23), (10, 22), "天秤座"),
                ((10, 23),(11, 21), "天蝎座"),
                ((11, 22),(12, 21), "射手座"),
                ((12, 22),(1, 19),  "摩羯座"),
            ]
            for start, end, name in zodiacs:
                if start[0] == end[0]:  # 同月
                    if month == start[0] and start[1] <= day <= end[1]:
                        return name
                else:
                    if (month == start[0] and day >= start[1]) or \
                       (month == end[0] and day <= end[1]):
                        return name
            return "摩羯座"

        zodiac = get_zodiac(target.month, target.day)

        if delta == 0:
            distance_desc = "就是今天"
        elif delta > 0:
            distance_desc = f"距今还有 {delta} 天"
        else:
            distance_desc = f"已过去 {abs(delta)} 天"

        return {
            "查询日期":   str(target),
            "星期":       weekday_zh,
            "是否周末":   "是" if is_weekend else "否",
            "今年第几天":  target.timetuple().tm_yday,
            "今年第几周":  target.isocalendar()[1],
            "所属星座":   zodiac,
            "距今":       distance_desc,
        }
    except ValueError:
        return {"error": "日期格式错误，请使用 'YYYY-MM-DD' 格式，例如 '2025-10-01'"}
    except Exception as e:
        return {"error": f"计算失败: {str(e)}"}