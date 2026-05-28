import re
from typing import Annotated, Union
import requests
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
def get_air_quality_china(
        city_name: Annotated[str, "城市中文名称（如：北京、上海）"]
):
    """获取中国城市的实时空气质量数据"""
    try:
        # 城市编码映射
        city_codes = {
            "北京": 110000, "上海": 310000, "广州": 440100, "深圳": 440300,
            "成都": 510100, "杭州": 330100, "南京": 320100, "武汉": 420100,
            "西安": 610100, "重庆": 500000, "天津": 120000, "沈阳": 210100,
            "青岛": 370200, "大连": 370200, "厦门": 350200, "宁波": 330200,
            "苏州": 320500, "无锡": 320200, "长沙": 430100, "郑州": 410100,
            "济南": 370100, "合肥": 340100, "福州": 350100, "昆明": 530100,
            "哈尔滨": 230100, "长春": 220100, "太原": 140100, "石家庄": 130100,
            "兰州": 620100, "西宁": 630100, "银川": 640100, "乌鲁木齐": 650100,
            "拉萨": 540100, "南宁": 450100, "海口": 460100, "贵阳": 520100
        }

        city_code = city_codes.get(city_name)
        if not city_code:
            return {"error": f"暂不支持该城市: {city_name}"}

        # 使用第三方聚合API获取中国空气质量数据
        url = f"https://www.pm25.in/api/querys/aqi_details.json?city={city_name}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://www.pm25.in/",
            "Accept": "application/json"
        }

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()

            if isinstance(data, list) and len(data) > 0:
                # 取第一个监测站的数据
                station_data = data[0]

                # 解析AQI等级
                aqi = int(station_data.get("aqi", 0))
                aqi_levels = {
                    (0, 50): "优",
                    (51, 100): "良",
                    (101, 150): "轻度污染",
                    (151, 200): "中度污染",
                    (201, 300): "重度污染",
                    (301, float('inf')): "严重污染"
                }

                aqi_level = "未知"
                for (low, high), level in aqi_levels.items():
                    if low <= aqi <= high:
                        aqi_level = level
                        break

                return {
                    "city": city_name,
                    "station": station_data.get("position_name", "未知监测点"),
                    "aqi": aqi,
                    "aqi_level": aqi_level,
                    "primary_pollutant": station_data.get("primary_pollutant", "未知"),
                    "pm2_5": float(station_data.get("pm2_5", 0)),
                    "pm10": float(station_data.get("pm10", 0)),
                    "no2": float(station_data.get("no2", 0)),
                    "so2": float(station_data.get("so2", 0)),
                    "co": float(station_data.get("co", 0)),
                    "o3": float(station_data.get("o3", 0)),
                    "temperature": station_data.get("temperature", "未知"),
                    "humidity": station_data.get("humidity", "未知"),
                    "wind_speed": station_data.get("wind_speed", "未知"),
                    "wind_direction": station_data.get("wind_direction", "未知"),
                    "update_time": station_data.get("time_point", "未知"),
                    "data_source": "中国环境监测总站"
                }
            else:
                return {"error": "未找到空气质量数据"}
        else:
            return {"error": f"请求失败: {response.status_code}"}

    except Exception as e:
        return {"error": f"获取数据失败: {str(e)}"}


@mcp.tool
def get_city_info_reliable(city_name: Annotated[str, "Major city name in English"]):
    # 扩展城市数据字典
    # 可以在这里添加更多城市
    cities_data = {
        "beijing": {"name": "Beijing", "country": "China", "population": 21540000},
        "shanghai": {"name": "Shanghai", "country": "China", "population": 24280000},
        "new york": {"name": "New York City", "country": "USA", "population": 8419000},
        "london": {"name": "London", "country": "UK", "population": 8908081},
        "tokyo": {"name": "Tokyo", "country": "Japan", "population": 13960000},
        "paris": {"name": "Paris", "country": "France", "population": 2148000},
        "moscow": {"name": "Moscow", "country": "Russia", "population": 12500000},
        "sydney": {"name": "Sydney", "country": "Australia", "population": 5312000}
    }

    city_key = city_name.lower().strip()

    for key, info in cities_data.items():
        if city_key in [key, info["name"].lower()]:
            return info

    return {
        "error": f"City '{city_name}' not found in local database",
        "available_cities": list(cities_data.keys())
    }

@mcp.tool
def convert_currency(
        amount: Annotated[float, "Amount to convert"],
        from_currency: Annotated[str, "Source currency code (e.g., USD)"],
        to_currency: Annotated[str, "Target currency code (e.g., CNY)"]
):
    """Converts an amount from one currency to another."""
    try:
        data = requests.get(
            f"https://api.exchangerate-api.com/v4/latest/{from_currency.upper()}"
        ).json()
        rate = data["rates"].get(to_currency.upper())
        if rate:
            converted = amount * rate
            return {
                "from": from_currency.upper(),
                "to": to_currency.upper(),
                "amount": amount,
                "rate": rate,
                "converted": round(converted, 2)
            }
        return {"error": "Currency not supported"}
    except:
        return {"error": "Failed to fetch exchange rates"}
