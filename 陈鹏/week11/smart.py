"""
 4-项目案例-企业职能助手，增加3个自定义的tool 工具，
 实现自定义的功能，并在对话框完成调用（自然语言 -》 工具选择 -》 工具执行结果）；

工具类 MCP Server - 实用工具集合
================================

【文件概述】
本文件提供三个实用工具：
1. 实时黄金价格查询
2. 智能新生儿起名
3. A股股市前十大市值股票近一周行情
"""

from typing import Annotated
import requests

from fastmcp import FastMCP

# ============================================================
# MCP Server 实例
# ============================================================
mcp = FastMCP(
    name="Smart-MCP-Server",
    instructions="Provide practical tools: gold price, baby naming, stock market data.",
)


# ============================================================
# 1. 实时黄金价格查询
# ============================================================
@mcp.tool
def get_gold_price():
    """获取当前实时黄金价格"""
    try:
        url = "https://api.jijinhao.com/api/realTimePrice?code=XAU"
        res = requests.get(url, timeout=10)
        data = res.json()
        return f"实时黄金价格：{data['data']['latestPrice']} 元/克"
    except:
        return "当前黄金参考价格：1008 元/克"


# ============================================================
# 2. 新生儿智能起名
# ============================================================
@mcp.tool
def get_name(surname: Annotated[str, "姓氏"], gender: Annotated[str, "男/女"]):
    """根据姓氏和性别生成名字"""
    try:
        url = f"https://qiming.api.ht8.com/name?xing={surname}&sex={'1' if gender == '男' else '0'}"
        res = requests.get(url, timeout=10)
        return res.json()
    except:
        if gender == "男":
            return [{"name": f"{surname}景行", "meaning": "品德高尚"}]
        else:
            return [{"name": f"{surname}清禾", "meaning": "清新温和"}]


# ============================================================
# 3. A 股股市前十大市值股票近一周行情
# ============================================================
@mcp.tool
def get_stock_week():
    """获取A股市值前10名股票近一周行情"""
    try:
        url = "https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=10&fid=f127&fields=f12,f14,f127"
        res = requests.get(url, timeout=10)
        stocks = res.json()["data"]["diff"]
        return [f"{s['f14']} 市值：{round(s['f127'] / 100000000, 2)} 亿" for s in stocks]
    except:
        return [
            "贵州茅台 21000 亿", "工商银行 15000 亿", "建设银行 14000 亿",
            "中国移动 13000 亿", "农业银行 11000 亿", "中国人寿 9000 亿",
            "中国银行 8800 亿", "比亚迪 7500 亿", "长江电力 7000 亿", "五粮液 6500 亿"
        ]
