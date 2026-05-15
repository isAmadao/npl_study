import argparse
import traceback

import requests
TOKEN = "zgaLG8unUPr"

def search_stock(keyword):
    """所有股票，支持代码和名称模糊查询"""
    url = "https://api.autostock.cn/v1/stock/all" + "?token=" + TOKEN
    url += "&keyWord=" + keyword

    payload = {}  # type: ignore
    headers = {}  # type: ignore
    try:
        response = requests.request("GET", url, headers=headers, data=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="股票价格查询工具")
    parser.add_argument("--name", type=str, required=True, help="股票名称或代码")

    args = parser.parse_args()
    print(search_stock(args.name))