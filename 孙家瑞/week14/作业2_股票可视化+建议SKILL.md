---
name: 股票可视化分析
description: 通过 autostock API 获取股票周K线和日K线数据，绘制周波动和日波动叠加图，基于波动大小给出买卖时机建议。
---

# 功能概述

本技能用于对单只股票进行可视化波动分析：

1. 获取指定股票的**周K线**和**日K线**数据
2. 在一张图上同时绘制**周波动**和**日波动**
3. 基于波动幅度大小，给出**买入/卖出/观望**的时间建议

# 数据来源

通过 https://api.autostock.cn 获取K线数据：

| 接口 | 端点 | 说明 |
|------|------|------|
| `get_week_line` | `/v1/stock/kline/week` | 周K线（前复权） |
| `get_day_line` | `/v1/stock/kline/day` | 日K线（前复权） |

# 分析方法

## 一、波动幅度计算

- **周波动** = (本周最高价 - 本周最低价) / 本周开盘价 × 100%
- **日波动** = (当日最高价 - 当日最低价) / 当日开盘价 × 100%

## 二、买卖信号判断

| 周波动趋势 | 日波动信号 | 成交量变化 | 操作建议 |
|-----------|-----------|-----------|---------|
| 周波动收缩中 | 日波动放大 + 阳线 | 放量 | **买入信号**：蓄力突破 |
| 周波动扩张中 | 日波动放大 + 阴线 | 放量 | **卖出信号**：趋势加速见顶 |
| 周波动平稳 | 日波动在均值附近 | 缩量 | 观望：等待方向选择 |
| 周波动收缩至低点 | 日波动极小 | 地量 | **关注**：变盘前兆 |

## 三、推荐输出

- 股票近期波动趋势图（周+日叠加）
- 当前波动所处的相对位置（高/中/低）
- 近期的买入/卖出时机标注
- 风险提示

# 调用方法

```python
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
from typing import Optional
import traceback

TOKEN = "zgaLG8unUPr"
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def get_stock_week_kline(
    code: str,
    startDate: Optional[str] = None,
    endDate: Optional[str] = None,
) -> dict:
    """获取周K线数据"""
    url = "https://api.autostock.cn/v1/stock/kline/week?token=" + TOKEN
    payload = {"code": code, "startDate": startDate, "endDate": endDate, "type": 1}
    try:
        response = requests.request("GET", url, params=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}


def get_stock_day_kline(
    code: str,
    startDate: Optional[str] = None,
    endDate: Optional[str] = None,
) -> dict:
    """获取日K线数据"""
    url = "https://api.autostock.cn/v1/stock/kline/day?token=" + TOKEN
    payload = {"code": code, "startDate": startDate, "endDate": endDate, "type": 1}
    try:
        response = requests.request("GET", url, params=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}


def analyze_and_plot(code: str, days_back: int = 90):
    """
    获取数据、计算波动、绘图并给出建议。

    Args:
        code: 股票代码，如 '000001' (深发展) 或 '600519' (贵州茅台)
        days_back: 回溯天数，默认90天
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    print(f"正在获取 {code} 的K线数据...")

    week_data = get_stock_week_kline(code, start_date, end_date)
    day_data = get_stock_day_kline(code, start_date, end_date)

    # 解析周K线
    week_records = week_data.get("data", [])
    if not week_records:
        print("未获取到周K线数据")
        return

    # 解析日K线
    day_records = day_data.get("data", [])
    if not day_records:
        print("未获取到日K线数据")
        return

    # ---- 计算波动 ----
    week_dates = []
    week_fluctuation = []  # 周波动 %
    for item in week_records:
        try:
            date = item.get("date", "")
            open_p = float(item.get("open", 0))
            high = float(item.get("high", 0))
            low = float(item.get("low", 0))
            close = float(item.get("close", 0))
            if open_p == 0:
                continue
            fluc = (high - low) / open_p * 100
            week_dates.append(datetime.strptime(date, "%Y-%m-%d"))
            week_fluctuation.append(fluc)
        except (ValueError, TypeError):
            continue

    day_dates = []
    day_fluctuation = []  # 日波动 %
    day_volume = []
    day_is_up = []  # 阳线/阴线
    for item in day_records:
        try:
            date = item.get("date", "")
            open_p = float(item.get("open", 0))
            high = float(item.get("high", 0))
            low = float(item.get("low", 0))
            close = float(item.get("close", 0))
            volume = float(item.get("volume", 0))
            if open_p == 0:
                continue
            fluc = (high - low) / open_p * 100
            day_dates.append(datetime.strptime(date, "%Y-%m-%d"))
            day_fluctuation.append(fluc)
            day_volume.append(volume)
            day_is_up.append(close >= open_p)
        except (ValueError, TypeError):
            continue

    if not week_fluctuation or not day_fluctuation:
        print("数据解析异常")
        return

    # ---- 计算信号 ----
    avg_week_fluc = np.mean(week_fluctuation)
    recent_week_fluc = week_fluctuation[-1] if week_fluctuation else 0
    prev_week_fluc = week_fluctuation[-2] if len(week_fluctuation) > 1 else recent_week_fluc
    avg_day_fluc = np.mean(day_fluctuation)
    recent_days_fluc = np.mean(day_fluctuation[-5:]) if len(day_fluctuation) >= 5 else avg_day_fluc

    # 成交量趋势
    if len(day_volume) >= 10:
        recent_vol = np.mean(day_volume[-5:])
        prev_vol = np.mean(day_volume[-10:-5])
        vol_trend = "放量" if recent_vol > prev_vol * 1.1 else ("缩量" if recent_vol < prev_vol * 0.9 else "平稳")
    else:
        vol_trend = "数据不足"

    # 判断信号
    week_shrinking = recent_week_fluc < avg_week_fluc * 0.8
    week_expanding = recent_week_fluc > avg_week_fluc * 1.3
    day_surging = recent_days_fluc > avg_day_fluc * 1.5
    last_day_up = day_is_up[-1] if day_is_up else False

    signal = ""
    if week_shrinking and day_surging and last_day_up and vol_trend == "放量":
        signal = "📈 买入信号：周波动收缩蓄力 + 日波动放大突破 + 放量阳线，建议关注买入时机"
    elif week_expanding and day_surging and not last_day_up and vol_trend == "放量":
        signal = "📉 卖出信号：周波动扩张加速 + 日波动放大阴线 + 放量下跌，建议减仓或卖出"
    elif week_shrinking and recent_days_fluc < avg_day_fluc * 0.6:
        signal = "👀 变盘前兆：周波动和日波动同时收缩至极低水平，配合地量即是买入良机"
    elif week_expanding:
        signal = "⚠️ 高波动期：周波动处于高位，注意风险控制，短线可波段操作"
    else:
        signal = "➡️ 观望：波动处于正常范围，等待更明确的方向信号"

    print(f"\n===== 波动分析报告 =====")
    print(f"股票代码: {code}")
    print(f"平均周波动: {avg_week_fluc:.2f}% | 最近周波动: {recent_week_fluc:.2f}%")
    print(f"平均日波动: {avg_day_fluc:.2f}% | 近5日日波动均值: {recent_days_fluc:.2f}%")
    print(f"成交量趋势: {vol_trend}")
    print(f"操作建议: {signal}")

    # ---- 绘图 ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    fig.suptitle(f"{code} 波动分析 - 周波动 & 日波动", fontsize=14, fontweight="bold")

    # 周波动子图
    colors_week = ["red" if f >= avg_week_fluc else "steelblue" for f in week_fluctuation]
    ax1.bar(week_dates, week_fluctuation, color=colors_week, alpha=0.85, width=3)
    ax1.axhline(y=avg_week_fluc, color="orange", linestyle="--", linewidth=1.5, label=f"均值 {avg_week_fluc:.2f}%")
    ax1.axhline(y=avg_week_fluc * 1.3, color="red", linestyle=":", linewidth=1, alpha=0.5)
    ax1.axhline(y=avg_week_fluc * 0.6, color="green", linestyle=":", linewidth=1, alpha=0.5)
    ax1.set_ylabel("周波动 (%)", fontsize=11)
    ax1.set_title("周K线波动幅度", fontsize=12)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

    # 日波动子图
    colors_day = []
    for i, f in enumerate(day_fluctuation):
        if f > avg_day_fluc * 1.3:
            colors_day.append("red")
        elif f < avg_day_fluc * 0.6:
            colors_day.append("green")
        else:
            colors_day.append("steelblue")

    ax2.bar(day_dates, day_fluctuation, color=colors_day, alpha=0.7, width=0.8)
    ax2.axhline(y=avg_day_fluc, color="orange", linestyle="--", linewidth=1.5, label=f"均值 {avg_day_fluc:.2f}%")
    ax2.set_ylabel("日波动 (%)", fontsize=11)
    ax2.set_xlabel("日期", fontsize=11)
    ax2.set_title("日K线波动幅度 (红色=高波动 / 绿色=低波动)", fontsize=12)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

    # 标注买卖信号
    if "买入" in signal:
        ax2.annotate(
            "买入信号",
            xy=(day_dates[-1], day_fluctuation[-1]),
            xytext=(day_dates[-1], day_fluctuation[-1] + max(day_fluctuation) * 0.15),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
            fontsize=11,
            color="red",
            fontweight="bold",
        )
    elif "卖出" in signal:
        ax2.annotate(
            "卖出信号",
            xy=(day_dates[-1], day_fluctuation[-1]),
            xytext=(day_dates[-1], day_fluctuation[-1] + max(day_fluctuation) * 0.15),
            arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
            fontsize=11,
            color="green",
            fontweight="bold",
        )

    plt.tight_layout()
    save_path = f"{code}_波动分析.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"图表已保存至: {save_path}")
    plt.show()
    return signal


# 使用示例
# analyze_and_plot("000001")  # 深发展
# analyze_and_plot("600519")  # 贵州茅台
```
