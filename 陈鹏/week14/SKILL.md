---
name: 股票波动分析与买卖建议
description: 获取股票日K线和周K线数据，绘制波动图表，并基于波动大小给出买入卖出的最佳时间建议。
---

# 功能概述

本 Skill 提供股票波动可视化分析功能，包括：
1. 获取股票日K线和周K线数据
2. 绘制日波动和周波动对比图
3. 基于波动大小分析，给出买入卖出时机建议

# 核心分析方法

## 一、波动率计算
日波动率 = (当日最高价 - 当日最低价) / 当日开盘价 × 100%
周波动率 = (当周最高价 - 当周最低价) / 当周开盘价 × 100%

## 二、波动等级与操作
| 波动状态 | 数值范围 | 操作建议 |
|----------|----------|----------|
| 低波动 | <3% | 观望，等待方向 |
| 中波动 | 3%~8% | 顺势操作 |
| 高波动 | >8% | 警惕变盘，减仓或快进快出 |

## 三、买卖信号
### 买入信号
1. 周波动连续收窄后突然放大
2. 日波动放大且收阳线
3. 价格在支撑位出现大波动阳线

### 卖出信号
1. 周波动持续放大后突然收窄
2. 日波动放大且收阴线
3. 价格在压力位出现大波动阴线

# 调用方法
```python
import requests
import matplotlib.pyplot as plt
import sys
import io

# Windows控制台编码修复
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

TOKEN = "zgaLG8unUPr"
BASE_URL = "https://api.autostock.cn/v1"

def get_day_line(code):
    url = f"{BASE_URL}/stock/kline/day?token={TOKEN}&code={code}"
    return requests.get(url, timeout=10).json()

def get_week_line(code):
    url = f"{BASE_URL}/stock/kline/week?token={TOKEN}&code={code}"
    return requests.get(url, timeout=10).json()

def calculate_volatility(data):
    vols = []
    for d in data:
        open_p = float(d[1])
        high_p = float(d[3])
        low_p = float(d[4])
        vols.append((high_p - low_p) / open_p * 100)
    return vols

def analyze(day_vol, week_vol):
    avg_day = sum(day_vol[-5:])/5
    avg_week = sum(week_vol[-4:])/4
    if avg_day < 3:
        return "低波动，观望"
    elif avg_day > 8:
        return "高波动，注意风险，快进快出"
    else:
        return "中波动，顺势操作"

def plot_and_analyze(code):
    day_data = get_day_line(code)["data"][-60:]
    week_data = get_week_line(code)["data"][-20:]
    day_vol = calculate_volatility(day_data)
    week_vol = calculate_volatility(week_data)
    
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,7))
    ax1.plot(day_vol, color="blue")
    ax1.set_title("日波动")
    ax2.plot(week_vol, color="red")
    ax2.set_title("周波动")
    plt.tight_layout()
    plt.savefig(f"{code}_wave.png")
    return analyze(day_vol, week_vol)

# 使用
# print(plot_and_analyze("000001"))
```

# 输出结果

分析完成后输出：
1. **波动率对比图** - 日波动和周波动可视化
2. **波动状态判断** - 低/中/高波动
3. **买卖时机建议** - BUY/SELL/HOLD
4. **具体操作建议** - 文字说明