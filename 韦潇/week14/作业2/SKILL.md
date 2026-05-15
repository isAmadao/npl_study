---
name: 股票可视化分析
description: 对股票进行可视化分析，绘制周波动和日波动图表，并基于波动大小给出买入卖出的最佳时间建议。
---

# 股票可视化分析技能

## 功能概述

本技能提供股票数据的可视化分析功能，主要包括：
- 绘制股票的日K线和周K线波动图
- 计算波动率指标
- 基于波动分析提供买入卖出建议

## 核心功能

### 1. 数据获取接口

| 接口名称 | 功能描述 | 参数 |
| :-------- | :-------- | :-------- |
| `get_stock_daily_data` | 获取日K线数据 | code: 股票代码 |
| `get_stock_weekly_data` | 获取周K线数据 | code: 股票代码 |
| `get_stock_info` | 获取股票基础信息 | code: 股票代码 |

### 2. 可视化功能

#### 2.1 波动图绘制

支持在同一图表中展示：
- **日波动曲线**：显示每日收盘价走势
- **周波动曲线**：显示每周收盘价走势
- **波动率指标**：计算并显示波动率变化

#### 2.2 图表元素

```
┌─────────────────────────────────────────────────────┐
│                   股票波动分析图                      │
├─────────────────────────────────────────────────────┤
│  日K线: ████▄▄▄███▀▀▀▄▄▄████████▄▄▄▀▀▀███▄▄▄      │
│  周K线: ═══════╪═══════╪═══════╪═══════╪══════      │
│                                                     │
│  支撑位: ───────────────────────────────────────     │
│  压力位: ───────────────────────────────────────     │
│                                                     │
│  波动率: ▁▁▁▁▂▂▂▃▃▃▄▄▄▅▅▅▆▆▆▇▇▇██████▇▇▇▆▆▆       │
└─────────────────────────────────────────────────────┘
```

### 3. 波动分析方法

#### 3.1 波动率计算

```python
# 日波动率 = |当日收盘价 - 前一日收盘价| / 前一日收盘价
daily_volatility = abs(close_price - prev_close) / prev_close * 100

# 周波动率 = |当周收盘价 - 上周收盘价| / 上周收盘价
weekly_volatility = abs(week_close - prev_week_close) / prev_week_close * 100

# 平均波动率 = 最近N日波动率的平均值
avg_volatility = mean(daily_volatility[-N:])
```

#### 3.2 波动等级划分

| 波动率范围 | 等级 | 市场状态 | 操作建议 |
| :--------- | :--- | :-------- | :-------- |
| < 1% | 低波动 | 横盘整理 | 观望或小幅建仓 |
| 1%-3% | 正常波动 | 健康走势 | 持有或波段操作 |
| 3%-5% | 中高波动 | 活跃行情 | 谨慎操作，控制仓位 |
| > 5% | 高波动 | 剧烈波动 | 减仓或等待企稳 |

### 4. 买卖时机建议

#### 4.1 买入信号

**多头信号组合：**
- 日K线放量突破周K线压力位
- 波动率从高位回落至正常范围
- 收盘价站上5日和10日均线
- MACD指标金叉向上

**买入时机评分：**
| 信号数量 | 买入强度 | 建议仓位 |
| :-------- | :-------- | :-------- |
| 4个 | 强烈买入 | 70%-100% |
| 3个 | 买入 | 50%-70% |
| 2个 | 谨慎买入 | 30%-50% |
| < 2个 | 观望 | 0%-30% |

#### 4.2 卖出信号

**空头信号组合：**
- 日K线放量跌破周K线支撑位
- 波动率急剧上升超过5%
- 收盘价跌破5日和10日均线
- MACD指标死叉向下

**卖出时机评分：**
| 信号数量 | 卖出强度 | 建议仓位 |
| :-------- | :-------- | :-------- |
| 4个 | 强烈卖出 | 清仓 |
| 3个 | 卖出 | 减仓至30%以下 |
| 2个 | 谨慎卖出 | 减仓50% |
| < 2个 | 观望 | 持有 |

#### 4.3 最佳买卖时间窗口

| 时段 | 特点 | 建议操作 |
| :---- | :---- | :-------- |
| 开盘后30分钟 | 波动较大，假信号多 | 观察为主，不急于操作 |
| 上午10:00-11:30 | 趋势逐渐明朗 | 适合买入操作 |
| 下午13:30-14:30 | 趋势确认阶段 | 适合加仓或减仓 |
| 收盘前30分钟 | 可能出现异动 | 谨慎操作，避免尾盘风险 |

## 调用示例

```python
import requests
import matplotlib.pyplot as plt
import pandas as pd

# 获取股票数据
def get_stock_data(code):
    url = f"https://api.autostock.cn/v1/stock/kline/day?token=zgaLG8unUPr&code={code}"
    response = requests.get(url)
    return response.json()

# 绘制波动图
def plot_volatility(stock_code):
    daily_data = get_stock_daily_data(stock_code)
    weekly_data = get_stock_weekly_data(stock_code)
    
    df_daily = pd.DataFrame(daily_data['data'])
    df_weekly = pd.DataFrame(weekly_data['data'])
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 日K线收盘价
    ax1.plot(df_daily['date'], df_daily['close'], label='日收盘价', color='blue', alpha=0.6)
    
    # 周K线收盘价
    ax1.plot(df_weekly['date'], df_weekly['close'], label='周收盘价', color='red', linewidth=2)
    
    ax1.set_xlabel('日期')
    ax1.set_ylabel('收盘价')
    ax1.legend()
    
    # 波动率子图
    ax2 = ax1.twinx()
    ax2.plot(df_daily['date'], df_daily['volatility'], label='日波动率', color='green', linestyle='--')
    ax2.set_ylabel('波动率(%)')
    ax2.legend()
    
    plt.title(f'{stock_code} 周/日波动分析图')
    plt.show()

# 获取买卖建议
def get_trade_recommendation(stock_code):
    data = analyze_stock(stock_code)
    return generate_recommendation(data)
```

## 输出格式

### 分析报告结构

```json
{
  "stock_code": "600519",
  "stock_name": "贵州茅台",
  "analysis_date": "2024-01-15",
  "volatility": {
    "daily_avg": 2.3,
    "weekly_avg": 8.5,
    "level": "正常波动",
    "trend": "上升"
  },
  "support_level": 1600.0,
  "resistance_level": 1800.0,
  "signal": {
    "buy_score": 3,
    "sell_score": 1,
    "recommendation": "买入",
    "suggested_position": "50%-70%"
  },
  "best_time": {
    "buy_window": "上午10:00-11:30",
    "sell_window": "下午13:30-14:30"
  }
}
```

### 可视化输出

生成的图表将包含：
1. 日K线和周K线叠加图
2. 波动率曲线
3. 支撑位和压力位标记
4. 买卖信号标注

## 使用场景

本技能适用于以下场景：
- 股票投资者进行技术分析
- 量化交易策略开发
- 投资顾问提供专业建议
- 金融教学演示

## 注意事项

1. 本分析基于历史数据，不构成投资建议
2. 建议结合基本面分析综合判断
3. 高波动时期注意风险控制
4. 使用前请确认API接口可用性

--- 

*风险提示：股市有风险，投资需谨慎。本工具仅供参考，不构成任何投资建议。*
