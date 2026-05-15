---
name: 股票可视化与交易信号
description: 基于autostock数据接口，实现K线可视化、周/日波动叠加分析，并生成买卖时机建议。
dependencies:
  - autostock

---

# 功能概述

本Skill基于autostock获取股票K线数据，通过图表可视化帮助用户分析股票走势，并基于波动率计算给出买卖时机建议。

# 使用方法

```python
# 在agent中调用
stock_visualization(代码, 时间范围)
stock_analysis(代码, 分析周期)
buy_sell_signal(代码)
```

# 核心函数

## 1. 获取并绘制K线图

```python
import requests
import json

TOKEN = "zgaLG8unUPr"

def get_week_kline(code: str, startDate: str = None, endDate: str = None) -> list:
    """获取周K线数据"""
    url = f"https://api.autostock.cn/v1/stock/kline/week?token={TOKEN}"
    params = {"code": code, "startDate": startDate, "endDate": endDate, "type": 1}
    response = requests.get(url, params=params, timeout=10)
    return response.json().get("data", [])

def get_day_kline(code: str, startDate: str = None, endDate: str = None) -> list:
    """获取日K线数据"""
    url = f"https://api.autostock.cn/v1/stock/kline/day?token={TOKEN}"
    params = {"code": code, "startDate": startDate, "endDate": endDate, "type": 1}
    response = requests.get(url, params=params, timeout=10)
    return response.json().get("data", [])

def get_minute_data(code: str) -> list:
    """获取分时数据"""
    url = f"https://api.autostock.cn/v1/stock/min?token={TOKEN}&code={code}"
    response = requests.get(url, timeout=10)
    return response.json().get("data", [])
```

## 2. 数据处理函数

```python
import pandas as pd
import numpy as np

def calculate_volatility(kline_data: list, window: int = 5) -> pd.DataFrame:
    """
    计算波动率指标
    window: 计算周期（默认5天/周）
    返回包含：收盘价、涨跌幅、波动率、移动平均
    """
    if not kline_data:
        return pd.DataFrame()

    df = pd.DataFrame(kline_data)
    df['price_change'] = df['close'].pct_change() * 100  # 涨跌幅%
    df['volatility'] = df['price_change'].rolling(window=window).std()  # 波动率标准差
    df['avg_volatility'] = df['price_change'].rolling(window=window).mean()  # 平均波动
    df['volume_ma'] = df['volume'].rolling(window=5).mean()  # 成交量均线

    return df

def merge_week_day_data(week_data: list, day_data: list) -> dict:
    """
    合并周线和日线数据用于叠加显示
    返回：{
        'dates': 时间轴,
        'week_close': 周线收盘价,
        'week_volatility': 周波动率,
        'day_close': 日线收盘价,
        'day_volatility': 日波动率
    }
    """
    result = {
        'dates': [],
        'week_close': [],
        'week_volatility': [],
        'day_close': [],
        'day_volatility': []
    }

    week_df = calculate_volatility(week_data, window=4)  # 4周波动
    day_df = calculate_volatility(day_data, window=5)     # 5日波动

    # 提取数据点（去重）
    week_dates = [d.get('date', d.get('date_')) for d in week_data]
    day_dates = [d.get('date', d.get('date_')) for d in day_data]

    result['dates'] = sorted(set(week_dates + day_dates))
    result['week_close'] = week_df['close'].tolist() if len(week_df) > 0 else []
    result['week_volatility'] = week_df['volatility'].tolist() if len(week_df) > 0 else []
    result['day_close'] = day_df['close'].tolist() if len(day_df) > 0 else []
    result['day_volatility'] = day_df['volatility'].tolist() if len(day_df) > 0 else []

    return result

def calculate_ma(data: list, period: int) -> list:
    """计算移动平均线"""
    if len(data) < period:
        return data
    result = []
    for i in range(len(data)):
        if i < period - 1:
            result.append(None)
        else:
            result.append(sum(data[i-period+1:i+1]) / period)
    return result
```

## 3. 可视化函数

```python
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def plot_kline_with_volatility(code: str, week_data: list, day_data: list, save_path: str = None):
    """
    绘制K线图 + 周/日波动叠加图

    参数:
        code: 股票代码
        week_data: 周K线数据
        day_data: 日K线数据
        save_path: 保存路径（可选）
    """
    if not week_data and not day_data:
        print("无数据可绘制")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    # 合并数据
    merged = merge_week_day_data(week_data, day_data)

    # === 上图：价格走势 + 均线 ===
    dates = merged['dates']

    if merged['week_close']:
        ax1.plot(range(len(dates)), merged['week_close'], 'b-', linewidth=2, label='周线', marker='o', markersize=4)
    if merged['day_close']:
        ax1.plot(range(len(dates)), merged['day_close'], 'g-', linewidth=1, alpha=0.7, label='日线')

    # 添加均线
    if merged['day_close']:
        ma5 = calculate_ma(merged['day_close'], 5)
        ma10 = calculate_ma(merged['day_close'], 10)
        ma20 = calculate_ma(merged['day_close'], 20)
        ax1.plot(range(len(dates)), ma5, 'purple', linewidth=1, alpha=0.5, label='MA5')
        ax1.plot(range(len(dates)), ma10, 'orange', linewidth=1, alpha=0.5, label='MA10')
        ax1.plot(range(len(dates)), ma20, 'red', linewidth=1, alpha=0.5, label='MA20')

    ax1.set_title(f'{code} 价格走势', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # === 下图：波动率对比 ===
    if merged['week_volatility']:
        ax2.fill_between(range(len(dates)), merged['week_volatility'], 0, alpha=0.3, color='blue', label='周波动率')
        ax2.plot(range(len(dates)), merged['week_volatility'], 'b-', linewidth=1.5)
    if merged['day_volatility']:
        ax2.fill_between(range(len(dates)), merged['day_volatility'], 0, alpha=0.3, color='green', label='日波动率')
        ax2.plot(range(len(dates)), merged['day_volatility'], 'g-', linewidth=1)

    ax2.set_title('波动率分析', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel('波动率 (%)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    else:
        plt.show()

    plt.close()

def plot_candlestick(code: str, day_data: list, save_path: str = None):
    """
    绘制蜡烛图（日K线）

    参数:
        code: 股票代码
        day_data: 日K线数据列表
        save_path: 保存路径（可选）
    """
    if not day_data:
        print("无数据可绘制")
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    for i, candle in enumerate(day_data[-60:]):  # 最近60根K线
        open_price = float(candle.get('open', 0))
        close_price = float(candle.get('close', 0))
        high_price = float(candle.get('high', 0))
        low_price = float(candle.get('low', 0))

        # 颜色：阳线红色，阴线绿色
        color = 'red' if close_price >= open_price else 'green'

        # 绘制影线（上下影线）
        ax.plot([i, i], [low_price, high_price], color=color, linewidth=1)

        # 绘制实体
        body_height = abs(close_price - open_price)
        bottom = min(open_price, close_price)
        rect = Rectangle((i-0.3, bottom), 0.6, body_height if body_height > 0 else 0.1,
                         facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.set_title(f'{code} 日K线蜡烛图', fontsize=14, fontweight='bold')
    ax.set_xlabel('交易日')
    ax.set_ylabel('价格')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"蜡烛图已保存: {save_path}")
    else:
        plt.show()

    plt.close()
```

## 4. 买卖信号分析

```python
def analyze_buy_sell_signals(code: str, week_data: list, day_data: list) -> dict:
    """
    基于波动率分析买卖信号

    核心逻辑：
    - 波动率低位 + 价格均线金叉 → 买入信号
    - 波动率高位 + 价格均线死叉 → 卖出信号
    - 波动率异常放大 → 关注转折点

    返回:
        {
            'buy_signals': [信号列表],
            'sell_signals': [信号列表],
            'summary': 总结,
            'best_buy_time': 最佳买入时间,
            'best_sell_time': 最佳卖出时间
        }
    """
    result = {
        'buy_signals': [],
        'sell_signals': [],
        'summary': '',
        'best_buy_time': None,
        'best_sell_time': None,
        'current_state': 'neutral'
    }

    if not day_data or len(day_data) < 20:
        result['summary'] = "数据不足，无法分析"
        return result

    # 计算技术指标
    day_df = calculate_volatility(day_data, window=5)

    closes = day_df['close'].tolist()
    volatilities = day_df['volatility'].dropna().tolist()

    if len(closes) < 20 or len(volatilities) < 5:
        result['summary'] = "数据不足，无法分析"
        return result

    # 计算均线
    ma5 = calculate_ma(closes, 5)
    ma10 = calculate_ma(closes, 10)
    ma20 = calculate_ma(closes, 20)

    # === 分析信号 ===

    # 买入信号检测
    for i in range(20, len(closes)):
        # 1. 均线金叉（MA5上穿MA10）
        if ma5[i] and ma5[i-1] and ma10[i] and ma10[i-1]:
            if ma5[i-1] <= ma10[i-1] and ma5[i] > ma10[i]:
                result['buy_signals'].append({
                    'date': day_data[i].get('date', day_data[i].get('date_', f'第{i}天')),
                    'type': 'MA金叉',
                    'price': closes[i],
                    'reason': '短期均线上穿中期均线，形成买入信号'
                })

        # 2. 波动率低位拐头
        if i >= 5:
            recent_vol = volatilities[max(0, len(volatilities)-10):]
            if len(recent_vol) >= 5:
                if recent_vol[-5] > recent_vol[-1] and recent_vol[-1] < np.mean(recent_vol):
                    result['buy_signals'].append({
                        'date': day_data[i].get('date', day_data[i].get('date_', f'第{i}天')),
                        'type': '波动率低位',
                        'price': closes[i],
                        'reason': f'波动率从{recent_vol[-5]:.2f}%下降至{recent_vol[-1]:.2f}%，市场趋于稳定'
                    })

        # 3. 价格触及支撑后反弹
        if i >= 20:
            recent_low = min(closes[i-20:i])
            if closes[i] <= recent_low * 1.02 and closes[i] > recent_low:
                result['buy_signals'].append({
                    'date': day_data[i].get('date', day_data[i].get('date_', f'第{i}天')),
                    'type': '支撑反弹',
                    'price': closes[i],
                    'reason': f'价格在支撑位{round(recent_low, 2)}附近企稳'
                })

    # 卖出信号检测
    for i in range(20, len(closes)):
        # 1. 均线死叉（MA5下穿MA10）
        if ma5[i] and ma5[i-1] and ma10[i] and ma10[i-1]:
            if ma5[i-1] >= ma10[i-1] and ma5[i] < ma10[i]:
                result['sell_signals'].append({
                    'date': day_data[i].get('date', day_data[i].get('date_', f'第{i}天')),
                    'type': 'MA死叉',
                    'price': closes[i],
                    'reason': '短期均线下穿中期均线，形成卖出信号'
                })

        # 2. 波动率异常放大
        if i >= 5 and len(volatilities) >= 5:
            avg_vol = np.mean(volatilities[:-5])
            if volatilities[-1] > avg_vol * 2:
                result['sell_signals'].append({
                    'date': day_data[i].get('date', day_data[i].get('date_', f'第{i}天')),
                    'type': '波动率异常',
                    'price': closes[i],
                    'reason': f'波动率{volatilities[-1]:.2f}%远超均值{avg_vol:.2f}%，注意风险'
                })

        # 3. 价格触及压力后回落
        if i >= 20:
            recent_high = max(closes[i-20:i])
            if closes[i] >= recent_high * 0.98 and closes[i] < recent_high:
                result['sell_signals'].append({
                    'date': day_data[i].get('date', day_data[i].get('date_', f'第{i}天')),
                    'type': '压力回落',
                    'price': closes[i],
                    'reason': f'价格在压力位{round(recent_high, 2)}附近受阻'
                })

    # === 生成综合建议 ===
    latest_vol = volatilities[-1] if volatilities else 0
    avg_vol = np.mean(volatilities) if volatilities else 0

    if latest_vol < avg_vol * 0.5:
        result['current_state'] = 'low_volatility'
        result['summary'] = f'当前波动率{latest_vol:.2f}%处于历史低位，可能酝酿趋势行情'
    elif latest_vol > avg_vol * 1.5:
        result['current_state'] = 'high_volatility'
        result['summary'] = f'当前波动率{latest_vol:.2f}%处于历史高位，注意风险控制'
    else:
        result['current_state'] = 'normal'
        result['summary'] = f'当前波动率{latest_vol:.2f}%处于正常区间'

    # 最佳买卖时机
    if result['buy_signals']:
        # 选择最近的买入信号
        latest_buy = result['buy_signals'][-1]
        result['best_buy_time'] = {
            'date': latest_buy['date'],
            'price': latest_buy['price'],
            'reason': latest_buy['reason']
        }

    if result['sell_signals']:
        latest_sell = result['sell_signals'][-1]
        result['best_sell_time'] = {
            'date': latest_sell['date'],
            'price': latest_sell['price'],
            'reason': latest_sell['reason']
        }

    return result

def generate_trading_report(code: str, week_data: list, day_data: list) -> str:
    """
    生成完整的交易分析报告
    """
    analysis = analyze_buy_sell_signals(code, week_data, day_data)

    report = f"""
{'='*60}
股票 {code} 交易分析报告
{'='*60}

【当前市场状态】
{analysis['summary']}

{'='*60}
买入信号 ({len(analysis['buy_signals'])}个)
{'='*60}
"""
    for signal in analysis['buy_signals'][-5:]:  # 显示最近5个
        report += f"""
• {signal['date']} | 价格: {signal['price']:.2f}
  类型: {signal['type']}
  原因: {signal['reason']}
"""

    report += f"""
{'='*60}
卖出信号 ({len(analysis['sell_signals'])}个)
{'='*60}
"""
    for signal in analysis['sell_signals'][-5:]:
        report += f"""
• {signal['date']} | 价格: {signal['price']:.2f}
  类型: {signal['type']}
  原因: {signal['reason']}
"""

    report += f"""
{'='*60}
交易建议
{'='*60}
"""
    if analysis['best_buy_time']:
        report += f"""
【最佳买入时机】
  时间: {analysis['best_buy_time']['date']}
  价格: {analysis['best_buy_time']['price']:.2f}
  理由: {analysis['best_buy_time']['reason']}
"""

    if analysis['best_sell_time']:
        report += f"""
【最佳卖出时机】
  时间: {analysis['best_sell_time']['date']}
  价格: {analysis['best_sell_time']['price']:.2f}
  理由: {analysis['best_sell_time']['reason']}
"""

    report += f"""
{'='*60}
风险提示
{'='*60}
• 建议止损位: {analysis.get('stop_loss', '需结合更多数据确定')}
• 仓位建议: 不超过总仓位的20%
• 注意事项: 本分析仅供参考，不构成投资建议

{'='*60}
"""

    return report
```

## 5. 使用示例

```python
# 完整使用流程
def run_stock_analysis(code: str):
    """运行完整的股票分析"""

    # 1. 获取数据
    print(f"正在获取 {code} 数据...")
    week_data = get_week_kline(code)
    day_data = get_day_kline(code)

    # 2. 生成分析报告
    report = generate_trading_report(code, week_data, day_data)
    print(report)

    # 3. 绘制图表
    print("正在生成可视化图表...")
    plot_kline_with_volatility(code, week_data, day_data)
    plot_candlestick(code, day_data)

    # 4. 返回分析结果
    analysis = analyze_buy_sell_signals(code, week_data, day_data)
    return analysis

# 示例调用
# analysis = run_stock_analysis("000001")  # 平安银行
```

# 与autostock配合使用

本skill依赖autostock获取数据，建议在agent中同时配置两个skill：

```yaml
skills:
  - name: autostock
    description: 获取股票K线数据
  - name: stock-visualization
    description: 股票可视化与交易信号
    dependencies:
      - autostock
```

# 输出格式

执行后输出：

1. **文字报告**：买卖信号、最佳时机、风险提示
2. **图表**：价格走势图 + 波动率叠加图 + 蜡烛图

