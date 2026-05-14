"""
作业2: 股票可视化分析Skill
功能：
  1. 获取股票历史数据
  2. 绘制周波动与日波动对比图
  3. 基于波动幅度给出买入/卖出最佳时间建议
参考：09_多Agent/03_skills.py 中的skill模式
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 股票数据获取 ====================

def fetch_stock_data(stock_code: str, period: str = "3mo") -> pd.DataFrame:
    """
    获取股票历史数据
    支持A股代码格式: 000001.SZ, 600000.SH
    也支持美股代码: AAPL, MSFT
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(stock_code)
        df = ticker.history(period=period)
        if df.empty:
            raise ValueError(f"未获取到股票 {stock_code} 的数据")
        return df
    except ImportError:
        print("请安装 yfinance: pip install yfinance")
        raise
    except Exception as e:
        raise ValueError(f"获取股票数据失败: {e}")


# ==================== 波动计算 ====================

def calculate_daily_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """计算日波动率 (最高-最低)/开盘价 * 100%"""
    df['daily_volatility'] = (df['High'] - df['Low']) / df['Open'] * 100
    df['daily_change'] = (df['Close'] - df['Open']) / df['Open'] * 100
    return df


def calculate_weekly_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """计算周波动率"""
    weekly = df.resample('W').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    weekly['weekly_volatility'] = (weekly['High'] - weekly['Low']) / weekly['Open'] * 100
    weekly['weekly_change'] = (weekly['Close'] - weekly['Open']) / weekly['Open'] * 100
    return weekly


# ==================== 可视化 ====================

def plot_stock_volatility(stock_code: str, df: pd.DataFrame, weekly: pd.DataFrame):
    """
    绘制股票周波动与日波动对比图
    包含：价格走势、日波动率、周波动率
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # 1. 价格走势图
    ax1 = axes[0]
    ax1.plot(df.index, df['Close'], label='收盘价', color='#2196F3', linewidth=1.5)
    ax1.fill_between(df.index, df['Low'], df['High'], alpha=0.15, color='#2196F3', label='日内区间')
    ax1.set_ylabel('价格')
    ax1.set_title(f'{stock_code} 股票价格与波动分析', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 2. 日波动率
    ax2 = axes[1]
    colors_daily = ['#4CAF50' if v >= 0 else '#F44336' for v in df['daily_change']]
    ax2.bar(df.index, df['daily_volatility'], color=colors_daily, alpha=0.6, label='日波动率')
    ax2.axhline(y=np.mean(df['daily_volatility']), color='orange', linestyle='--',
                label=f'平均日波动: {np.mean(df["daily_volatility"]):.2f}%')
    ax2.set_ylabel('日波动率 (%)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # 3. 周波动率
    ax3 = axes[3] if len(axes) > 3 else axes[2]
    colors_weekly = ['#4CAF50' if v >= 0 else '#F44336' for v in weekly['weekly_change']]
    ax3.bar(weekly.index, weekly['weekly_volatility'], color=colors_weekly, alpha=0.7,
            width=4, label='周波动率')
    ax3.axhline(y=np.mean(weekly['weekly_volatility']), color='orange', linestyle='--',
                label=f'平均周波动: {np.mean(weekly["weekly_volatility"]):.2f}%')
    ax3.set_ylabel('周波动率 (%)')
    ax3.set_xlabel('日期')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    # 日期格式
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax3.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)

    plt.tight_layout()

    # 保存图片
    output_dir = os.path.join(os.path.dirname(__file__), "stock_charts")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{stock_code}_volatility.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return filepath


# ==================== 买卖建议分析 ====================

def analyze_timing(df: pd.DataFrame, weekly: pd.DataFrame) -> str:
    """
    基于波动幅度分析，给出买入/卖出的最佳时间建议
    策略：
    - 低波动 + 下跌 -> 买入信号（价格稳定在低位）
    - 高波动 + 上涨 -> 卖出信号（价格波动剧烈可能见顶）
    - 连续低波动 -> 横盘整理，观望
    """
    recent_days = df.tail(10)
    recent_weeks = weekly.tail(4)

    avg_daily_vol = np.mean(df['daily_volatility'])
    avg_weekly_vol = np.mean(weekly['weekly_volatility'])
    recent_avg_daily_vol = np.mean(recent_days['daily_volatility'])
    recent_avg_weekly_vol = np.mean(recent_weeks['weekly_volatility'])

    current_price = df['Close'].iloc[-1]
    price_change_5d = (df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5] * 100 if len(df) >= 5 else 0
    price_change_20d = (df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20] * 100 if len(df) >= 20 else 0

    signals = []
    score = 0

    # 波动率分析
    if recent_avg_daily_vol < avg_daily_vol * 0.7:
        signals.append("日波动率显著降低，市场情绪趋于平稳")
        if price_change_20d < -5:
            signals.append(">> 建议：低位缩量企稳，可考虑分批买入")
            score += 2
        elif price_change_20d > 10:
            signals.append(">> 建议：高位缩量，可能见顶，考虑卖出")
            score -= 2
        else:
            signals.append(">> 建议：横盘整理，暂时观望")
    elif recent_avg_daily_vol > avg_daily_vol * 1.5:
        signals.append("日波动率显著放大，市场情绪激烈")
        if price_change_5d > 5:
            signals.append(">> 建议：放量上涨，短期可能见顶，可考虑分批卖出")
            score -= 1
        elif price_change_5d < -5:
            signals.append(">> 建议：放量下跌，可能是恐慌性抛售，等待企稳后再考虑买入")
            score -= 1
        else:
            signals.append(">> 建议：波动剧烈方向不明，观望为主")
    else:
        signals.append("波动率处于正常水平")

    # 周趋势分析
    weekly_trend = recent_weeks['weekly_change'].values
    if len(weekly_trend) >= 3:
        if all(w > 0 for w in weekly_trend[-3:]):
            signals.append("连续3周上涨，处于上升趋势")
            if recent_avg_weekly_vol > avg_weekly_vol:
                signals.append(">> 建议：上升趋势中但波动加大，可持有但注意回调风险")
            else:
                signals.append(">> 建议：上升趋势健康，可继续持有")
            score += 1
        elif all(w < 0 for w in weekly_trend[-3:]):
            signals.append("连续3周下跌，处于下降趋势")
            if recent_avg_weekly_vol > avg_weekly_vol:
                signals.append(">> 建议：下跌趋势中放量，可能加速下跌，建议卖出")
                score -= 2
            else:
                signals.append(">> 建议：下跌趋势中缩量，可能接近底部，观望")
                score += 0

    # 综合建议
    if score >= 2:
        action = "【买入】建议：当前是较好的买入时机"
    elif score <= -2:
        action = "【卖出】建议：当前是较好的卖出时机"
    elif score > 0:
        action = "【持有】建议：当前适合继续持有"
    elif score < 0:
        action = "【减仓】建议：建议适当减仓控制风险"
    else:
        action = "【观望】建议：市场方向不明，建议观望"

    # 最佳交易时间建议
    best_time = ""
    hour_now = datetime.now().hour
    if 9 <= hour_now < 11:
        best_time = "建议在开盘后30分钟到1小时（10:00-10:30）观察趋势后再做决定"
    elif 11 <= hour_now < 13:
        best_time = "午盘开盘后（13:00-13:30）通常会有方向性选择，可关注"
    elif 13 <= hour_now < 15:
        best_time = "尾盘（14:30-15:00）是较好的交易时间窗口"
    else:
        best_time = "建议在下一个交易日的早盘（9:30-10:00）观察开盘情况"

    # 构建分析报告
    report = f"""
{'='*50}
股票波动分析报告
{'='*50}
当前价格: {current_price:.2f}
近5日涨跌: {price_change_5d:+.2f}%
近20日涨跌: {price_change_20d:+.2f}%

【波动率统计】
  平均日波动率: {avg_daily_vol:.2f}%
  近10日平均日波动: {recent_avg_daily_vol:.2f}%
  平均周波动率: {avg_weekly_vol:.2f}%
  近4周平均周波动: {recent_avg_weekly_vol:.2f}%

【信号分析】
{chr(10).join(f'  • {s}' for s in signals)}

【综合建议】
  {action}
  {best_time}
{'='*50}
"""
    return report


# ==================== Skill 定义 ====================

@tool
def stock_visualization_skill(stock_code: str, period: str = "3mo") -> str:
    """
    股票可视化分析Skill
    功能：获取股票数据、绘制周/日波动图、给出买卖建议

    Args:
        stock_code: 股票代码，如 000001.SZ（中国平安）、600519.SH（贵州茅台）、AAPL（苹果）
        period: 数据周期，如 1mo（1个月）、3mo（3个月）、6mo（6个月）、1y（1年）
    """
    print(f"\n正在分析股票: {stock_code}, 周期: {period}")

    # 1. 获取数据
    df = fetch_stock_data(stock_code, period)
    print(f"获取到 {len(df)} 条日线数据")

    # 2. 计算波动率
    df = calculate_daily_volatility(df)
    weekly = calculate_weekly_volatility(df)
    print("波动率计算完成")

    # 3. 绘制图表
    chart_path = plot_stock_volatility(stock_code, df, weekly)
    print(f"波动图已保存: {chart_path}")

    # 4. 分析买卖建议
    report = analyze_timing(df, weekly)
    print(report)

    return f"分析完成！图表已保存至: {chart_path}\n\n{report}"


# ==================== 独立运行 ====================

def main():
    """独立运行模式 - 直接分析股票"""
    print("=" * 50)
    print("股票可视化分析工具")
    print("=" * 50)

    stock_code = input("请输入股票代码 (如: 600519.SH, 000001.SZ, AAPL): ").strip()
    period = input("请输入分析周期 (1mo/3mo/6mo/1y, 默认3mo): ").strip() or "3mo"

    result = stock_visualization_skill.invoke({
        "stock_code": stock_code,
        "period": period
    })
    print(result)


# ==================== Agent模式运行（参考教程） ====================

def run_with_agent():
    """通过LangChain Agent调用Skill（参考03_Agent基础使用.py）"""
    llm = ChatOpenAI(
        model="qwen-flash",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="sk-4fedee4ece6541d3b17a7173f0b3c16f"
    )

    agent = create_agent(
        model=llm,
        tools=[stock_visualization_skill],
        system_prompt=(
            "你是一个股票分析助手。当用户询问股票分析时：\n"
            "1. 使用 stock_visualization_skill 工具获取股票数据并生成波动图\n"
            "2. 根据分析结果向用户解释图表含义和买卖建议\n"
            "3. 用中文回答，清晰说明波动情况和交易建议"
        ),
    )

    print("\n" + "=" * 50)
    print("Agent模式 - 股票分析助手")
    print("=" * 50)
    print("输入 'quit' 退出")

    while True:
        question = input("\n请输入你的问题: ").strip()
        if question.lower() == "quit":
            break
        if not question:
            continue

        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        print(f"\n回答: {result['messages'][-1].content}")


if __name__ == "__main__":
    print("选择运行模式:")
    print("1. 独立运行（直接分析股票）")
    print("2. Agent模式（通过对话分析）")
    choice = input("请选择 (1/2): ").strip()

    if choice == "2":
        run_with_agent()
    else:
        main()
