import argparse
import traceback
from typing import Annotated, Optional, Dict
import requests
import pandas as pd

# 导入 Bokeh 相关组件
from bokeh.plotting import figure, output_file, save, show
from bokeh.models import ColumnDataSource, HoverTool, LinearAxis, Range1d, DatetimeTickFormatter

TOKEN = "zgaLG8unUPr"


def get_stock_day_kline(
        code: Annotated[str, "股票代码"],
        startDate: Annotated[Optional[str], "开始时间"] = None,
        endDate: Annotated[Optional[str], "结束时间"] = None,
        type: Annotated[int, "0不复权,1前复权,2后复权"] = 0,
) -> Dict:
    """获取日K数据"""
    url = f"https://api.autostock.cn/v1/stock/kline/day?token={TOKEN}"
    try:
        payload = {"code": code, "startDate": startDate, "endDate": endDate, "type": type}
        response = requests.get(url, params=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}


def analyze_and_plot_bokeh(api_response: Dict, code: str, save_path: str):
    """使用 Bokeh 进行分析、双Y轴绘图并输出交互式HTML"""
    if not api_response or api_response.get('code') != 200 or 'data' not in api_response:
        print("错误：API 未返回有效数据。")
        return

    # 1. 数据预处理
    df = pd.DataFrame(api_response['data'])
    df = df.iloc[:, :6]
    df.columns = ['date', 'open', 'close', 'high', 'low', 'volume']
    df['date'] = pd.to_datetime(df['date'])
    for col in ['open', 'close', 'high', 'low', 'volume']:
        df[col] = pd.to_numeric(df[col])
    df = df.sort_values('date').reset_index(drop=True)

    # 2. 计算波动率指标
    df['daily_vol'] = (df['high'] - df['low']) / df['close']
    df['weekly_vol'] = df['close'].rolling(window=5).std() / df['close'].rolling(window=5).mean()
    df['weekly_vol'] = df['weekly_vol'].bfill()

    # 3. 制定买卖建议策略 (基于滚动分位数)
    buy_thresh = df['weekly_vol'].quantile(0.20)
    sell_thresh = df['weekly_vol'].quantile(0.80)

    # 标记信号价格，不触发信号的设为 NaN 绘图时会自动忽略
    df['buy_price'] = df.apply(
        lambda r: r['close'] if (r['weekly_vol'] <= buy_thresh and r['close'] > r['open']) else None, axis=1)
    df['sell_price'] = df.apply(
        lambda r: r['close'] if (r['weekly_vol'] >= sell_thresh and r['close'] < r['open']) else None, axis=1)

    # 用于 HoverTool 悬停显示的字符串日期
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')

    # 4. 创建 Bokeh ColumnDataSource
    source = ColumnDataSource(df)

    # 5. 初始化画布并配置文件字体
    CHINESE_FONT = "Microsoft YaHei, SimHei, STHeiti, sans-serif"

    p = figure(
        title=f"股票 {code} 日/周波动率分析与买卖建议",
        x_axis_type="datetime",
        width=1200, height=650,
        tools="pan,box_zoom,wheel_zoom,reset,save",
        active_drag="pan"
    )

    # 标题字体美化
    p.title.text_font = CHINESE_FONT
    p.title.text_font_size = "14pt"

    # 设置主 X 轴和主 Y 轴 (股价)
    p.xaxis.axis_label = "时间轴"
    p.xaxis.axis_label_text_font = CHINESE_FONT
    p.yaxis.axis_label = "股票收盘价"
    p.yaxis.axis_label_text_font = CHINESE_FONT

    # 限制主Y轴（价格）的显示范围
    p.y_range = Range1d(start=df['close'].min() * 0.95, end=df['close'].max() * 1.05)

    # 6. 绘制主轴数据（收盘价折线、买卖标记点）
    price_line = p.line(x='date', y='close', source=source, color="#3498DB", line_width=2, legend_label="收盘价")

    # 绿色正三角表示买入
    buy_scatter = p.triangle(x='date', y='buy_price', source=source, color="green", size=12,
                             legend_label="最佳买入建议点")
    # 红色倒三角表示卖出
    sell_scatter = p.inverted_triangle(x='date', y='sell_price', source=source, color="red", size=12,
                                       legend_label="最佳卖出建议点")

    # 7. 创建并配置副 Y 轴 (波动率)
    max_vol = max(df['daily_vol'].max(), df['weekly_vol'].max()) * 1.1
    p.extra_y_ranges = {"vol_range": Range1d(start=0, end=max_vol)}

    vol_axis = LinearAxis(y_range_name="vol_range", axis_label="波动率指数")
    vol_axis.axis_label_text_font = CHINESE_FONT
    p.add_layout(vol_axis, 'right')

    # 绘制副轴数据（日波动柱状图、周波动折线）
    # width=86400000 毫秒代表 1 天的宽度
    p.vbar(x='date', top='daily_vol', width=86400000 * 0.6, source=source, y_range_name="vol_range",
           color="#AED6F1", alpha=0.4, legend_label="日波动率 (最高-最低)/收盘")

    p.line(x='date', y='weekly_vol', source=source, y_range_name="vol_range",
           color="#E67E22", line_width=2.5, legend_label="周波动率 (5日滚动标准差)")

    # 8. 丰富交互：配置鼠标悬停提示 (HoverTool)
    hover = HoverTool(
        renderers=[price_line],  # 绑定在价格线上，避免鼠标乱晃时弹窗乱跳
        tooltips=[
            ("日期", "@date_str"),
            ("收盘价", "@close{0.00}"),
            ("日波动率", "@daily_vol{0.0000}"),
            ("周波动率", "@weekly_vol{0.0000}"),
        ]
    )
    p.add_tools(hover)

    # 9. 美化日期格式与图例
    p.xaxis.formatter = DatetimeTickFormatter(days="%Y-%m-%d", months="%Y-%m", years="%Y")
    p.legend.location = "top_left"
    p.legend.label_text_font = CHINESE_FONT
    p.legend.click_policy = "hide"  # 点击图例可以隐藏/显示对应线条

    # 10. 保存并展示
    # 确保保存路径为 .html 格式
    html_path = save_path.replace('.png', '.html') if save_path.endswith('.png') else save_path
    output_file(html_path)
    save(p)
    print(f"分析图表已成功保存为交互式网页: {html_path}")
    # show(p)

    # 11. 终端打印建议
    print("\n--- 【最佳买卖时间建议】 ---")
    buys = df[df['buy_price'].notna()]
    sells = df[df['sell_price'].notna()]
    print(f"💡 建议【买入】日期: {buys['date_str'].tolist()[-3:] if not buys.empty else '暂无'}")
    print(f"🚨 建议【卖出】日期: {sells['date_str'].tolist()[-3:] if not sells.empty else '暂无'}")
    print("----------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bokeh 股票画图与分析工具")
    parser.add_argument("--code", type=str, required=True, help="股票代码")
    parser.add_argument("--start", type=str, required=True, help="开始时间")
    parser.add_argument("--end", type=str, required=True, help="结束时间")
    # 默认将路径变更为 html
    parser.add_argument("--plot_save_path", type=str, default="./stock_analysis.html", help="画图保存路径")

    args = parser.parse_args()

    res = get_stock_day_kline(args.code, args.start, args.end)
    analyze_and_plot_bokeh(res, args.code, args.plot_save_path)