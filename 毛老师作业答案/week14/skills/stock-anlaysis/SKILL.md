---
name: 股票可视化和买入卖出分析
description: 对股票的可视化功能，对于股票的周波动、日波动绘制在一个图中，并基于大小给出一个买入卖出的最佳时间建议；
---

## 分析逻辑

1. 首先通过用户输入的描述，搜索股票名称，获取股票代码，并返回给用户。
2. 通过的股票代码获取得到股票日k线（最近半年），进行画图 和 波动分析。

## 使用逻辑

```commandline
python script/search_stock.py --name=茅台

输出结果为：
{'code': 200, 'message': '操作成功', 'traceId': 'MBqYCQ89cs_Qyg', 'data': [['sh600519', '贵州茅台']]}
```

```commandline
python script/plot_stock_and_analysis.py --code=sh600519 --start=2026-01-01 --end=2026-05-01 --plot_save_path=./stock_analysis.html

python script/plot_stock_and_analysis.py --code=sh600519 --start=2026-01-01 --end=2026-05-10 --plot_save_path=./plot_stock_kline.html
BokehDeprecationWarning: 'triangle() method' was deprecated in Bokeh 3.4.0 and will be removed, use "scatter(marker='triangle', ...) instead" instead.
BokehDeprecationWarning: 'inverted_triangle() method' was deprecated in Bokeh 3.4.0 and will be removed, use "scatter(marker='inverted_triangle', ...) instead" instead.
分析图表已成功保存为交互式网页: ./plot_stock_kline.html

--- 【最佳买卖时间建议】 ---
💡 建议【买入】日期: ['2026-03-27', '2026-03-30', '2026-04-23']
🚨 建议【卖出】日期: ['2026-03-23', '2026-03-24', '2026-04-30']
----------------------------
```

## 依赖环境
```commandline
bokeh
pandas
```