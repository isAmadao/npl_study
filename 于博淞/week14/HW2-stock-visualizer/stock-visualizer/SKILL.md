---
name: stock-visualizer
description: Visualize a stock's daily and weekly volatility together on one chart and generate buy/sell timing suggestions based on volatility magnitude and technical indicators. Use this skill whenever the user asks to analyze, visualize, chart, or evaluate any stock or ticker (e.g. "show me AAPL", "draw a chart for 0700.HK", "should I buy TSLA", "analyze NVDA price action", "what does Microsoft stock look like lately"), or whenever the user mentions stock volatility, candlestick charts, buy/sell timing, entry points, or technical signals. Trigger this skill even when the user does not explicitly say "visualize" — any request that involves looking at a stock's price behavior or asking when to buy/sell counts.
---

# Stock Visualizer

This skill produces a single combined chart that shows a stock's **daily volatility** (candlesticks) and **weekly volatility** (weekly range envelope) overlaid in one figure, and outputs a buy/sell timing recommendation derived from the volatility magnitude plus a small set of classic technical indicators.

## When to use

- User names a ticker symbol or company and asks to chart, visualize, plot, or analyze it.
- User asks "when should I buy/sell X", "is now a good time for X", "what's the best entry for X".
- User mentions volatility, candlesticks, or technical signals on a specific stock.

Always run the script — do not hand-draw or estimate prices from memory.

## How to run

Use `scripts/analyze_stock.py`. It takes a ticker symbol and an optional period, fetches data from Yahoo Finance via `yfinance`, generates the chart as a PNG, and prints a structured recommendation.

```bash
python /path/to/stock-visualizer/scripts/analyze_stock.py <TICKER> [--period 6mo] [--output /mnt/user-data/outputs/<ticker>.png]
```

Valid `--period` values: `1mo`, `3mo`, `6mo` (default), `1y`, `2y`, `5y`, `max`.

The default output path is `/mnt/user-data/outputs/<ticker>_analysis.png`. Always save outputs there so they can be presented to the user.

### Dependencies

Install once per session if missing:

```bash
pip install --break-system-packages yfinance pandas numpy matplotlib mplfinance
```

## What the chart shows

A single figure with shared x-axis containing:

1. **Daily candlesticks** — open/high/low/close per trading day. Green = up day, red = down day. This is the daily volatility.
2. **Weekly range envelope** — a shaded band running from each week's low to each week's high, plotted across all days of that week. This makes weekly volatility visible in the same view as daily.
3. **20-day moving average line** — context for trend.
4. **Bollinger Bands (20-day, 2σ)** — dashed lines for mean-reversion reference.
5. **Buy/Sell markers** — green ▲ at recommended buy points, red ▼ at recommended sell points, plotted at the bar where the signal fired.

The recommendation is also rendered as a text box in the top-left corner of the chart, and printed to stdout as JSON so it can be parsed.

## How the recommendation is computed

The script combines three signal families. Each contributes a score in [−1, +1] where +1 is "strong buy" and −1 is "strong sell". Final score is the weighted average.

| Signal | Weight | Logic |
|---|---|---|
| Bollinger Band position | 0.35 | Price near lower band → buy; near upper band → sell. `score = (mid − price) / (2σ)`, clamped. |
| RSI(14) | 0.30 | RSI < 30 → buy; RSI > 70 → sell. `score = (50 − RSI) / 20`, clamped. |
| Weekly volatility percentile | 0.20 | When current weekly range is in its top quartile vs the last N weeks AND price is in upper half of that range → sell pressure; bottom quartile + lower half → buy opportunity. |
| MA20 vs price trend | 0.15 | Price crossing above MA20 with rising MA → mild buy; crossing below → mild sell. |

**Final mapping:**

| Final score | Recommendation | Confidence |
|---|---|---|
| ≥ +0.5 | STRONG BUY | high |
| +0.2 to +0.5 | BUY | medium |
| −0.2 to +0.2 | HOLD / WATCH | low |
| −0.5 to −0.2 | SELL | medium |
| ≤ −0.5 | STRONG SELL | high |

The script also identifies the **best historical buy and sell points** within the displayed window (local minima/maxima of close price, filtered by weekly volatility being elevated — i.e. real swing points, not noise) and marks them on the chart. These are the "best timing" markers the user asked for.

## After running

1. Show the chart to the user with `present_files`.
2. Summarize the recommendation in prose: current price, the verdict (e.g. "BUY, medium confidence"), the 1–2 strongest contributing signals, and the most recent marked swing point.
3. Always add one line of caveat: this is a technical-pattern analysis, not financial advice, and the user should make their own decisions. Do not embellish with disclaimers beyond this — one line is enough.

## Examples of triggering phrases

- "Chart AAPL for me"
- "画一下特斯拉最近的走势"
- "Should I buy NVDA now?"
- "0700.HK 现在能进吗"
- "Analyze Microsoft stock"
- "Show me Apple's volatility"

All of these should trigger this skill.
