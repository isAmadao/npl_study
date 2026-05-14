#!/usr/bin/env python3
"""
Stock Visualizer — daily + weekly volatility on a single chart, with a
buy/sell timing recommendation.

Usage:
    python analyze_stock.py AAPL
    python analyze_stock.py TSLA --period 1y
    python analyze_stock.py 0700.HK --period 6mo --output /tmp/tencent.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import yfinance as yf


# ---------- indicators ----------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def bollinger(close: pd.Series, period: int = 20, k: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid, mid + k * std, mid - k * std


def weekly_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Compute high/low per ISO week, broadcast back to daily index."""
    weekly = df.resample("W").agg({"High": "max", "Low": "min"})
    weekly.columns = ["WeekHigh", "WeekLow"]
    weekly["WeekRange"] = weekly["WeekHigh"] - weekly["WeekLow"]
    # forward-fill weekly bounds onto each trading day in that week
    daily_week = df.index.to_series().dt.to_period("W").dt.to_timestamp("W")
    out = pd.DataFrame(index=df.index)
    out["WeekHigh"] = daily_week.map(weekly["WeekHigh"])
    out["WeekLow"] = daily_week.map(weekly["WeekLow"])
    out["WeekRange"] = daily_week.map(weekly["WeekRange"])
    return out


# ---------- swing-point detection ----------

def find_swing_points(close: pd.Series, week_range: pd.Series, window: int = 5):
    """Return (buy_idx, sell_idx) — local minima/maxima where weekly volatility
    is in the upper half of its distribution (i.e. real swings, not flat noise).
    """
    vol_threshold = week_range.quantile(0.5)
    buys, sells = [], []
    for i in range(window, len(close) - window):
        win = close.iloc[i - window : i + window + 1]
        if week_range.iloc[i] < vol_threshold:
            continue
        if close.iloc[i] == win.min():
            buys.append(close.index[i])
        elif close.iloc[i] == win.max():
            sells.append(close.index[i])
    return buys, sells


# ---------- recommendation engine ----------

def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def compute_recommendation(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    price = float(last["Close"])

    # Bollinger position
    mid, upper, lower = last["BB_mid"], last["BB_up"], last["BB_low"]
    bb_score = 0.0
    if not np.isnan(mid) and upper > lower:
        bb_score = clamp((mid - price) / ((upper - lower) / 2))

    # RSI
    rsi_val = float(last["RSI"]) if not np.isnan(last["RSI"]) else 50.0
    rsi_score = clamp((50 - rsi_val) / 20)

    # Weekly volatility percentile + position within the week
    week_ranges = df["WeekRange"].dropna()
    pct = (week_ranges <= last["WeekRange"]).mean() if len(week_ranges) else 0.5
    week_pos = 0.5
    if last["WeekHigh"] > last["WeekLow"]:
        week_pos = (price - last["WeekLow"]) / (last["WeekHigh"] - last["WeekLow"])
    vol_score = 0.0
    if pct > 0.75 and week_pos > 0.5:
        vol_score = -(pct - 0.75) * 4 * (week_pos - 0.5) * 2
    elif pct > 0.75 and week_pos < 0.5:
        vol_score = (pct - 0.75) * 4 * (0.5 - week_pos) * 2
    vol_score = clamp(vol_score)

    # MA20 trend
    ma20 = last["BB_mid"]
    ma_score = 0.0
    if not np.isnan(ma20) and len(df) > 25:
        ma_slope = (ma20 - df["BB_mid"].iloc[-5]) / ma20
        if price > ma20 and ma_slope > 0:
            ma_score = 0.3
        elif price < ma20 and ma_slope < 0:
            ma_score = -0.3

    final = 0.35 * bb_score + 0.30 * rsi_score + 0.20 * vol_score + 0.15 * ma_score

    if final >= 0.5:
        verdict, conf = "STRONG BUY", "high"
    elif final >= 0.2:
        verdict, conf = "BUY", "medium"
    elif final <= -0.5:
        verdict, conf = "STRONG SELL", "high"
    elif final <= -0.2:
        verdict, conf = "SELL", "medium"
    else:
        verdict, conf = "HOLD / WATCH", "low"

    return {
        "price": round(price, 2),
        "verdict": verdict,
        "confidence": conf,
        "final_score": round(final, 3),
        "signals": {
            "bollinger": round(bb_score, 3),
            "rsi": round(rsi_score, 3),
            "rsi_value": round(rsi_val, 1),
            "weekly_volatility": round(vol_score, 3),
            "weekly_range_percentile": round(pct, 2),
            "ma20_trend": round(ma_score, 3),
        },
    }


# ---------- charting ----------

def draw_chart(ticker, df, rec, buys, sells, output_path):
    fig, ax = plt.subplots(figsize=(14, 8))

    # Weekly range envelope — shaded band
    ax.fill_between(
        df.index, df["WeekLow"], df["WeekHigh"],
        color="#4a90e2", alpha=0.12, label="Weekly range (weekly volatility)"
    )
    ax.plot(df.index, df["WeekHigh"], color="#4a90e2", alpha=0.4, lw=0.8)
    ax.plot(df.index, df["WeekLow"], color="#4a90e2", alpha=0.4, lw=0.8)

    # Daily candlesticks — manual draw to control overlay order
    width = 0.6
    for date, row in df.iterrows():
        o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
        color = "#26a69a" if c >= o else "#ef5350"
        x = mdates.date2num(date)
        ax.vlines(x, l, h, color=color, lw=0.8)
        ax.add_patch(Rectangle(
            (x - width / 2, min(o, c)),
            width, max(abs(c - o), 0.001),
            facecolor=color, edgecolor=color
        ))

    # Bollinger + MA20
    ax.plot(df.index, df["BB_mid"], color="#666", lw=1.0, label="MA20")
    ax.plot(df.index, df["BB_up"], color="#999", lw=0.8, ls="--", label="Bollinger ±2σ")
    ax.plot(df.index, df["BB_low"], color="#999", lw=0.8, ls="--")

    # Buy / sell markers
    if buys:
        ax.scatter(buys, df.loc[buys, "Close"] * 0.985,
                   marker="^", s=140, color="#2ecc71", edgecolor="black",
                   linewidths=0.8, zorder=5, label="Best buy timing")
    if sells:
        ax.scatter(sells, df.loc[sells, "Close"] * 1.015,
                   marker="v", s=140, color="#e74c3c", edgecolor="black",
                   linewidths=0.8, zorder=5, label="Best sell timing")

    # Recommendation box
    verdict_color = {
        "STRONG BUY": "#1b7a3e", "BUY": "#2ecc71",
        "HOLD / WATCH": "#888",
        "SELL": "#e74c3c", "STRONG SELL": "#8b1a1a",
    }[rec["verdict"]]
    txt = (
        f"{ticker.upper()}   ${rec['price']}\n"
        f"{rec['verdict']}  ({rec['confidence']} confidence)\n"
        f"Score: {rec['final_score']:+.2f}    RSI: {rec['signals']['rsi_value']}"
    )
    ax.text(0.012, 0.975, txt, transform=ax.transAxes,
            fontsize=11, va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor=verdict_color,
                      edgecolor="black", alpha=0.85),
            color="white", weight="bold")

    ax.set_title(f"{ticker.upper()} — daily candles + weekly volatility envelope",
                 fontsize=13, weight="bold")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------- main ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("ticker", help="e.g. AAPL, TSLA, 0700.HK, 600519.SS")
    p.add_argument("--period", default="6mo",
                   choices=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"])
    p.add_argument("--output", default=None)
    args = p.parse_args()

    ticker = args.ticker.upper()
    output = args.output or f"/mnt/user-data/outputs/{ticker.replace('.', '_')}_analysis.png"
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    df = yf.download(ticker, period=args.period, progress=False, auto_adjust=False)
    if df.empty:
        print(json.dumps({"error": f"No data returned for {ticker}"}))
        sys.exit(1)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["RSI"] = rsi(df["Close"])
    df["BB_mid"], df["BB_up"], df["BB_low"] = bollinger(df["Close"])
    wk = weekly_ranges(df)
    df = df.join(wk)

    buys, sells = find_swing_points(df["Close"], df["WeekRange"])
    rec = compute_recommendation(df)
    draw_chart(ticker, df, rec, buys, sells, output)

    result = {
        "ticker": ticker,
        "period": args.period,
        "chart": output,
        "recommendation": rec,
        "historical_buy_points": [str(d.date()) for d in buys[-5:]],
        "historical_sell_points": [str(d.date()) for d in sells[-5:]],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
