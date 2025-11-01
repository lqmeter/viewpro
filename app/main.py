"""Streamlit app for a crypto quantitative trading workstation."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .data import OHLCVRequest, fetch_ohlcv
from .pine_executor import PineExecutionError, PineInterpreter
from .strategy import backtest_signals

st.set_page_config(page_title="Crypto Quant Lab", layout="wide")


@st.cache_data(show_spinner=False)
def load_price_data(symbol: str, interval: str, lookback: int) -> pd.DataFrame:
    request = OHLCVRequest(symbol=symbol, interval=interval, lookback_days=lookback)
    return fetch_ohlcv(request)


def render_chart(data: pd.DataFrame, plots, long_signal, short_signal) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=data["timestamp"],
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="Price",
        )
    )

    overlay_plots = [plot for plot in plots if plot.overlay]
    panel_plots = [plot for plot in plots if not plot.overlay]

    for plot in overlay_plots:
        fig.add_trace(
            go.Scatter(
                x=data["timestamp"],
                y=plot.series,
                name=plot.title,
                line=dict(color=plot.color, width=plot.linewidth),
            )
        )

    if long_signal is not None and long_signal.any():
        fig.add_trace(
            go.Scatter(
                x=data["timestamp"][long_signal],
                y=data["close"][long_signal],
                mode="markers",
                marker=dict(color="#00e676", symbol="triangle-up", size=12),
                name="Long Entry",
            )
        )

    if short_signal is not None and short_signal.any():
        fig.add_trace(
            go.Scatter(
                x=data["timestamp"][short_signal],
                y=data["close"][short_signal],
                mode="markers",
                marker=dict(color="#ff1744", symbol="triangle-down", size=12),
                name="Exit",
            )
        )

    fig.update_layout(height=550, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    for plot in panel_plots:
        panel_fig = go.Figure()
        panel_fig.add_trace(
            go.Scatter(
                x=data["timestamp"],
                y=plot.series,
                name=plot.title,
                line=dict(color=plot.color, width=plot.linewidth),
            )
        )
        panel_fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(panel_fig, use_container_width=True)


def render_backtest(result) -> None:
    st.subheader("策略回测结果")
    col1, col2, col3 = st.columns(3)
    col1.metric("最终资产", f"${result.final_value:,.2f}")
    col2.metric("总收益率", f"{result.return_pct:.2f}%")
    col3.metric("最大回撤", f"{result.max_drawdown:.2f}%")

    st.write(f"交易次数: **{result.trades}**")
    equity_fig = go.Figure()
    equity_fig.add_trace(
        go.Scatter(
            x=result.equity_curve.index,
            y=result.equity_curve.values,
            name="Equity",
            line=dict(color="#42a5f5", width=2),
        )
    )
    equity_fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(equity_fig, use_container_width=True)


def default_script() -> str:
    return """// 基于双均线的示例策略
indicator(title=\"双均线策略\", overlay=True)
fastLength = input.int(20, title=\"快线周期\")
slowLength = input.int(60, title=\"慢线周期\")
fastMA = sma(close, fastLength)
slowMA = sma(close, slowLength)
plot(fastMA, title=\"快线\", color=color.orange, linewidth=2)
plot(slowMA, title=\"慢线\", color=color.blue, linewidth=2)
long = crossover(fastMA, slowMA)
short = crossunder(fastMA, slowMA)
"""


def main() -> None:
    st.title("加密货币量化交易工作站")

    st.sidebar.header("行情设置")
    symbol = st.sidebar.selectbox("交易对", ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"])
    interval = st.sidebar.selectbox("时间周期", ["15m", "30m", "1h", "4h", "1d"], index=2)
    lookback = st.sidebar.slider("回看天数", 5, 120, 30)

    st.sidebar.header("策略脚本")
    script = st.sidebar.text_area("Pine 脚本", default_script(), height=320)

    try:
        data = load_price_data(symbol, interval, lookback)
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"数据获取失败: {exc}")
        return

    interpreter = PineInterpreter(data)
    try:
        pine_result = interpreter.execute(script)
    except PineExecutionError as exc:
        st.error(f"脚本执行错误: {exc}")
        return

    long_signal = pine_result.context.get("long")
    short_signal = pine_result.context.get("short")
    if isinstance(long_signal, pd.Series):
        long_signal = long_signal.fillna(False)
    else:
        long_signal = None
    if isinstance(short_signal, pd.Series):
        short_signal = short_signal.fillna(False)
    else:
        short_signal = None

    st.subheader(f"{pine_result.indicator_name} - {symbol}")
    render_chart(data, pine_result.plots, long_signal, short_signal)

    if long_signal is not None or short_signal is not None:
        result = backtest_signals(data, long_signal, short_signal)
        render_backtest(result)
    else:
        st.info("在脚本中定义 long/short 序列以启用策略回测。")


if __name__ == "__main__":
    main()
