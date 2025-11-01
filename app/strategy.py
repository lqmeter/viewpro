"""Backtesting utilities for Pine-based strategies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: int
    final_value: float
    return_pct: float
    max_drawdown: float


def backtest_signals(
    data: pd.DataFrame,
    long_signal: pd.Series | None,
    short_signal: pd.Series | None,
    initial_capital: float = 10_000.0,
) -> BacktestResult:
    """Run a simple long-only backtest given long/short boolean signals."""
    closes = data["close"].reset_index(drop=True)
    if long_signal is None:
        long_signal = pd.Series(False, index=closes.index)
    else:
        long_signal = long_signal.reindex(closes.index, fill_value=False)

    if short_signal is None:
        short_signal = pd.Series(False, index=closes.index)
    else:
        short_signal = short_signal.reindex(closes.index, fill_value=False)

    position = 0.0
    cash = initial_capital
    equity_values: List[float] = []
    trades = 0

    for idx, price in closes.items():
        if long_signal.iloc[idx] and position == 0:
            position = cash / price
            cash = 0.0
            trades += 1
        elif short_signal.iloc[idx] and position > 0:
            cash = position * price
            position = 0.0
            trades += 1
        equity = cash + position * price
        equity_values.append(equity)

    if position > 0:
        cash = position * closes.iloc[-1]
        position = 0.0
    final_value = cash

    equity_curve = pd.Series(equity_values, index=closes.index)
    return_pct = (final_value / initial_capital - 1) * 100
    max_drawdown = _max_drawdown(equity_curve)

    return BacktestResult(
        equity_curve=equity_curve,
        trades=trades,
        final_value=final_value,
        return_pct=return_pct,
        max_drawdown=max_drawdown,
    )


def _max_drawdown(equity_curve: pd.Series) -> float:
    cumulative_max = equity_curve.cummax()
    drawdown = (equity_curve - cumulative_max) / cumulative_max
    return float(drawdown.min() * 100)
