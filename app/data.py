"""Data utilities for fetching cryptocurrency OHLCV data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import yfinance as yf


@dataclass
class OHLCVRequest:
    """Parameters for requesting historical price data."""

    symbol: str = "BTC-USD"
    interval: str = "1h"
    lookback_days: int = 14

    def period(self) -> str:
        """Return the minimal Yahoo Finance period that satisfies the request."""
        if self.lookback_days <= 7:
            return "7d"
        if self.lookback_days <= 30:
            return "1mo"
        if self.lookback_days <= 60:
            return "2mo"
        if self.lookback_days <= 90:
            return "3mo"
        if self.lookback_days <= 180:
            return "6mo"
        return "1y"


def fetch_ohlcv(request: OHLCVRequest) -> pd.DataFrame:
    """Fetch OHLCV data for the given request using Yahoo Finance."""
    ticker = yf.Ticker(request.symbol)
    history = ticker.history(period=request.period(), interval=request.interval)

    if history.empty:
        raise ValueError("No data returned for the given parameters.")

    history = history.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    history.index.name = "timestamp"

    index_tz = history.index.tz
    if index_tz is not None:
        end_time = pd.Timestamp.now(tz=index_tz)
    else:
        end_time = pd.Timestamp.utcnow()

    start_time = end_time - pd.Timedelta(days=request.lookback_days)
    trimmed = history.loc[history.index >= start_time]

    if trimmed.empty:
        trimmed = history.tail(request.lookback_days * 24)

    return trimmed.reset_index()
