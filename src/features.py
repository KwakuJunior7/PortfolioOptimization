"""Feature engineering + targets for next-day forecasting.

All features are scale-free (returns, ratios, oscillators) so a model trained
on 2020 prices still applies to 2026 prices. Targets are next-day *returns*;
forecast.py converts them back to price levels.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .indicators import add_indicators

FEATURES = [
    "ret_1", "ret_5", "ret_10", "ret_20", "vol_20", "gap", "hl_range",
    "px_sma20", "px_sma50", "ema_spread", "macd_hist_n", "RSI14",
    "Stoch_K", "Stoch_D", "BB_PctB", "BB_Width", "atr_n", "volume_z", "mkt_ret_1",
]
TARGETS = ["y_open_ret", "y_close_ret", "y_direction"]


def build_features(df: pd.DataFrame, market: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    x = add_indicators(df)
    c = x["Close"]
    r = c.pct_change()
    x["ret_1"] = r
    for n in (5, 10, 20):
        x[f"ret_{n}"] = c.pct_change(n)
    x["vol_20"] = r.rolling(20).std()
    x["gap"] = x["Open"] / c.shift(1) - 1
    x["hl_range"] = (x["High"] - x["Low"]) / c
    x["px_sma20"] = c / x["SMA20"] - 1
    x["px_sma50"] = c / x["SMA50"] - 1
    x["ema_spread"] = x["EMA12"] / x["EMA26"] - 1
    x["macd_hist_n"] = x["MACD_Hist"] / c
    x["atr_n"] = x["ATR14"] / c
    lv = np.log(x["Volume"].replace(0, np.nan))
    x["volume_z"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()
    if market is not None:
        x["mkt_ret_1"] = market["Close"].pct_change().reindex(x.index)
    else:
        x["mkt_ret_1"] = 0.0
    # targets (t+1) - the last row has no target and is used for the live signal
    x["y_open_ret"] = x["Open"].shift(-1) / c - 1
    x["y_close_ret"] = c.shift(-1) / c - 1
    x["y_direction"] = (c.shift(-1) > c).astype(float).where(c.shift(-1).notna())
    return x.dropna(subset=FEATURES)


def split_xy(frame: pd.DataFrame):
    """(labelled rows, latest unlabelled row)."""
    labelled = frame.dropna(subset=TARGETS)
    latest = frame.iloc[[-1]]
    return labelled, latest
