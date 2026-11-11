"""Moving-average crossover back-test (refactor of the original
"Build a simple trading strategy" notebook).

Rule: hold 1 share while MA_fast > MA_slow, else hold nothing.
Profit_t = Close_{t+1} - Close_t when holding.

Fixes vs. the original notebook:
  * pd.DataFrame.from_csv was removed from pandas -> pd.read_csv
  * 'Close1' was used before it was created in the Microsoft section
  * 'wealth' vs 'Wealth' column-name mismatch
  * list comprehensions replaced with vectorised pandas
"""
from __future__ import annotations

import pandas as pd


def ma_crossover(df: pd.DataFrame, fast: int = 10, slow: int = 50, shares: int = 1) -> pd.DataFrame:
    out = df[["Close"]].copy()
    out[f"MA{fast}"] = out["Close"].rolling(fast).mean()
    out[f"MA{slow}"] = out["Close"].rolling(slow).mean()
    out = out.dropna()
    out["Shares"] = (out[f"MA{fast}"] > out[f"MA{slow}"]).astype(int) * shares
    out["Close1"] = out["Close"].shift(-1)
    out["Profit"] = (out["Close1"] - out["Close"]) * out["Shares"]
    out["Wealth"] = out["Profit"].cumsum()
    out["BuyHold"] = (out["Close1"] - out["Close"]).cumsum() * shares
    return out.dropna()


def summary(bt: pd.DataFrame) -> dict:
    return {"Total profit (strategy, $/share)": float(bt["Wealth"].iloc[-1]),
            "Total profit (buy & hold, $/share)": float(bt["BuyHold"].iloc[-1]),
            "Days in market": int(bt["Shares"].gt(0).sum()),
            "Share of days in market": float(bt["Shares"].gt(0).mean())}
