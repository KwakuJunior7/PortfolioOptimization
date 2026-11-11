"""Module 2 - Technical indicators (trend, momentum, volatility)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"MACD": line, "MACD_Signal": sig, "MACD_Hist": line - sig})


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(loss.notna())


def stochastic(df: pd.DataFrame, n: int = 14, d: int = 3) -> pd.DataFrame:
    lo, hi = df["Low"].rolling(n).min(), df["High"].rolling(n).max()
    k = 100 * (df["Close"] - lo) / (hi - lo).replace(0, np.nan)
    return pd.DataFrame({"Stoch_K": k, "Stoch_D": k.rolling(d).mean()})


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid, sd = sma(close, n), close.rolling(n).std()
    up, lo = mid + k * sd, mid - k * sd
    return pd.DataFrame({"BB_Mid": mid, "BB_Upper": up, "BB_Lower": lo,
                         "BB_Width": (up - lo) / mid, "BB_PctB": (close - lo) / (up - lo)})


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev).abs(),
                    (df["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of an OHLCV frame with all indicator columns appended."""
    out = df.copy()
    c = out["Close"]
    out["SMA20"], out["SMA50"] = sma(c, 20), sma(c, 50)
    out["EMA12"], out["EMA26"] = ema(c, 12), ema(c, 26)
    out = out.join(macd(c))
    out["RSI14"] = rsi(c)
    out = out.join(stochastic(out)).join(bollinger(c))
    out["ATR14"] = atr(out)
    return out
