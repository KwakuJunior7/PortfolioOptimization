"""Module 1 - Data acquisition.

Downloads OHLCV prices for a list of tickers plus the S&P 500 benchmark and
the 13-week T-bill yield (risk-free rate) from Yahoo Finance via yfinance.
An offline sample generator is included so the full pipeline and tests can
run without network access; its output is synthetic and labelled as such.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable

import numpy as np
import pandas as pd

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def fetch_prices(tickers: Iterable[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """Return {ticker: OHLCV DataFrame} downloaded with yfinance (split/dividend adjusted)."""
    import yfinance as yf  # imported lazily so offline use doesn't require it

    import time

    tickers = list(tickers)
    raw = None
    for attempt in range(3):  # Yahoo occasionally rate-limits; retry with back-off
        try:
            raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                              group_by="ticker", progress=False, threads=True)
            if raw is not None and not raw.empty:
                break
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"  download attempt {attempt + 1} failed: {exc}")
        time.sleep(3 * (attempt + 1))
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data - check your internet connection and tickers, "
                           "or run with --offline to test the pipeline on sample data.")
    out, missing = {}, []
    for t in tickers:
        try:
            df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = df[OHLCV].dropna(how="all").dropna()
        except KeyError:
            df = pd.DataFrame()
        if df.empty:
            missing.append(t)
            continue
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Date"
        out[t] = df
    if missing:
        raise RuntimeError(f"No data returned for: {', '.join(missing)}. Check the ticker symbols.")
    return out


def fetch_risk_free(start: str, end: str, ticker: str = "^IRX", default: float = 0.04) -> float:
    """Latest annualised risk-free rate as a decimal (e.g. 0.043)."""
    try:
        import yfinance as yf
        irx = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)["Close"].dropna()
        return float(np.asarray(irx).ravel()[-1]) / 100.0
    except Exception:
        return default


def to_long(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [df.assign(Ticker=t).reset_index() for t, df in data.items()]
    return pd.concat(frames, ignore_index=True)[["Date", "Ticker"] + OHLCV]


def from_long(long_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    long_df = long_df.copy()
    long_df["Date"] = pd.to_datetime(long_df["Date"])
    return {t: g.set_index("Date")[OHLCV].sort_index() for t, g in long_df.groupby("Ticker")}


def save_csv(data: Dict[str, pd.DataFrame], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    to_long(data).to_csv(path, index=False)


def load_csv(path: str) -> Dict[str, pd.DataFrame]:
    return from_long(pd.read_csv(path))


def close_matrix(data: Dict[str, pd.DataFrame], tickers: Iterable[str]) -> pd.DataFrame:
    return pd.concat({t: data[t]["Close"] for t in tickers}, axis=1).dropna()


def generate_sample_data(tickers: Iterable[str], benchmark: str = "^GSPC",
                         start: str = "2020-01-01", end: str = "2026-01-01",
                         seed: int = 14) -> Dict[str, pd.DataFrame]:
    """SYNTHETIC one-factor GBM prices for offline demos and tests.

    Not real market data. Each stock = beta * market shock + idiosyncratic shock,
    with drift/volatility loosely in line with large-cap tech. Because the paths
    are random walks, ML direction models should score ~50% on them.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    params = {  # annual drift, annual idiosyncratic vol, beta, start price
        "AAPL": (0.20, 0.18, 1.15, 75.0), "MSFT": (0.22, 0.16, 1.05, 160.0),
        "NVDA": (0.55, 0.38, 1.70, 6.0), "GOOGL": (0.18, 0.20, 1.10, 68.0),
        "AMZN": (0.15, 0.22, 1.20, 95.0), "META": (0.25, 0.30, 1.30, 210.0),
    }
    dt = 1 / 252
    mkt_mu, mkt_vol = 0.11, 0.19
    mkt_shock = rng.standard_normal(n)
    out = {}

    def _ohlcv(close: np.ndarray, vol_d: float, base_volume: float) -> pd.DataFrame:
        prev = np.r_[close[0], close[:-1]]
        open_ = prev * np.exp(rng.normal(0, vol_d * 0.3, n))
        hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, vol_d * 0.5, n)))
        lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, vol_d * 0.5, n)))
        volume = (base_volume * np.exp(rng.normal(0, 0.3, n))).astype(np.int64)
        return pd.DataFrame({"Open": open_, "High": hi, "Low": lo, "Close": close,
                             "Volume": volume}, index=pd.Index(dates, name="Date"))

    m_ret = (mkt_mu - 0.5 * mkt_vol ** 2) * dt + mkt_vol * np.sqrt(dt) * mkt_shock
    out[benchmark] = _ohlcv(3230 * np.exp(np.cumsum(m_ret)), mkt_vol / np.sqrt(252), 4e9)
    for t in tickers:
        mu, ivol, beta, p0 = params.get(t, (0.12, 0.25, 1.0, 100.0))
        total_vol = np.sqrt((beta * mkt_vol) ** 2 + ivol ** 2)
        shock = beta * mkt_vol * mkt_shock + ivol * rng.standard_normal(n)
        r = (mu - 0.5 * total_vol ** 2) * dt + np.sqrt(dt) * shock
        out[t] = _ohlcv(p0 * np.exp(np.cumsum(r)), total_vol / np.sqrt(252), 5e7)
    return out
