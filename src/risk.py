"""Module 9 - Risk metrics (VaR, CVaR, drawdown, beta, Sharpe/Sortino/Treynor)."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS = 252


def var_historical(r: pd.Series, level: float = 0.95) -> float:
    """1-day historical VaR as a positive loss fraction."""
    return float(-np.percentile(r.dropna(), (1 - level) * 100))


def var_parametric(r: pd.Series, level: float = 0.95) -> float:
    return float(-(r.mean() + norm.ppf(1 - level) * r.std()))


def cvar(r: pd.Series, level: float = 0.95) -> float:
    """Expected shortfall: mean loss on days worse than the VaR threshold."""
    r = r.dropna()
    cutoff = np.percentile(r, (1 - level) * 100)
    return float(-r[r <= cutoff].mean())


def drawdown_series(r: pd.Series) -> pd.Series:
    wealth = (1 + r.fillna(0)).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown_series(r).min())


def beta(r: pd.Series, market: pd.Series) -> float:
    df = pd.concat([r, market], axis=1).dropna()
    return float(np.cov(df.iloc[:, 0], df.iloc[:, 1])[0, 1] / df.iloc[:, 1].var())


def sharpe(r: pd.Series, rf: float = 0.0) -> float:
    ex = r.mean() * TRADING_DAYS - rf
    return float(ex / (r.std() * np.sqrt(TRADING_DAYS)))


def sortino(r: pd.Series, rf: float = 0.0) -> float:
    downside = r[r < 0]
    dd = np.sqrt((downside ** 2).sum() / len(r)) * np.sqrt(TRADING_DAYS)
    return float((r.mean() * TRADING_DAYS - rf) / dd)


def treynor(r: pd.Series, market: pd.Series, rf: float = 0.0) -> float:
    return float((r.mean() * TRADING_DAYS - rf) / beta(r, market))


def risk_report(r: pd.Series, market: Optional[pd.Series] = None, rf: float = 0.0) -> Dict[str, float]:
    out = {
        "Annual Return": float(r.mean() * TRADING_DAYS),
        "Annual Volatility": float(r.std() * np.sqrt(TRADING_DAYS)),
        "Sharpe Ratio": sharpe(r, rf),
        "Sortino Ratio": sortino(r, rf),
        "Max Drawdown": max_drawdown(r),
        "VaR 95% (1d)": var_historical(r, 0.95),
        "VaR 99% (1d)": var_historical(r, 0.99),
        "CVaR 95% (1d)": cvar(r, 0.95),
        "CVaR 99% (1d)": cvar(r, 0.99),
        "Parametric VaR 95% (1d)": var_parametric(r, 0.95),
    }
    if market is not None:
        out["Beta"] = beta(r, market)
        out["Treynor Ratio"] = treynor(r, market, rf)
    return out
