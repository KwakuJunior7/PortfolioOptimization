"""Modules 5-7 - Modern Portfolio Theory, Sharpe optimisation, efficient frontier.

    R_p     = sum_i w_i R_i
    sigma_p = sqrt(w^T Sigma w)
    Sharpe  = (R_p - R_f) / sigma_p
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def annual_stats(returns: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    return returns.mean() * TRADING_DAYS, returns.cov() * TRADING_DAYS


def portfolio_performance(w: np.ndarray, mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0):
    ret = float(np.dot(w, mu))
    vol = float(np.sqrt(w @ cov.values @ w))
    return ret, vol, (ret - rf) / vol if vol > 0 else np.nan


def _constraints(n, max_weight):
    bounds = tuple((0.0, max_weight) for _ in range(n))
    return bounds, [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]


def max_sharpe(mu, cov, rf=0.0, max_weight: float = 1.0) -> np.ndarray:
    n = len(mu)
    bounds, cons = _constraints(n, max_weight)
    res = minimize(lambda w: -portfolio_performance(w, mu, cov, rf)[2], np.full(n, 1 / n),
                   method="SLSQP", bounds=bounds, constraints=cons)
    return _clean(res.x)


def min_variance(mu, cov, max_weight: float = 1.0) -> np.ndarray:
    n = len(mu)
    bounds, cons = _constraints(n, max_weight)
    res = minimize(lambda w: w @ cov.values @ w, np.full(n, 1 / n), method="SLSQP",
                   bounds=bounds, constraints=cons)
    return _clean(res.x)


def efficient_frontier(mu, cov, n_points: int = 50, max_weight: float = 1.0) -> pd.DataFrame:
    """Minimum volatility for each target return between the min-variance and max-return portfolios."""
    n = len(mu)
    bounds, base = _constraints(n, max_weight)
    r_min = portfolio_performance(min_variance(mu, cov, max_weight), mu, cov)[0]
    r_max = _max_return(mu, max_weight)
    pts = []
    for target in np.linspace(r_min, r_max, n_points):
        cons = base + [{"type": "eq", "fun": lambda w, t=target: np.dot(w, mu) - t}]
        res = minimize(lambda w: w @ cov.values @ w, np.full(n, 1 / n), method="SLSQP",
                       bounds=bounds, constraints=cons)
        if res.success:
            pts.append({"Return": target, "Volatility": float(np.sqrt(res.fun))})
    return pd.DataFrame(pts)


def random_portfolios(mu, cov, n: int = 10_000, rf: float = 0.0, seed: int = 42,
                      max_weight: float = 1.0) -> pd.DataFrame:
    """Monte-Carlo long-only portfolios that respect the same weight cap as the optimiser."""
    rng = np.random.default_rng(seed)
    k = len(mu)
    if max_weight * k < 1:
        raise ValueError("max_weight too small: weights can't sum to 1")
    chunks, have = [], 0
    while have < n:
        W = rng.dirichlet(np.ones(k), n)
        W = W[(W <= max_weight + 1e-12).all(axis=1)]
        chunks.append(W); have += len(W)
    W = np.vstack(chunks)[:n]
    rets = W @ mu.values
    vols = np.sqrt(np.einsum("ij,jk,ik->i", W, cov.values, W))
    df = pd.DataFrame(W, columns=mu.index)
    df["Return"], df["Volatility"] = rets, vols
    df["Sharpe"] = (rets - rf) / vols
    return df


def _max_return(mu: pd.Series, max_weight: float) -> float:
    """Highest achievable return under long-only + cap: fill best assets greedily."""
    left, total = 1.0, 0.0
    for r in mu.sort_values(ascending=False):
        take = min(max_weight, left)
        total += take * r; left -= take
        if left <= 1e-12:
            break
    return float(total) - 1e-9


def _clean(w: np.ndarray) -> np.ndarray:
    w = np.where(w < 1e-6, 0.0, w)
    return w / w.sum()


def optimise(prices: pd.DataFrame, rf: float = 0.0, max_weight: float = 1.0,
             n_random: int = 10_000) -> Dict[str, object]:
    """One call that returns everything the dashboard/report needs."""
    rets = daily_returns(prices)
    mu, cov = annual_stats(rets)
    w_ms, w_mv = max_sharpe(mu, cov, rf, max_weight), min_variance(mu, cov, max_weight)
    w_eq = np.full(len(mu), 1 / len(mu))
    perf = {name: dict(zip(("Return", "Volatility", "Sharpe"), portfolio_performance(w, mu, cov, rf)))
            for name, w in (("Max Sharpe", w_ms), ("Min Variance", w_mv), ("Equal Weight", w_eq))}
    weights = pd.DataFrame({"Max Sharpe": w_ms, "Min Variance": w_mv, "Equal Weight": w_eq},
                           index=mu.index)
    return {"returns": rets, "mu": mu, "cov": cov, "weights": weights, "performance": perf,
            "frontier": efficient_frontier(mu, cov, max_weight=max_weight),
            "random": random_portfolios(mu, cov, n_random, rf, max_weight=max_weight)}
