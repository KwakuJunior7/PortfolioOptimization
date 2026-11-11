"""End-to-end pipeline: data -> indicators -> forecasts -> MPT -> risk -> gains."""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import config
from . import data_loader as dl
from .forecast import run_forecast
from .gain_estimator import estimate_gain
from .indicators import add_indicators
from .optimizer import optimise
from .risk import drawdown_series, risk_report
from .strategy import ma_crossover, summary


def load_data(tickers: List[str], offline: bool = False, csv: Optional[str] = None):
    """Returns (data dict, risk-free rate, source label)."""
    if csv:
        return dl.load_csv(csv), config.DEFAULT_RISK_FREE, f"CSV: {csv}"
    if offline:
        return (dl.generate_sample_data(tickers, config.BENCHMARK, config.START_DATE, "2026-01-01"),
                config.DEFAULT_RISK_FREE, "SYNTHETIC sample data (offline mode)")
    data = dl.fetch_prices(tickers + [config.BENCHMARK], config.START_DATE, config.END_DATE)
    dl.save_csv(data, os.path.join(config.DATA_DIR, "historical_prices.csv"))
    rf = dl.fetch_risk_free(config.START_DATE, config.END_DATE, config.RISK_FREE_TICKER,
                            config.DEFAULT_RISK_FREE)
    return data, rf, "Yahoo Finance (yfinance)"


@contextmanager
def _step(log: Callable[[str], None], name: str):
    log(f"[{name}] ...")
    t0 = time.perf_counter()
    yield
    log(f"[{name}] done in {time.perf_counter() - t0:.1f}s")


def run(tickers: List[str] = None, offline: bool = False, csv: Optional[str] = None,
        capitals=(10_000, 50_000, 100_000), max_weight: float = config.MAX_WEIGHT,
        progress: Optional[Callable[[str], None]] = None) -> Dict:
    """Run modules 1-9 and return everything in one dict (used by run.py and the dashboard)."""
    log = progress or (lambda msg: None)
    tickers = tickers or config.TICKERS

    with _step(log, "1/6 Data acquisition"):
        data, rf, source = load_data(tickers, offline, csv)
        bench = data.get(config.BENCHMARK)
        prices = dl.close_matrix(data, tickers)
        log(f"      {source} | {len(prices)} trading days "
            f"{prices.index[0].date()} -> {prices.index[-1].date()} | risk-free {rf:.2%}")

    with _step(log, "2/6 Technical indicators"):
        indicators = {t: add_indicators(data[t]) for t in tickers}

    forecasts = {}
    with _step(log, "3/6 Price prediction + bull/bear classification"):
        for t in tickers:
            t0 = time.perf_counter()
            forecasts[t] = run_forecast(data[t], t, bench, config.TEST_SIZE, config.RANDOM_STATE)
            s = forecasts[t].signal
            log(f"      {t:6s} {s['signal']:8s} {s['confidence']:.0%}  ({time.perf_counter() - t0:.1f}s)")

    with _step(log, "4/6 Portfolio optimisation + efficient frontier"):
        opt = optimise(prices, rf, max_weight, config.N_RANDOM_PORTFOLIOS)
        w = opt["weights"]["Max Sharpe"]
        port_r = (opt["returns"] * w).sum(axis=1)
        mkt_r = bench["Close"].pct_change().reindex(port_r.index) if bench is not None else None

    with _step(log, "5/6 Risk metrics"):
        risk = {"Max Sharpe Portfolio": risk_report(port_r, mkt_r, rf)}
        for t in tickers:
            risk[t] = risk_report(opt["returns"][t], mkt_r, rf)

    with _step(log, "6/6 Daily gain estimate + MA-crossover back-test"):
        # portfolio-level next-day expected return = weighted sum of model forecasts
        pred_ret = float(sum(w[t] * forecasts[t].next_day["pred_return"] for t in tickers))
        pred_sd = float(np.sqrt(sum((w[t] * forecasts[t].next_day["return_sd"]) ** 2 for t in tickers)))
        gains = estimate_gain(capitals, pred_ret, pred_sd, risk["Max Sharpe Portfolio"]["VaR 95% (1d)"])
        strategy = {t: ma_crossover(data[t]) for t in tickers}

    return {"source": source, "risk_free": rf, "max_weight": max_weight, "tickers": tickers,
            "data": data, "prices": prices, "forecasts": forecasts, "opt": opt, "risk": risk,
            "gains": gains, "portfolio_returns": port_r, "market_returns": mkt_r,
            "drawdown": drawdown_series(port_r), "indicators": indicators, "strategy": strategy}


def to_json(res: Dict, path: str) -> Dict:
    """Serialise the headline results (used by the static preview + README)."""
    tk = res["tickers"]
    opt = res["opt"]
    wealth = (1 + res["portfolio_returns"]).cumprod()
    eq = (1 + opt["returns"].mean(axis=1)).cumprod()
    mkt = (1 + res["market_returns"].fillna(0)).cumprod() if res["market_returns"] is not None else None
    step = 3
    norm = res["prices"] / res["prices"].iloc[0]
    rnd = opt["random"].sample(2500, random_state=1)
    out = {
        "source": res["source"], "risk_free": res["risk_free"], "tickers": tk,
        "max_weight": res.get("max_weight"), "n_random": int(len(opt["random"])),
        "start": str(res["prices"].index[0].date()), "end": str(res["prices"].index[-1].date()),
        "annual_return": opt["mu"].round(6).to_dict(),
        "annual_vol": (np.sqrt(np.diag(opt["cov"])) ).round(6).tolist(),
        "correlation": opt["returns"].corr().round(3).values.tolist(),
        "weights": {k: v.round(4).to_dict() for k, v in opt["weights"].items()},
        "performance": opt["performance"],
        "frontier": opt["frontier"].round(5).to_dict("list"),
        "random": {"Return": rnd["Return"].round(4).tolist(), "Volatility": rnd["Volatility"].round(4).tolist(),
                   "Sharpe": rnd["Sharpe"].round(3).tolist()},
        "growth": {"dates": [str(d.date()) for d in wealth.index[::step]],
                   "Max Sharpe": wealth.iloc[::step].round(4).tolist(),
                   "Equal Weight": eq.iloc[::step].round(4).tolist(),
                   "S&P 500": mkt.iloc[::step].round(4).tolist() if mkt is not None else []},
        "normalized": {t: norm[t].iloc[::step].round(4).tolist() for t in tk},
        "drawdown": res["drawdown"].iloc[::step].round(4).tolist(),
        "risk": res["risk"],
        "gains": res["gains"].round({c: 2 for c in res["gains"].columns if c != "Predicted Return"})
                             .round({"Predicted Return": 6}).to_dict("records"),
        "forecasts": {}, "indicators": {}, "strategy": {},
    }
    for t, f in res["forecasts"].items():
        out["forecasts"][t] = {
            "next_day": f.next_day, "signal": f.signal,
            "regression": f.regression_metrics.reset_index().replace({np.nan: None}).to_dict("records"),
            "classification": f.classification_metrics.reset_index().to_dict("records"),
            "importance": f.feature_importance.head(8).round(4).to_dict(),
            "test_tail": {"dates": [str(d.date()) for d in f.test_predictions.index[-90:]],
                          **{c: f.test_predictions[c].iloc[-90:].round(2).tolist()
                             for c in f.test_predictions.columns}},
        }
        ind = res["indicators"][t].iloc[-180:]
        out["indicators"][t] = {"dates": [str(d.date()) for d in ind.index],
                                **{c: ind[c].round(3).tolist() for c in
                                   ["Close", "SMA20", "SMA50", "EMA12", "BB_Upper", "BB_Lower", "RSI14",
                                    "MACD", "MACD_Signal", "Stoch_K", "ATR14"]}}
        out["indicators"][t]["latest"] = {c: round(float(res["indicators"][t][c].iloc[-1]), 3)
                                          for c in ["Close", "SMA20", "SMA50", "RSI14", "MACD",
                                                    "Stoch_K", "BB_PctB", "ATR14"]}
        out["strategy"][t] = summary(res["strategy"][t])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, default=float)
    return out
