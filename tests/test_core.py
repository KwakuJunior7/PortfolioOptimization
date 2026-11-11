import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data_loader as dl
from src.forecast import run_forecast
from src.gain_estimator import estimate_gain
from src.indicators import add_indicators, rsi
from src.optimizer import max_sharpe, min_variance, optimise, portfolio_performance
from src.risk import cvar, max_drawdown, var_historical
from src.strategy import ma_crossover

TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL"]


@pytest.fixture(scope="module")
def data():
    return dl.generate_sample_data(TICKERS, start="2021-01-01", end="2024-01-01")


def test_indicators_ranges(data):
    ind = add_indicators(data["AAPL"]).dropna()
    assert ind["RSI14"].between(0, 100).all()
    assert ind["Stoch_K"].between(0, 100).all()
    assert (ind["BB_Upper"] >= ind["BB_Lower"]).all()
    assert (ind["ATR14"] > 0).all()


def test_rsi_monotonic_up_is_100():
    s = pd.Series(np.arange(1, 40, dtype=float))
    assert rsi(s).dropna().iloc[-1] == pytest.approx(100)


def test_weights_sum_and_bounds(data):
    prices = dl.close_matrix(data, TICKERS)
    r = prices.pct_change().dropna()
    mu, cov = r.mean() * 252, r.cov() * 252
    for w in (max_sharpe(mu, cov, 0.04, 0.4), min_variance(mu, cov, 0.4)):
        assert w.sum() == pytest.approx(1)
        assert (w >= 0).all() and (w <= 0.4 + 1e-6).all()


def test_max_sharpe_beats_random(data):
    prices = dl.close_matrix(data, TICKERS)
    opt = optimise(prices, rf=0.04, n_random=2000)
    assert opt["performance"]["Max Sharpe"]["Sharpe"] >= opt["random"]["Sharpe"].max() - 1e-3
    assert opt["performance"]["Min Variance"]["Volatility"] <= opt["random"]["Volatility"].min() + 1e-3


def test_portfolio_performance_formula():
    mu = pd.Series([0.1, 0.2]); cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.09]])
    ret, vol, sh = portfolio_performance(np.array([0.5, 0.5]), mu, cov, rf=0.05)
    assert ret == pytest.approx(0.15)
    assert vol == pytest.approx(np.sqrt(0.25 * 0.04 + 0.25 * 0.09))
    assert sh == pytest.approx((0.15 - 0.05) / vol)


def test_risk_metrics():
    r = pd.Series(np.r_[np.full(90, 0.01), np.full(10, -0.05)])
    assert var_historical(r, 0.95) == pytest.approx(0.05)
    assert cvar(r, 0.95) >= var_historical(r, 0.95)
    assert max_drawdown(pd.Series([0.1, -0.5, 0.2])) == pytest.approx(-0.5)


def test_gain_estimator():
    g = estimate_gain([10_000, 50_000], 0.018)
    assert g["Predicted Gain"].tolist() == pytest.approx([180, 900])


def test_forecast_no_leakage_and_outputs(data):
    res = run_forecast(data["MSFT"], "MSFT", data["^GSPC"])
    assert res.signal["signal"] in {"Bullish", "Bearish"}
    assert 0.5 <= res.signal["confidence"] <= 1
    assert res.next_day["pred_close"] > 0
    assert res.test_predictions.index.min() > data["MSFT"].index.min()


def test_strategy(data):
    bt = ma_crossover(data["MSFT"])
    assert set(bt["Shares"].unique()) <= {0, 1}
    assert bt["Wealth"].iloc[-1] == pytest.approx(bt["Profit"].sum())


def test_report_outputs(tmp_path):
    from src.pipeline import run, to_json
    from src import report
    res = run(TICKERS, offline=True)
    to_json(res, str(tmp_path / "results.json"))
    assert len(report.export_tables(res, str(tmp_path / "tables"))) >= 10
    text = report.write_summary(res, str(tmp_path / "summary.md"), [], 1.0)
    assert "Optimal weights" in text and "Next-session signals" in text


def test_fetch_prices_parses_yfinance_layout(monkeypatch):
    """Mimic yfinance's (Ticker, Price) MultiIndex output without touching the network."""
    import types
    sample = dl.generate_sample_data(["AAPL", "MSFT"], start="2023-01-01", end="2023-03-01")
    raw = pd.concat({t: sample[t] for t in ["AAPL", "MSFT"]}, axis=1)
    fake = types.SimpleNamespace(download=lambda *a, **k: raw)
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    out = dl.fetch_prices(["AAPL", "MSFT"], "2023-01-01", "2023-03-01")
    assert set(out) == {"AAPL", "MSFT"} and list(out["AAPL"].columns) == dl.OHLCV
