"""Outputs of a pipeline run: CSV tables, README figures, a markdown summary and saved models."""
from __future__ import annotations

import os
import platform
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

# Palette (light mode) - categorical order is fixed so a ticker keeps its colour everywhere
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
GOOD, BAD = "#0ca30c", "#d03b3b"


# --------------------------------------------------------------------------- tables
def export_tables(res: Dict, outdir: str) -> List[str]:
    os.makedirs(outdir, exist_ok=True)
    tk, opt = res["tickers"], res["opt"]
    files = {}
    files["weights.csv"] = opt["weights"]
    files["portfolio_performance.csv"] = pd.DataFrame(opt["performance"]).T
    files["signals.csv"] = signals_table(res)
    files["risk_metrics.csv"] = pd.DataFrame(res["risk"]).T
    files["daily_gain_estimate.csv"] = res["gains"].set_index("Capital")
    files["regression_metrics.csv"] = pd.concat(
        {t: res["forecasts"][t].regression_metrics for t in tk}, names=["Ticker", "Model"])
    files["classification_metrics.csv"] = pd.concat(
        {t: res["forecasts"][t].classification_metrics for t in tk}, names=["Ticker", "Model"])
    files["feature_importance.csv"] = pd.DataFrame({t: res["forecasts"][t].feature_importance for t in tk})
    files["correlation.csv"] = opt["returns"].corr()
    files["annual_stats.csv"] = pd.DataFrame({"Annual Return": opt["mu"],
                                              "Annual Volatility": np.sqrt(np.diag(opt["cov"]))})
    files["ma_crossover_backtest.csv"] = pd.DataFrame(
        {t: _strategy_summary(res["strategy"][t]) for t in tk}).T
    files["efficient_frontier.csv"] = opt["frontier"].set_index("Return")
    written = []
    for name, df in files.items():
        path = os.path.join(outdir, name)
        df.to_csv(path)
        written.append(path)
    return written


def signals_table(res: Dict) -> pd.DataFrame:
    rows = []
    for t, f in res["forecasts"].items():
        nd, s = f.next_day, f.signal
        rows.append({"Ticker": t, "As of": s["as_of"], "Signal": s["signal"],
                     "Confidence": s["confidence"], "Classifier": s["model"],
                     "Last Close": nd["last_close"], "Pred Open": nd["pred_open"],
                     "Pred Close": nd["pred_close"], "Pred Return": nd["pred_return"],
                     "Regressor": nd["model"]})
    return pd.DataFrame(rows).set_index("Ticker")


def _strategy_summary(bt: pd.DataFrame) -> Dict[str, float]:
    from .strategy import summary
    return summary(bt)


# --------------------------------------------------------------------------- figures
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 13, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
        "font.size": 10, "legend.frameon": False, "lines.linewidth": 2,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    })
    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    return plt


def make_figures(res: Dict, outdir: str, dpi: int = 160) -> List[str]:
    """Save README-ready PNG charts. Returns the list of written files."""
    plt = _style()
    import matplotlib.dates as mdates
    import matplotlib.ticker as mtick

    os.makedirs(outdir, exist_ok=True)
    tk, opt = res["tickers"], res["opt"]
    color = {t: SERIES[i % len(SERIES)] for i, t in enumerate(tk)}
    pct = mtick.PercentFormatter(1.0, decimals=0)
    written = []

    def save(fig, name):
        path = os.path.join(outdir, name)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        written.append(path)

    # 1. growth of $1
    fig, ax = plt.subplots(figsize=(10, 4.6))
    wealth = (1 + res["portfolio_returns"]).cumprod()
    ax.plot(wealth, color=SERIES[0], label="Max-Sharpe portfolio")
    ax.plot((1 + opt["returns"].mean(axis=1)).cumprod(), color=SERIES[1], label="Equal weight")
    if res["market_returns"] is not None:
        ax.plot((1 + res["market_returns"].fillna(0)).cumprod(), color=MUTED, ls="--", label="S&P 500")
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.2f}"))
    ax.set_title("Growth of $1 invested")
    ax.legend(loc="upper left")
    save(fig, "growth.png")

    # 2. efficient frontier
    fig, ax = plt.subplots(figsize=(10, 6))
    rnd, fr = opt["random"], opt["frontier"]
    sc = ax.scatter(rnd["Volatility"], rnd["Return"], c=rnd["Sharpe"], cmap="Blues", s=5, alpha=0.8,
                    linewidths=0)
    fig.colorbar(sc, ax=ax, label="Sharpe ratio", pad=0.01)
    ax.plot(fr["Volatility"], fr["Return"], color=INK, lw=2, label="Efficient frontier")
    for (name, marker, c) in (("Max Sharpe", "*", SERIES[1]), ("Min Variance", "D", SERIES[2]),
                              ("Equal Weight", "o", SERIES[3])):
        p = opt["performance"][name]
        ax.scatter(p["Volatility"], p["Return"], marker=marker, s=260 if marker == "*" else 110,
                   color=c, edgecolor=SURFACE, linewidth=2, zorder=5, label=name)
    ax.xaxis.set_major_formatter(pct); ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("Volatility (annualised)"); ax.set_ylabel("Expected return (annualised)")
    ax.set_title(f"Efficient frontier - {len(rnd):,} random portfolios")
    ax.legend(loc="lower right")
    save(fig, "efficient_frontier.png")

    # 3. allocation
    fig, ax = plt.subplots(figsize=(10, 4.2))
    w = opt["weights"]
    x = np.arange(len(tk)); bw = 0.26
    for i, col in enumerate(w.columns):
        bars = ax.bar(x + (i - 1) * bw, w[col], bw - 0.03, color=SERIES[i], label=col)
        for b, v in zip(bars, w[col]):
            if v > 0.005:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center", fontsize=8,
                        color=INK2)
    ax.set_xticks(x, tk); ax.yaxis.set_major_formatter(pct); ax.grid(axis="x", visible=False)
    ax.set_ylim(0, w.values.max() * 1.15)
    ax.set_title("Portfolio allocation", pad=26)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9)
    save(fig, "allocation.png")

    # 4. correlation heatmap (diverging, grey midpoint)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    corr = opt["returns"].corr()
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(tk)), tk); ax.set_yticks(range(len(tk)), tk); ax.grid(False)
    for i in range(len(tk)):
        for j in range(len(tk)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color=INK, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Daily return correlation")
    save(fig, "correlation.png")

    # 5. predicted vs actual (small multiples)
    n = len(tk); cols = 2; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.2 * rows), squeeze=False)
    for ax, t in zip(axes.ravel(), tk):
        tp = res["forecasts"][t].test_predictions.iloc[-90:]
        ax.plot(tp["Actual Close"], color=INK, lw=1.8, label="Actual")
        best = res["forecasts"][t].next_day["model"]
        ax.plot(tp[best], color=color[t], lw=1.4, label=f"Predicted ({best})")
        ax.set_title(t, fontsize=11); ax.legend(fontsize=8, loc="upper left")
        loc = mdates.AutoDateLocator(maxticks=5)
        ax.xaxis.set_major_locator(loc); ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle("Next-day close: predicted vs actual (hold-out, last 90 days)", x=0.01, ha="left",
                 fontsize=13, fontweight="semibold", color=INK)
    save(fig, "forecast_vs_actual.png")

    # 6. model accuracy vs baseline
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(tk))
    base = [res["forecasts"][t].classification_metrics.loc["Majority-class baseline", "Accuracy"] for t in tk]
    best = [res["forecasts"][t].classification_metrics.drop("Majority-class baseline")["Accuracy"].max()
            for t in tk]
    ax.bar(x - 0.18, base, 0.34, color=MUTED, label="Baseline (always guess the common side)")
    ax.bar(x + 0.18, best, 0.34, color=SERIES[0], label="Best classifier")
    for i, (b, m) in enumerate(zip(base, best)):
        ax.text(i - 0.18, b + 0.005, f"{b:.1%}", ha="center", fontsize=8, color=INK2)
        ax.text(i + 0.18, m + 0.005, f"{m:.1%}", ha="center", fontsize=8, color=INK2)
    ax.axhline(0.5, color=INK2, lw=1, ls=":")
    ax.set_ylim(0.4, max(best + base) + 0.06); ax.set_xticks(x, tk); ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(pct)
    ax.set_title("Direction accuracy on unseen data vs baseline"); ax.legend(loc="upper right", fontsize=8)
    save(fig, "direction_accuracy.png")

    # 7. drawdown
    fig, ax = plt.subplots(figsize=(10, 3.4))
    dd = res["drawdown"]
    ax.fill_between(dd.index, dd, 0, color=BAD, alpha=0.15, lw=0)
    ax.plot(dd, color=BAD, lw=1.4)
    ax.yaxis.set_major_formatter(pct)
    ax.set_title(f"Max-Sharpe portfolio drawdown (worst {dd.min():.1%})")
    save(fig, "drawdown.png")

    # 8. technical indicators for the first ticker
    t = tk[0]
    ind = res["indicators"][t].iloc[-180:]
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
    a1.fill_between(ind.index, ind["BB_Lower"], ind["BB_Upper"], color=SERIES[0], alpha=0.10, lw=0,
                    label="Bollinger band")
    a1.plot(ind["Close"], color=color[t], label="Close")
    a1.plot(ind["SMA20"], color=SERIES[1], lw=1.4, label="SMA20")
    a1.plot(ind["SMA50"], color=MUTED, lw=1.4, ls="--", label="SMA50")
    a1.set_title(f"{t} - technical indicators (last 180 sessions)"); a1.legend(ncol=4, loc="upper left")
    a2.plot(ind["MACD"], color=SERIES[0], lw=1.4, label="MACD")
    a2.plot(ind["MACD_Signal"], color=SERIES[1], lw=1.4, label="Signal")
    a2.bar(ind.index, ind["MACD_Hist"], color=np.where(ind["MACD_Hist"] >= 0, GOOD, BAD), alpha=0.4, width=1)
    a2.legend(ncol=2, loc="upper left", fontsize=8)
    a3.plot(ind["RSI14"], color=SERIES[6], lw=1.4, label="RSI 14")
    for lvl in (30, 70):
        a3.axhline(lvl, color=MUTED, lw=1, ls=":")
    a3.set_ylim(0, 100); a3.legend(loc="upper left", fontsize=8)
    save(fig, f"indicators_{t}.png")

    return written


# --------------------------------------------------------------------------- summary
def _md_table(df: pd.DataFrame, fmt: Optional[Dict[str, str]] = None, index: bool = True) -> str:
    fmt = fmt or {}
    d = df.reset_index().rename(columns={"index": ""}) if index else df.copy()
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join(["---"] + ["---:"] * (len(d.columns) - 1)) + "|"
    lines = [head, sep]
    for _, row in d.iterrows():
        cells = []
        for c in d.columns:
            v = row[c]
            f = fmt.get(c)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append("-")
            elif f and isinstance(v, (int, float, np.floating, np.integer)):
                cells.append(format(v, f))
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def package_versions(names: Iterable[str]) -> Dict[str, str]:
    from importlib import metadata
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = "not installed"
    return out


def write_summary(res: Dict, path: str, warnings: List[str], runtime_s: float,
                  lstm: Optional[Dict[str, Dict]] = None, notes: Optional[List[str]] = None) -> str:
    tk, opt = res["tickers"], res["opt"]
    P = "{:.2%}"
    pv = package_versions(["pandas", "numpy", "scipy", "scikit-learn", "xgboost", "yfinance",
                           "matplotlib", "dash", "plotly", "tensorflow"])
    L = [f"# Pipeline results summary",
         "",
         f"_Generated by `run.py` on {datetime.now():%Y-%m-%d %H:%M} ({runtime_s:.0f}s). Snapshot only - "
         "signals go stale after one session and nothing here is investment advice._",
         "",
         "## Run info",
         f"- **Data source:** {res['source']}",
         f"- **Tickers:** {', '.join(tk)}  |  **Benchmark:** S&P 500",
         f"- **Period:** {res['prices'].index[0].date()} to {res['prices'].index[-1].date()} "
         f"({len(res['prices'])} trading days)",
         f"- **Risk-free rate:** {res['risk_free']:.2%}  |  **Per-asset weight cap:** {res['max_weight']:.0%}",
         f"- **Python:** {platform.python_version()} on {platform.system()} {platform.release()}",
         "- **Packages:** " + ", ".join(f"{k} {v}" for k, v in pv.items()),
         "",
         "## Optimal weights",
         "_Optimised on the full history above, so returns below are historical (in-sample), not forecasts._",
         "",
         _md_table(opt["weights"], {c: ".1%" for c in opt["weights"].columns}),
         "",
         "## Portfolio performance (annualised)",
         _md_table(pd.DataFrame(opt["performance"]).T, {"Return": ".2%", "Volatility": ".2%", "Sharpe": ".3f"}),
         "",
         "## Per-stock statistics",
         _md_table(pd.DataFrame({"Annual Return": opt["mu"], "Annual Volatility": np.sqrt(np.diag(opt["cov"])),
                                 "Sharpe": [res["risk"][t]["Sharpe Ratio"] for t in tk],
                                 "Beta": [res["risk"][t].get("Beta", np.nan) for t in tk]}),
                   {"Annual Return": ".2%", "Annual Volatility": ".2%", "Sharpe": ".2f", "Beta": ".2f"}),
         "",
         "## Next-session signals",
         _md_table(signals_table(res), {"Confidence": ".1%", "Last Close": ",.2f", "Pred Open": ",.2f",
                                        "Pred Close": ",.2f", "Pred Return": "+.3%"}),
         "",
         "## Model evaluation (chronological hold-out, last 20%)"]
    for t in tk:
        f = res["forecasts"][t]
        L += [f"### {t}", "**Price models**", "",
              _md_table(f.regression_metrics, {"Close RMSE ($)": ".3f", "Close MAE ($)": ".3f",
                                               "Open RMSE ($)": ".3f", "Direction hit-rate": ".1%"}),
              "", "**Direction classifiers**", "",
              _md_table(f.classification_metrics, {"Accuracy": ".1%", "ROC AUC": ".3f"}),
              "", "Top features: " + ", ".join(f"{k} ({v:.3f})" for k, v in f.feature_importance.head(5).items()),
              ""]
    if lstm:
        L += ["### LSTM", _md_table(pd.DataFrame(lstm).T, {c: ".3f" for c in next(iter(lstm.values()))}), ""]
    risk_df = pd.DataFrame(res["risk"])
    pct_rows = [r for r in risk_df.index if any(k in r for k in ("Return", "Volatility", "VaR", "Drawdown"))]
    risk_fmt = risk_df.copy().astype(object)
    for r in risk_df.index:
        risk_fmt.loc[r] = [P.format(v) if r in pct_rows else f"{v:.3f}" for v in risk_df.loc[r]]
    L += ["## Risk metrics", _md_table(risk_fmt), "",
          "## Speculative daily gain estimate (Max-Sharpe portfolio)",
          _md_table(res["gains"], {"Capital": ",.0f", "Predicted Return": "+.3%", "Predicted Gain": ",.2f",
                                   "Low (-1 sd)": ",.2f", "High (+1 sd)": ",.2f", "1-day VaR 95% loss": ",.2f"},
                    index=False), "",
          "## MA-crossover back-test (1 share, MA10 vs MA50)",
          _md_table(pd.DataFrame({t: _strategy_summary(res["strategy"][t]) for t in tk}).T,
                    {"Total profit (strategy, $/share)": ",.2f", "Total profit (buy & hold, $/share)": ",.2f",
                     "Days in market": ",.0f", "Share of days in market": ".1%"}), "",
          "## Warnings / errors",
          *(["- " + w for w in warnings] or ["- none"]), "",
          "## Environment notes",
          *(["- " + n for n in (notes or [])] or ["- all optional packages installed"]), ""]
    text = "\n".join(L)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


# --------------------------------------------------------------------------- models
def save_models(res: Dict, model_dir: str) -> List[str]:
    import joblib
    os.makedirs(model_dir, exist_ok=True)
    written = []
    for t, f in res["forecasts"].items():
        path = os.path.join(model_dir, f"{t.replace('^', '')}_models.joblib")
        joblib.dump({**f.models, "ticker": t, "trained_through": f.next_day["as_of"],
                     "regressor_name": f.next_day["model"], "classifier_name": f.signal["model"]}, path)
        written.append(path)
    return written
