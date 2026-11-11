"""Full pipeline - one command (or F5 in VS Code) runs every module and saves every output.

    python run.py                         # live Yahoo Finance data (default)
    python run.py --offline               # synthetic sample data, no internet needed
    python run.py --tickers AAPL MSFT NVDA GOOGL AMZN --max-weight 0.35
    python run.py --lstm                  # also train the LSTM (needs tensorflow)

Outputs
    reports/RESULTS_SUMMARY.md   <- human-readable summary of the run (send this back)
    reports/results.json         <- machine-readable results (feeds the HTML preview)
    reports/tables/*.csv         <- every table: weights, signals, metrics, risk, ...
    reports/pipeline.log         <- full console log
    docs/images/*.png            <- charts for the README
    docs/preview.html            <- self-contained dashboard snapshot
    data/historical_prices.csv   <- downloaded prices (live mode)
    models/*.joblib              <- fitted models per ticker
"""
import argparse
import logging
import os
import sys
import time
import traceback
import warnings as pywarnings

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)                      # make relative paths work however the script is launched
sys.path.insert(0, ROOT)

try:                                # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

REQUIRED = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "joblib"]
OPTIONAL = {"yfinance": "live data", "xgboost": "XGBoost models", "dash": "dashboard",
            "plotly": "dashboard", "tensorflow": "LSTM (--lstm)"}


def check_environment(log, need_yf: bool, need_tf: bool) -> list:
    import importlib
    notes, missing = [], []
    for m in REQUIRED:
        try:
            importlib.import_module(m)
        except ImportError:
            missing.append(m)
    if missing:
        log.error("Missing required packages: %s\n  -> run:  pip install -r requirements.txt", ", ".join(missing))
        sys.exit(1)
    for m, why in OPTIONAL.items():
        try:
            importlib.import_module(m)
        except ImportError:
            msg = f"optional package '{m}' not installed ({why})"
            if (m == "yfinance" and need_yf) or (m == "tensorflow" and need_tf):
                log.error("%s -> pip install %s", msg, m)
                sys.exit(1)
            notes.append(msg)
            log.warning(msg)
    return notes


def setup_logging() -> logging.Logger:
    os.makedirs("reports", exist_ok=True)
    log = logging.getLogger("pipeline")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler("reports/pipeline.log", mode="w", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def main():
    import config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=config.TICKERS)
    ap.add_argument("--offline", action="store_true", help="use synthetic sample data (no internet)")
    ap.add_argument("--csv", help="re-use saved prices: long CSV with Date,Ticker,Open,High,Low,Close,Volume")
    ap.add_argument("--max-weight", type=float, default=config.MAX_WEIGHT,
                    help=f"per-asset weight cap (default {config.MAX_WEIGHT}; 1.0 = unconstrained)")
    ap.add_argument("--lstm", action="store_true", help="also train the LSTM model (needs tensorflow)")
    ap.add_argument("--no-figures", action="store_true", help="skip PNG charts")
    a = ap.parse_args()

    log = setup_logging()
    pywarnings.filterwarnings("ignore", category=FutureWarning)
    t_start = time.perf_counter()
    log.info("=" * 70)
    log.info("AI-Driven Portfolio Optimization & Market Direction Prediction")
    log.info("=" * 70)
    notes = check_environment(log, need_yf=not (a.offline or a.csv), need_tf=a.lstm)
    issues = []

    from src.pipeline import run, to_json
    from src import report

    # ---- modules 1-9 -------------------------------------------------------
    try:
        res = run(a.tickers, a.offline, a.csv, max_weight=a.max_weight, progress=log.info)
    except Exception as exc:
        log.error("Pipeline failed: %s", exc)
        log.debug(traceback.format_exc())
        traceback.print_exc()
        sys.exit(1)

    # ---- outputs: each step is independent so one failure doesn't lose the rest ---
    def attempt(name, fn):
        t0 = time.perf_counter()
        try:
            out = fn()
            log.info("[output] %-28s ok (%.1fs)", name, time.perf_counter() - t0)
            return out
        except Exception as exc:
            msg = f"{name} failed: {type(exc).__name__}: {exc}"
            log.warning("[output] %s", msg)
            log.debug(traceback.format_exc())
            issues.append(msg)
            return None

    if not (a.offline or a.csv):
        from src.data_loader import save_csv
        attempt("prices CSV", lambda: save_csv(res["data"], "data/historical_prices.csv"))
    attempt("results.json", lambda: to_json(res, "reports/results.json"))
    attempt("CSV tables", lambda: report.export_tables(res, "reports/tables"))
    attempt("saved models", lambda: report.save_models(res, "models"))
    if not a.no_figures:
        attempt("README figures", lambda: report.make_figures(res, "docs/images"))

    def _preview():
        from scripts.build_preview import main as build
        build()
    attempt("HTML preview", _preview)

    lstm = None
    if a.lstm:
        def _lstm():
            from src.lstm_model import train_lstm
            out = {}
            for t in res["tickers"]:
                _, m = train_lstm(res["data"][t], res["data"].get(config.BENCHMARK),
                                  save_path=f"models/{t}_lstm.keras")
                out[t] = m
                log.info("      LSTM %-6s RMSE $%.3f vs naive $%.3f", t, m["LSTM Close RMSE ($)"],
                         m["Naive Close RMSE ($)"])
            return out
        lstm = attempt("LSTM models", _lstm)

    runtime = time.perf_counter() - t_start
    attempt("RESULTS_SUMMARY.md", lambda: report.write_summary(res, "reports/RESULTS_SUMMARY.md",
                                                               issues, runtime, lstm, notes))

    # ---- console summary -----------------------------------------------------
    opt = res["opt"]
    print("\n" + "=" * 70)
    print(f"Data   : {res['source']}  ({res['prices'].index[0].date()} -> {res['prices'].index[-1].date()})")
    print(f"Rf     : {res['risk_free']:.2%}    weight cap: {res['max_weight']:.0%}\n")
    print("Optimal weights")
    print(opt["weights"].apply(lambda col: col.map(lambda x: f"{x:.1%}")).to_string(), "\n")
    for name, p in opt["performance"].items():
        print(f"  {name:13s} return {p['Return']:7.2%}   vol {p['Volatility']:7.2%}   Sharpe {p['Sharpe']:.2f}")
    print("\nNext-session signals")
    for t, f in res["forecasts"].items():
        nd, s = f.next_day, f.signal
        print(f"  {t:6s} {s['signal']:8s} conf {s['confidence']:4.0%}   "
              f"open ${nd['pred_open']:,.2f}  close ${nd['pred_close']:,.2f}  ({nd['pred_return']:+.2%})")
    g = res["gains"].iloc[0]
    print(f"\nGain estimate on ${g['Capital']:,.0f}: {g['Predicted Gain']:+,.2f} "
          f"(range {g['Low (-1 sd)']:+,.2f} to {g['High (+1 sd)']:+,.2f}; 1-day VaR95 {g['1-day VaR 95% loss']:,.2f})")
    print("=" * 70)
    print(f"Finished in {runtime:.0f}s - {len(issues)} output error(s), {len(notes)} optional package(s) missing.")
    print("Send back : reports/RESULTS_SUMMARY.md  (plus docs/images/*.png if you like)")
    print("Dashboard : python dashboard/app.py   ->  http://127.0.0.1:8050")
    print("Preview   : open docs/preview.html in your browser")


if __name__ == "__main__":
    main()
