"""Module 10 - Dash dashboard.

    python dashboard/app.py            # live Yahoo Finance data
    python dashboard/app.py --offline  # synthetic sample data
Then open http://127.0.0.1:8050
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dash_table, dcc, html
from dash.dash_table import FormatTemplate

import config
from src.gain_estimator import estimate_gain
from src.pipeline import run

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GOOD, BAD, MUTED = "#0ca30c", "#d03b3b", "#898781"
LAYOUT = dict(template="plotly_white", margin=dict(l=50, r=20, t=40, b=40), hovermode="x unified",
              font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=13))


def build_app(res):
    tk = res["tickers"]
    color = {t: COLORS[i % len(COLORS)] for i, t in enumerate(tk)}
    opt = res["opt"]
    app = dash.Dash(__name__, title="AI Portfolio Optimizer")

    def card(title, value, sub="", tone=None):
        return html.Div([html.Div(title, className="k"), html.Div(value, className="v", style={"color": tone}),
                         html.Div(sub, className="s")], className="card")

    # ---------- static figures ----------
    growth = go.Figure(layout=LAYOUT)
    wealth = (1 + res["portfolio_returns"]).cumprod()
    growth.add_scatter(x=wealth.index, y=wealth, name="Max Sharpe", line=dict(color=COLORS[0], width=2))
    growth.add_scatter(x=wealth.index, y=(1 + opt["returns"].mean(axis=1)).cumprod(), name="Equal Weight",
                       line=dict(color=COLORS[1], width=2))
    if res["market_returns"] is not None:
        growth.add_scatter(x=wealth.index, y=(1 + res["market_returns"].fillna(0)).cumprod(), name="S&P 500",
                           line=dict(color=MUTED, width=2, dash="dot"))
    growth.update_layout(title="Growth of $1", yaxis_title="Value ($)")

    w = opt["weights"]
    alloc = go.Figure(layout={**LAYOUT, "hovermode": "closest"})
    for i, col in enumerate(w.columns):
        alloc.add_bar(x=w.index, y=w[col], name=col, marker_color=COLORS[i])
    alloc.update_layout(title="Portfolio allocation", yaxis_tickformat=".0%", barmode="group")

    rnd = opt["random"]
    front = go.Figure(layout={**LAYOUT, "hovermode": "closest"})
    front.add_scatter(x=rnd["Volatility"], y=rnd["Return"], mode="markers", name="Random portfolios",
                      marker=dict(size=4, color=rnd["Sharpe"], colorscale="Blues", showscale=True,
                                  colorbar=dict(title="Sharpe")), opacity=0.6)
    fr = opt["frontier"]
    front.add_scatter(x=fr["Volatility"], y=fr["Return"], mode="lines", name="Efficient frontier",
                      line=dict(color="#0b0b0b", width=2))
    for name, sym, c in (("Max Sharpe", "star", COLORS[1]), ("Min Variance", "diamond", COLORS[2])):
        p = opt["performance"][name]
        front.add_scatter(x=[p["Volatility"]], y=[p["Return"]], mode="markers+text", name=name, text=[name],
                          textposition="top center", marker=dict(size=16, symbol=sym, color=c,
                                                                  line=dict(color="white", width=2)))
    front.update_layout(title="Efficient frontier", xaxis_title="Volatility (σ)", yaxis_title="Expected return",
                        xaxis_tickformat=".0%", yaxis_tickformat=".0%")

    dd = go.Figure(layout=LAYOUT)
    dd.add_scatter(x=res["drawdown"].index, y=res["drawdown"], fill="tozeroy", name="Drawdown",
                   line=dict(color=BAD, width=1.5))
    dd.update_layout(title="Max-Sharpe portfolio drawdown", yaxis_tickformat=".0%")

    risk_df = pd.DataFrame(res["risk"]).T.reset_index().rename(columns={"index": "Asset"})
    pct_cols = [c for c in risk_df.columns if any(k in c for k in ("Return", "Volatility", "VaR", "Drawdown"))]
    risk_tbl = dash_table.DataTable(
        data=risk_df.to_dict("records"), style_table={"overflowX": "auto"},
        columns=[{"name": c, "id": c, "type": "numeric",
                  "format": FormatTemplate.percentage(2) if c in pct_cols else {"specifier": ".2f"}}
                 if c != "Asset" else {"name": c, "id": c} for c in risk_df.columns],
        style_cell={"fontFamily": "system-ui", "padding": "6px"}, style_header={"fontWeight": "600"})

    sig_rows = [{"Ticker": t, "Signal": f.signal["signal"], "Confidence": f"{f.signal['confidence']:.0%}",
                 "Pred. Open": f"${f.next_day['pred_open']:.2f}", "Pred. Close": f"${f.next_day['pred_close']:.2f}",
                 "Pred. Return": f"{f.next_day['pred_return']:+.2%}", "Model": f.signal["model"],
                 "As of": f.signal["as_of"]} for t, f in res["forecasts"].items()]

    ms = opt["performance"]["Max Sharpe"]
    app.layout = html.Div([
        html.H1("AI-Driven Portfolio Optimization & Market Direction Prediction"),
        html.Div(f"Data: {res['source']} · Risk-free {res['risk_free']:.2%} · "
                 f"{res['prices'].index[0].date()} → {res['prices'].index[-1].date()}", className="sub"),
        html.Div([card("Expected return", f"{ms['Return']:.1%}", "Max-Sharpe, annualised"),
                  card("Volatility", f"{ms['Volatility']:.1%}", "annualised σ"),
                  card("Sharpe ratio", f"{ms['Sharpe']:.2f}", "(Rp − Rf) / σp"),
                  card("VaR 95%", f"{res['risk']['Max Sharpe Portfolio']['VaR 95% (1d)']:.2%}", "1-day loss")],
                 className="cards"),
        dcc.Tabs([
            dcc.Tab(label="Portfolio", children=[dcc.Graph(figure=growth), dcc.Graph(figure=alloc)]),
            dcc.Tab(label="Stock forecast", children=[
                dcc.Dropdown(tk, tk[0], id="tick", clearable=False, style={"width": 220, "margin": "12px 0"}),
                html.Div(id="next-day", className="cards"),
                dcc.Graph(id="price-fig"), dcc.Graph(id="pred-fig"), dcc.Graph(id="osc-fig"),
                html.H4("Model comparison (chronological hold-out)"), html.Div(id="metrics")]),
            dcc.Tab(label="Bull / Bear signal", children=[
                dash_table.DataTable(data=sig_rows, columns=[{"name": c, "id": c} for c in sig_rows[0]],
                                     style_cell={"fontFamily": "system-ui", "padding": "8px"},
                                     style_data_conditional=[
                                         {"if": {"filter_query": '{Signal} = "Bullish"', "column_id": "Signal"},
                                          "color": GOOD, "fontWeight": "600"},
                                         {"if": {"filter_query": '{Signal} = "Bearish"', "column_id": "Signal"},
                                          "color": BAD, "fontWeight": "600"}]),
                html.P("Confidence is the classifier's probability for the predicted side. Compare with the "
                       "hold-out accuracy on the Stock forecast tab before acting on it.", className="note")]),
            dcc.Tab(label="Efficient frontier", children=[dcc.Graph(figure=front, style={"height": 600})]),
            dcc.Tab(label="Daily gain estimator", children=[
                html.Div([html.Label("Capital ($)"), dcc.Input(id="cap", type="number", value=10_000, step=1000)],
                         style={"margin": "12px 0"}),
                html.Div(id="gain-out"),
                html.P("Speculative: a point forecast from the models above, weighted by the Max-Sharpe "
                       "allocation. The ±1 sd band and VaR show how wide the realistic range is.", className="note")]),
            dcc.Tab(label="Risk dashboard", children=[risk_tbl, dcc.Graph(figure=dd)]),
        ]),
    ], className="wrap")

    @app.callback(Output("price-fig", "figure"), Output("pred-fig", "figure"), Output("osc-fig", "figure"),
                  Output("next-day", "children"), Output("metrics", "children"), Input("tick", "value"))
    def _ticker(t):
        ind = res["indicators"][t].iloc[-250:]
        f1 = go.Figure(layout=LAYOUT)
        f1.add_scatter(x=ind.index, y=ind["BB_Upper"], line=dict(width=0), showlegend=False, hoverinfo="skip")
        f1.add_scatter(x=ind.index, y=ind["BB_Lower"], fill="tonexty", fillcolor="rgba(42,120,214,0.10)",
                       line=dict(width=0), name="Bollinger band")
        f1.add_scatter(x=ind.index, y=ind["Close"], name="Close", line=dict(color=color[t], width=2))
        f1.add_scatter(x=ind.index, y=ind["SMA20"], name="SMA20", line=dict(color=COLORS[1], width=1.5))
        f1.add_scatter(x=ind.index, y=ind["SMA50"], name="SMA50", line=dict(color=COLORS[2], width=1.5))
        f1.update_layout(title=f"{t} price, moving averages & Bollinger bands")

        fc = res["forecasts"][t]
        tp = fc.test_predictions.iloc[-120:]
        f2 = go.Figure(layout=LAYOUT)
        f2.add_scatter(x=tp.index, y=tp["Actual Close"], name="Actual", line=dict(color="#0b0b0b", width=2))
        for i, c in enumerate([c for c in tp.columns if c not in ("Actual Close", "Naive")]):
            f2.add_scatter(x=tp.index, y=tp[c], name=c, line=dict(color=COLORS[i], width=1.5))
        f2.update_layout(title="Next-day close: predicted vs actual (hold-out)")

        f3 = go.Figure(layout=LAYOUT)
        f3.add_scatter(x=ind.index, y=ind["RSI14"], name="RSI 14", line=dict(color=COLORS[6], width=2))
        f3.add_scatter(x=ind.index, y=ind["Stoch_K"], name="Stochastic %K", line=dict(color=COLORS[3], width=1.5))
        for lvl in (30, 70):
            f3.add_hline(y=lvl, line=dict(color=MUTED, dash="dot", width=1))
        f3.update_layout(title="Momentum oscillators", yaxis_range=[0, 100])

        nd, s = fc.next_day, fc.signal
        tone = GOOD if s["signal"] == "Bullish" else BAD
        cards = [card("Signal", ("▲ " if s["signal"] == "Bullish" else "▼ ") + s["signal"],
                      f"confidence {s['confidence']:.0%}", tone),
                 card("Predicted open", f"${nd['pred_open']:.2f}", f"last close ${nd['last_close']:.2f}"),
                 card("Predicted close", f"${nd['pred_close']:.2f}", f"{nd['pred_return']:+.2%} · {nd['model']}")]
        tbl = lambda df: dash_table.DataTable(  # noqa: E731
            data=df.reset_index().round(4).to_dict("records"),
            columns=[{"name": c, "id": c} for c in df.reset_index().columns],
            style_cell={"fontFamily": "system-ui", "padding": "6px"})
        return f1, f2, f3, cards, html.Div([tbl(fc.regression_metrics), html.Br(), tbl(fc.classification_metrics)])

    @app.callback(Output("gain-out", "children"), Input("cap", "value"))
    def _gain(cap):
        cap = cap or 0
        w = opt["weights"]["Max Sharpe"]
        pr = sum(w[t] * res["forecasts"][t].next_day["pred_return"] for t in tk)
        sd = sum((w[t] * res["forecasts"][t].next_day["return_sd"]) ** 2 for t in tk) ** 0.5
        g = estimate_gain([cap, 10_000, 50_000, 100_000], pr, sd,
                          res["risk"]["Max Sharpe Portfolio"]["VaR 95% (1d)"]).drop_duplicates("Capital")
        money = lambda v: f"-${abs(v):,.2f}" if v < -0.005 else f"${abs(v):,.2f}"  # noqa: E731
        g = g.assign(**{c: g[c].map(money) for c in g.columns if c != "Predicted Return"})
        g["Predicted Return"] = g["Predicted Return"].map(lambda v: f"{v:+.2%}")
        return dash_table.DataTable(data=g.to_dict("records"), columns=[{"name": c, "id": c} for c in g.columns],
                                    style_cell={"fontFamily": "system-ui", "padding": "8px"})

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--port", type=int, default=8050)
    a = ap.parse_args()
    try:
        results = run(config.TICKERS, offline=a.offline, max_weight=config.MAX_WEIGHT, progress=print)
    except Exception as e:  # network down, rate-limited, etc.
        print(f"Live download failed ({e}); falling back to SYNTHETIC sample data.")
        results = run(config.TICKERS, offline=True, max_weight=config.MAX_WEIGHT, progress=print)
    print(f"Dashboard ready -> http://127.0.0.1:{a.port}")
    build_app(results).run(debug=False, port=a.port)
