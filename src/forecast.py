"""Modules 3 & 4 - Next-day price prediction and bullish/bearish classification.

* Regression: tomorrow's Open and Close (Linear Regression, Random Forest,
  XGBoost, optional LSTM). Models learn next-day returns; predictions are
  converted to prices. Every model is compared with a naive baseline
  (tomorrow = today's close) on a chronological hold-out - no shuffling, so
  there is no look-ahead leakage.
* Classification: P(Close[t+1] > Close[t]) with Random Forest / XGBoost /
  Logistic Regression, reported against the majority-class baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES, build_features, split_xy

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:  # pragma: no cover - optional dependency
    HAS_XGB = False


def regressors(seed: int = 42) -> Dict[str, object]:
    m = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=20,
                                               n_jobs=-1, random_state=seed),
    }
    if HAS_XGB:
        m["XGBoost"] = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                    subsample=0.8, colsample_bytree=0.8, random_state=seed)
    return m


def classifiers(seed: int = 42) -> Dict[str, object]:
    m = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)),
        "Random Forest": RandomForestClassifier(n_estimators=400, max_depth=5, min_samples_leaf=25,
                                                n_jobs=-1, random_state=seed),
    }
    if HAS_XGB:
        m["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03,
                                     subsample=0.8, colsample_bytree=0.8, random_state=seed,
                                     eval_metric="logloss")
    return m


def chrono_split(df: pd.DataFrame, test_size: float = 0.2):
    cut = int(len(df) * (1 - test_size))
    return df.iloc[:cut], df.iloc[cut:]


@dataclass
class ForecastResult:
    ticker: str
    regression_metrics: pd.DataFrame
    classification_metrics: pd.DataFrame
    next_day: Dict[str, float]
    signal: Dict[str, object]
    test_predictions: pd.DataFrame = field(repr=False)
    feature_importance: pd.Series = field(repr=False)
    models: Dict[str, object] = field(default_factory=dict, repr=False)


def _rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def run_forecast(df: pd.DataFrame, ticker: str, market: Optional[pd.DataFrame] = None,
                 test_size: float = 0.2, seed: int = 42) -> ForecastResult:
    frame = build_features(df, market)
    labelled, latest = split_xy(frame)
    train, test = chrono_split(labelled, test_size)
    Xtr, Xte = train[FEATURES], test[FEATURES]

    # ---------- regression: next open / next close ----------
    rows, preds = [], pd.DataFrame(index=test.index)
    preds["Actual Close"] = test["Close"] * (1 + test["y_close_ret"])
    preds["Naive"] = test["Close"]
    actual_open = test["Close"] * (1 + test["y_open_ret"])
    actual_dir = np.sign(test["y_close_ret"])
    rows.append({"Model": "Naive baseline (tomorrow = today)",
                 "Close RMSE ($)": _rmse(preds["Actual Close"], test["Close"]),
                 "Close MAE ($)": float(mean_absolute_error(preds["Actual Close"], test["Close"])),
                 "Open RMSE ($)": _rmse(actual_open, test["Close"]),
                 "Direction hit-rate": np.nan})
    best_close_model, best_rmse = None, np.inf
    for name, model in regressors(seed).items():
        mc = model.fit(Xtr, train["y_close_ret"])
        pc = mc.predict(Xte)
        close_px = test["Close"] * (1 + pc)
        rmse_c = _rmse(preds["Actual Close"], close_px)
        mo = clone(model)
        mo.fit(Xtr, train["y_open_ret"])
        open_px = test["Close"] * (1 + mo.predict(Xte))
        preds[name] = close_px
        rows.append({"Model": name, "Close RMSE ($)": rmse_c,
                     "Close MAE ($)": float(mean_absolute_error(preds["Actual Close"], close_px)),
                     "Open RMSE ($)": _rmse(actual_open, open_px),
                     "Direction hit-rate": float((np.sign(pc) == actual_dir).mean())})
        if rmse_c < best_rmse:
            best_rmse, best_close_model = rmse_c, (name, model, mo)
    reg_metrics = pd.DataFrame(rows).set_index("Model")

    # refit the best regressor on all labelled data for tomorrow's estimate
    name, mc, mo = best_close_model
    mc.fit(labelled[FEATURES], labelled["y_close_ret"])
    mo.fit(labelled[FEATURES], labelled["y_open_ret"])
    last_close = float(latest["Close"].iloc[0])
    pred_close_ret = float(mc.predict(latest[FEATURES])[0])
    pred_open_ret = float(mo.predict(latest[FEATURES])[0])
    resid_sd = float(np.std(labelled["y_close_ret"] - mc.predict(labelled[FEATURES])))
    next_day = {"model": name, "as_of": str(latest.index[0].date()), "last_close": last_close,
                "pred_open": last_close * (1 + pred_open_ret),
                "pred_close": last_close * (1 + pred_close_ret),
                "pred_return": pred_close_ret, "return_sd": resid_sd}

    # ---------- classification: bullish / bearish ----------
    rows = [{"Model": "Majority-class baseline",
             "Accuracy": float(max(test["y_direction"].mean(), 1 - test["y_direction"].mean())),
             "ROC AUC": 0.5}]
    best_clf, best_auc = None, -np.inf
    for cname, clf in classifiers(seed).items():
        clf.fit(Xtr, train["y_direction"].astype(int))
        p = clf.predict_proba(Xte)[:, 1]
        auc = float(roc_auc_score(test["y_direction"].astype(int), p))
        rows.append({"Model": cname, "Accuracy": float(accuracy_score(test["y_direction"], p > 0.5)),
                     "ROC AUC": auc})
        if auc > best_auc:
            best_auc, best_clf = auc, (cname, clf)
    clf_metrics = pd.DataFrame(rows).set_index("Model")

    cname, clf = best_clf
    clf.fit(labelled[FEATURES], labelled["y_direction"].astype(int))
    p_bull = float(clf.predict_proba(latest[FEATURES])[0, 1])
    signal = {"model": cname, "as_of": next_day["as_of"],
              "signal": "Bullish" if p_bull >= 0.5 else "Bearish",
              "confidence": max(p_bull, 1 - p_bull), "p_bullish": p_bull}

    imp_model = RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=20,
                                      n_jobs=-1, random_state=seed).fit(Xtr, train["y_close_ret"])
    importance = pd.Series(imp_model.feature_importances_, index=FEATURES).sort_values(ascending=False)

    models = {"close_regressor": mc, "open_regressor": mo, "direction_classifier": clf,
              "features": list(FEATURES)}
    return ForecastResult(ticker, reg_metrics, clf_metrics, next_day, signal, preds, importance, models)
