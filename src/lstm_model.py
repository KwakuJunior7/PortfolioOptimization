"""Optional LSTM forecaster (requires tensorflow).

Learns next-day close return from a rolling window of the scale-free features.
On daily equity data LSTMs rarely beat a well-regularised tree model or even
the naive baseline out of sample - compare its hold-out RMSE before trusting it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .features import FEATURES, build_features, split_xy


def make_windows(X: np.ndarray, y: np.ndarray, window: int):
    xs = np.stack([X[i - window:i] for i in range(window, len(X) + 1)])
    return xs, y[window - 1:]


def train_lstm(df: pd.DataFrame, market=None, window: int = 30, test_size: float = 0.2,
               epochs: int = 30, save_path: str | None = None):
    from tensorflow.keras import Sequential, callbacks
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

    frame, _ = split_xy(build_features(df, market))
    cut = int(len(frame) * (1 - test_size))
    scaler = StandardScaler().fit(frame[FEATURES].iloc[:cut])
    X = scaler.transform(frame[FEATURES])
    y = frame["y_close_ret"].to_numpy()
    Xw, yw = make_windows(X, y, window)
    split = cut - window + 1
    model = Sequential([Input((window, len(FEATURES))), LSTM(32, return_sequences=True), Dropout(0.2),
                        LSTM(16), Dropout(0.2), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(Xw[:split], yw[:split], validation_split=0.1, epochs=epochs, batch_size=32, verbose=0,
              callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    pred = model.predict(Xw[split:], verbose=0).ravel()
    closes = frame["Close"].to_numpy()[window - 1:][split:]
    actual = closes * (1 + yw[split:])
    rmse = float(np.sqrt(np.mean((actual - closes * (1 + pred)) ** 2)))
    naive = float(np.sqrt(np.mean((actual - closes) ** 2)))
    if save_path:
        model.save(save_path)
    return model, {"LSTM Close RMSE ($)": rmse, "Naive Close RMSE ($)": naive,
                   "Direction hit-rate": float((np.sign(pred) == np.sign(yw[split:])).mean())}
