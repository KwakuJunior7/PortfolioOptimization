"""Module 8 - Speculative daily gain estimator.

gain = capital x predicted_return. Because a point forecast hides how uncertain
it is, the estimator also returns a 1-sigma band (from the model's residual
standard deviation) and the 1-day 95% historical VaR loss for the same capital.
"""
from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


def estimate_gain(capitals: Iterable[float], pred_return: float, return_sd: float = 0.0,
                  var95: Optional[float] = None) -> pd.DataFrame:
    rows = []
    for c in capitals:
        row = {"Capital": c, "Predicted Return": pred_return, "Predicted Gain": c * pred_return,
               "Low (-1 sd)": c * (pred_return - return_sd), "High (+1 sd)": c * (pred_return + return_sd)}
        if var95 is not None:
            row["1-day VaR 95% loss"] = -c * var95
        rows.append(row)
    return pd.DataFrame(rows)
