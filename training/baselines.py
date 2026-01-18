# training/baselines.py
import numpy as np
import pandas as pd

def seasonal_naive_forecast(
    series: pd.Series,
    horizon: int,
    season_hours: int = 168,
) -> pd.Series:
    """
    Forecast next `horizon` hours using the value from `season_hours` ago
    (default: same hour last week).
    """
    series = series.sort_index()

    last_ts = series.index.max()
    future_index = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1),
        periods=horizon,
        freq="H",
        tz=series.index.tz
    )

    preds = []
    for ts in future_index:
        ref_ts = ts - pd.Timedelta(hours=season_hours)
        preds.append(series.get(ref_ts, np.nan))

    yhat = pd.Series(preds, index=future_index)

    # Fallbacks if the season reference is missing:
    # 1) forward fill inside yhat
    # 2) use last observed value
    yhat = yhat.ffill()
    if yhat.isna().any():
        yhat = yhat.fillna(series.dropna().iloc[-1])

    return yhat
