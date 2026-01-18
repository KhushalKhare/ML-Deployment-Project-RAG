import pandas as pd
import numpy as np
from training.features import make_features_at
from training.baselines import seasonal_naive_forecast

def forecast_horizon(series: pd.Series, model, horizon: int) -> pd.DataFrame:
    """
    Recursive multi-step forecast using lag features.
    Returns DataFrame: timestamp, yhat, baseline.
    """
    s = series.sort_index().asfreq("h").copy()
    last_ts = s.index.max()

    future_index = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1),
        periods=horizon,
        freq="h",
        tz=s.index.tz
    )

    preds = []
    for ts in future_index:
        x = make_features_at(s, ts).to_frame().T
        yhat = float(model.predict(x)[0])
        preds.append(yhat)
        s.loc[ts] = yhat  # feed prediction back for future lags

    baseline = seasonal_naive_forecast(series, horizon=horizon).values.astype(float)

    return pd.DataFrame({
        "timestamp": future_index,
        "yhat": np.array(preds, dtype=float),
        "baseline": baseline
    })
