import numpy as np
import pandas as pd

def _time_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame(index=idx)
    hour = idx.hour
    dow = idx.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"]  = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"]  = np.cos(2 * np.pi * dow / 7.0)
    df["month"] = idx.month
    return df

def make_supervised(series: pd.Series) -> pd.DataFrame:
    """
    Create supervised dataset using only past info (no leakage).
    """
    s = series.sort_index().asfreq("h")
    df = pd.DataFrame(index=s.index)
    df["y"] = s

    df["lag_1"] = s.shift(1)
    df["lag_24"] = s.shift(24)
    df["lag_168"] = s.shift(168)

    df["roll_mean_24"] = s.shift(1).rolling(24).mean()
    df["roll_mean_168"] = s.shift(1).rolling(168).mean()
    df["roll_std_24"] = s.shift(1).rolling(24).std()

    df = pd.concat([df, _time_features(df.index)], axis=1)
    return df.dropna()

def make_features_at(series: pd.Series, ts: pd.Timestamp) -> pd.Series:
    """
    Build one feature row for a specific timestamp ts using series history.
    Used for iterative multi-step forecasting.
    """
    s = series.sort_index().asfreq("h")

    feats = {
        "lag_1": s.get(ts - pd.Timedelta(hours=1), np.nan),
        "lag_24": s.get(ts - pd.Timedelta(hours=24), np.nan),
        "lag_168": s.get(ts - pd.Timedelta(hours=168), np.nan),
        "roll_mean_24": s.loc[:ts - pd.Timedelta(hours=1)].tail(24).mean(),
        "roll_mean_168": s.loc[:ts - pd.Timedelta(hours=1)].tail(168).mean(),
        "roll_std_24": s.loc[:ts - pd.Timedelta(hours=1)].tail(24).std(),
    }
    tf = _time_features(pd.DatetimeIndex([ts]))
    for c in tf.columns:
        feats[c] = float(tf.iloc[0][c])

    return pd.Series(feats)
