import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error

DATA = Path("data/processed/generation_de_hourly.csv")

def seasonal_naive(series: pd.Series, horizon: int = 168, season_hours: int = 168) -> pd.Series:
    series = series.sort_index().asfreq("H")
    last_ts = series.index.max()
    future_index = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=horizon, freq="H", tz=series.index.tz)

    yhat = []
    for ts in future_index:
        yhat.append(series.get(ts - pd.Timedelta(hours=season_hours)))
    yhat = pd.Series(yhat, index=future_index).ffill()
    if yhat.isna().any():
        yhat = yhat.fillna(series.dropna().iloc[-1])
    return yhat

def main():
    df = pd.read_csv(DATA)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    series = pd.Series(df["value"].values, index=df["timestamp"]).asfreq("H")

    if len(series) < 24 * 14:
        raise ValueError(f"Need >= 336 hourly points (14 days). You have {len(series)}.")

    train = series.iloc[:-168]
    test = series.iloc[-168:]

    forecast = seasonal_naive(train, horizon=168)

    mae = mean_absolute_error(test.values, forecast.values)
    print("Test window:", test.index.min(), "->", test.index.max())
    print("Baseline MAE:", round(mae, 2))

if __name__ == "__main__":
    main()
