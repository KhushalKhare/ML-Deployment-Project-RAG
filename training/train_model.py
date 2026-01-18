import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from training.features import make_supervised
from training.baselines import seasonal_naive_forecast

DATA = Path("data/processed/generation_de_hourly.csv")
MODEL_PATH = Path("models/model.pkl")
META_PATH = Path("models/metadata.json")

def main():
    df = pd.read_csv(DATA)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    series = pd.Series(df["value"].values, index=df["timestamp"]).asfreq("h")

    sup = make_supervised(series)
    X = sup.drop(columns=["y"])
    y = sup["y"]

    # Hold out last week for evaluation
    X_train, X_test = X.iloc[:-168], X.iloc[-168:]
    y_train, y_test = y.iloc[:-168], y.iloc[-168:]

    model = HistGradientBoostingRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    baseline = seasonal_naive_forecast(series.iloc[:-168], horizon=168)
    mae_model = float(mean_absolute_error(y_test.values, y_pred))
    mae_base = float(mean_absolute_error(y_test.values, baseline.values))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    meta = {
        "model_version": "v1",
        "data_file": str(DATA),
        "train_end": str(X_train.index.max()),
        "test_start": str(X_test.index.min()),
        "test_end": str(X_test.index.max()),
        "mae_model": mae_model,
        "mae_baseline": mae_base,
        "improvement_pct": float((mae_base - mae_model) / mae_base * 100.0),
        "features": list(X.columns),
        "notes": "HGBR on lag + rolling + calendar features. Frequency hourly."
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
