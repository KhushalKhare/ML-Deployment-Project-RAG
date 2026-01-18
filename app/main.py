import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from training.forecast import forecast_horizon

DATA_PATH = Path("data/processed/generation_de_hourly.csv")
MODEL_PATH = Path("models/model.pkl")
META_PATH = Path("models/metadata.json")

app = FastAPI(title="SMARD Germany Generation Forecast API", version="1.0.0")

def load_series() -> pd.Series:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed data not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    return pd.Series(df["value"].values, index=df["timestamp"]).asfreq("h")

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Train first.")
    return joblib.load(MODEL_PATH)

def load_meta():
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return {}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_exists": MODEL_PATH.exists(),
        "data_exists": DATA_PATH.exists(),
        "meta_exists": META_PATH.exists(),
    }

@app.get("/metrics")
def metrics():
    return load_meta()

@app.get("/forecast")
def forecast(horizon: int = 24):
    if horizon < 1 or horizon > 24 * 31:
        raise HTTPException(status_code=400, detail="horizon must be between 1 and 744 (31 days)")
    series = load_series()
    model = load_model()
    meta = load_meta()

    df = forecast_horizon(series, model, horizon=horizon)
    return {
        "horizon": horizon,
        "model_version": meta.get("model_version"),
        "unit": "MWh (per hour interval)",
        "points": df.to_dict(orient="records"),
    }

@app.get("/forecast/month")
def forecast_month(days: int = 30):
    if days < 1 or days > 31:
        raise HTTPException(status_code=400, detail="days must be between 1 and 31")
    horizon = days * 24
    series = load_series()
    model = load_model()
    meta = load_meta()

    df = forecast_horizon(series, model, horizon=horizon)
    return {
        "days": days,
        "horizon": horizon,
        "model_version": meta.get("model_version"),
        "unit": "MWh (per hour interval)",
        "points": df.to_dict(orient="records"),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
