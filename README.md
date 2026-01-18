# 🇩🇪 German Electricity Generation Forecasting (SMARD)

An end-to-end machine learning system that forecasts **hourly German electricity generation** using official government data.  
The project focuses on **real-world ML workflows**: data cleaning, baseline benchmarking, model training, API deployment, and Dockerized delivery.

---

## Problem Statement

Electricity generation changes every hour and follows strong daily and weekly patterns.  
Simple rule-based approaches (e.g. copying last week’s value) often fail during holidays or irregular periods.

**Goal:**  
Build a system that learns patterns from historical data and produces **reliable short- to medium-term forecasts** that can be consumed by other applications.

---

## Data Source

- **SMARD – Bundesnetzagentur (Germany)**
- Official German electricity market data
- Hourly resolution
- Actual electricity generation (aggregated across sources)

This project uses a cleaned and aggregated time series derived from SMARD CSV exports.

---

##  Approach

### 1. Data Processing
- Parsed raw SMARD CSV exports
- Handled missing values and European number formats
- Aggregated all generation sources into a single hourly time series

### 2. Baseline
- **Seasonal Naive Baseline**  
  Forecast = same hour from the previous week (t−168)
- Used as a reference to ensure ML adds real value

### 3. Feature Engineering
The model uses only past information (no leakage):
- Lag features: 1 hour, 24 hours, 168 hours
- Rolling statistics: mean & standard deviation
- Cyclical time features (hour of day, day of week, month)

### 4. Model
- **HistGradientBoostingRegressor**
- Chosen for:
  - strong performance on tabular data
  - robustness with limited data
  - fast training and explainability

### 5. Forecasting Strategy
- **Recursive multi-step forecasting**
- Each predicted hour is fed back to generate the next prediction
- Supports horizons up to **30 days (720 hours)**

---

## Results

| Metric | Value |
|------|------|
| Baseline MAE | ~67,000 MWh |
| ML Model MAE | ~31,000 MWh |
| Improvement | ~54% |

The ML model significantly outperforms the seasonal-naive baseline on held-out data.

---

##  API (FastAPI)

The trained model is served via a FastAPI application.

### Available Endpoints
- `GET /health` – service status
- `GET /metrics` – model performance & metadata
- `GET /forecast?horizon=24` – hourly forecast
- `GET /forecast/month?days=30` – 30-day hourly forecast
- `GET /docs` – interactive Swagger UI

The API returns both:
- the ML prediction
- the baseline prediction (for comparison)

---

##  Docker Support

The entire application is containerized.

### Build
```bash
docker build -t energy-forecast .
