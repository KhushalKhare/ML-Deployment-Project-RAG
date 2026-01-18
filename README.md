# 🇩🇪 German Electricity Forecasting (SMARD)

End-to-end machine learning system that forecasts hourly German electricity generation using official SMARD (Bundesnetzagentur) data.

## Key Highlights
- Official German government energy data (SMARD)
- Seasonal-naive baseline benchmarking
- ML model (HistGradientBoosting) with ~54% MAE improvement
- Recursive multi-step forecasting (up to 30 days)
- FastAPI service with documented endpoints
- Fully Dockerized for reproducible deployment

## API Preview
- `/health`
- `/metrics`
- `/forecast?horizon=24`
- `/forecast/month?days=30`
## Why this project
This project focuses on production-oriented machine learning:
- reproducible data pipelines
- baseline-first evaluation
- explainable feature engineering
- deployable API design

The goal was not maximum accuracy, but a clean, end-to-end system that mirrors real-world ML workflows.
