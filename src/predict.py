"""
Inference for Time Series Forecasting — dual-dataset.
Load trained ARIMA model and forecast future values.
"""
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"

DATASETS = {
    "datacenter": {"model_name": "arima_datacenter.pkl"},
    "silicon_aging": {"model_name": "arima_silicon_aging.pkl"},
}


def predict(steps: int = 10, model_path: str = None) -> list:
    if model_path is None:
        model_path = str(MODEL_DIR / "arima_datacenter.pkl")
    model = joblib.load(model_path)
    forecast = model.forecast(steps=steps)
    return forecast.tolist()


if __name__ == "__main__":
    for key, cfg in DATASETS.items():
        preds = predict(steps=10, model_path=str(MODEL_DIR / cfg["model_name"]))
        print(f"{key} (next 10): {[f'{v:.2f}' for v in preds]}")
