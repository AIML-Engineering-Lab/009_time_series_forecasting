"""
Train pipeline for Time Series Forecasting — dual-dataset.
Project 1: Datacenter Power Demand  (ARIMA)
Project 2: Silicon Aging Vth Shift  (ARIMA)
"""
import pandas as pd
import joblib
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

DATASETS = {
    "datacenter": {
        "file": "datacenter_power_demand.csv",
        "target": "power_kw",
        "order": (5, 1, 0),
        "model_name": "arima_datacenter.pkl",
    },
    "silicon_aging": {
        "file": "silicon_aging_vth.csv",
        "target": "vth_shift_mv",
        "order": (5, 1, 0),
        "model_name": "arima_silicon_aging.pkl",
    },
}


def train(key: str):
    cfg = DATASETS[key]
    print(f"\n=== Training {key} ===")
    df = pd.read_csv(DATA_DIR / cfg["file"])
    print(f"  Rows: {len(df)}")

    y = df[cfg["target"]].dropna().values
    print(f"  Fitting ARIMA{cfg['order']}...")
    model = ARIMA(y, order=cfg["order"])
    result = model.fit()

    print(f"  AIC: {result.aic:.2f}")

    model_path = MODEL_DIR / cfg["model_name"]
    joblib.dump(result, model_path)
    print(f"  Saved → {model_path}")
    return result


if __name__ == "__main__":
    for k in DATASETS:
        train(k)
