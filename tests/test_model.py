"""Tests for Time Series Forecasting — dual-dataset."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_datacenter_model_exists():
    assert (ROOT / "models" / "arima_datacenter.pkl").exists()


def test_silicon_aging_model_exists():
    assert (ROOT / "models" / "arima_silicon_aging.pkl").exists()


def test_datacenter_forecast():
    from predict import predict
    preds = predict(steps=5, model_path=str(ROOT / "models" / "arima_datacenter.pkl"))
    assert len(preds) == 5


def test_silicon_aging_forecast():
    from predict import predict
    preds = predict(steps=5, model_path=str(ROOT / "models" / "arima_silicon_aging.pkl"))
    assert len(preds) == 5


if __name__ == "__main__":
    test_datacenter_model_exists()
    test_silicon_aging_model_exists()
    test_datacenter_forecast()
    test_silicon_aging_forecast()
    print("All 4 tests passed.")
