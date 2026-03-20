"""
FastAPI serving endpoint for Time Series Forecasting.
POST forecast request -> future predictions (datacenter ARIMA by default).
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from pathlib import Path

app = FastAPI(title="Time Series Forecasting API", version="1.0.0")

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "arima_datacenter.pkl"
_model = None


class ForecastRequest(BaseModel):
    steps: int = 10


class ForecastResponse(BaseModel):
    forecast: list[float]
    model: str = "ARIMA"


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health")
def health():
    return {"status": "healthy", "model": "ARIMA"}


@app.get("/info")
def info():
    return {"project": "009_time_series_forecasting", "description": "Time Series Forecasting", "task": "time_series_forecast"}


@app.post("/predict", response_model=ForecastResponse)
def predict(req: ForecastRequest):
    try:
        model = get_model()
        forecast = model.forecast(steps=req.steps)
        return ForecastResponse(forecast=forecast.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
