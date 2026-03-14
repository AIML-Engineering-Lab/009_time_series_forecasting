# Post 009 — Time Series Forecasting: ARIMA, SARIMA, Stationarity

**AI Engineering Lab Series** | Era 1: Classic Machine Learning

## Overview

This project demonstrates time series analysis and forecasting using ARIMA/SARIMA — covering stationarity testing, decomposition, ACF/PACF analysis, and multi-horizon forecasting.

| Concept | Description |
|---|---|
| **Stationarity** | A time series whose statistical properties do not change over time |
| **ADF Test** | Augmented Dickey-Fuller test for stationarity (p < 0.05 = stationary) |
| **Differencing** | Removing trend by subtracting consecutive values (d parameter) |
| **ACF/PACF** | Autocorrelation and Partial Autocorrelation — used to choose p and q |
| **ARIMA(p,d,q)** | AutoRegressive Integrated Moving Average model |
| **SARIMA** | ARIMA with seasonal component (P,D,Q,s) |

## Datasets

### Dataset A: Datacenter Power Demand
- **Records:** 17,520 hourly readings (2 years) | **Target:** Power (kW)
- **Patterns:** Daily, weekly, and annual seasonality + long-term trend

### Dataset B: Silicon Aging — Threshold Voltage Shift (Post-Silicon Validation)
- **Records:** 1,095 daily readings (3 years) | **Target:** Vth shift (mV)
- **Patterns:** Logarithmic BTI degradation + temperature-driven seasonal acceleration

## Quick Start

```bash
git clone https://github.com/AIML-Engineering-Lab/009_time_series_forecasting.git
cd 009_time_series_forecasting
pip install -r requirements.txt
python src/data_generator.py
jupyter notebook notebooks/
```

*Part of the [AI Engineering Lab](https://github.com/AIML-Engineering-Lab) series.*
