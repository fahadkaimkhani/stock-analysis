import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import StandardScaler
import streamlit as st  # ✅ Optional, used for displaying safe errors in Streamlit


# --- Download stock data safely ---
def get_data(ticker):
    # ✅ Added auto_adjust=True (fixes warning)
    stock_data = yf.download(ticker, start='2024-01-01', auto_adjust=True)

    # ✅ If Yahoo Finance returns nothing, return empty DataFrame
    if stock_data.empty:
        st.warning(f"No stock data found for '{ticker}'. Please try another symbol.")
        return pd.DataFrame()

    # ✅ Ensure regular business-day frequency
    stock_data = stock_data.asfreq('B')
    stock_data.dropna(inplace=True)

    # ✅ Return only the Close column
    return stock_data[['Close']]


# --- ADF Stationarity Test ---
def stationary_check(close_price):
    if close_price.empty:
        return 1  # If no data, return high p-value so differencing continues safely
    adf_test = adfuller(close_price)
    p_value = round(adf_test[1], 3)
    return p_value


# --- Apply 7-day rolling average ---
def get_rolling_mean(close_price):
    if close_price.empty:
        return pd.DataFrame()
    return close_price.rolling(window=7).mean().dropna()


# --- Determine differencing order safely ---
def get_differencing_order(close_price):
    if close_price.empty:
        return 0
    p_value = stationary_check(close_price)
    d = 0
    while p_value > 0.05 and d < 3:  # ⛔ Limit to avoid infinite loop
        d += 1
        close_price = close_price.diff().dropna()
        if close_price.empty:
            break
        p_value = stationary_check(close_price)
    return d


# --- Fit ARIMA model and forecast safely ---
def fit_model(data, differencing_order):
    if data.empty or len(data) < 10:
        # Not enough data for ARIMA
        return pd.Series(dtype=float)

    try:
        model = ARIMA(data, order=(5, differencing_order, 5))
        model_fit = model.fit()
        forecast = model_fit.get_forecast(steps=30)
        return forecast.predicted_mean
    except Exception as e:
        st.error(f"ARIMA model failed: {e}")
        return pd.Series(dtype=float)


# --- Evaluate model performance ---
def evaluate_model(original_price, differencing_order):
    if len(original_price) < 60:
        st.warning("Not enough data to evaluate model properly.")
        return 0.0

    train_data = original_price[:-30]
    test_data = original_price[-30:]
    predictions = fit_model(train_data, differencing_order)

    # ✅ Prevent mismatch if prediction is empty or shorter
    if predictions.empty or len(predictions) != len(test_data):
        return 0.0

    rmse = np.sqrt(mean_squared_error(test_data, predictions))
    return round(rmse, 2)


# --- Forecast the next 30 days safely ---
def get_forecast(original_price, differencing_order):
    predictions = fit_model(original_price, differencing_order)
    if predictions.empty:
        return pd.DataFrame()

    start_date = datetime.now().date()
    forecast_index = pd.date_range(start=start_date, periods=30, freq='D')
    forecast_df = pd.DataFrame(predictions.values, index=forecast_index, columns=['Close'])
    return forecast_df


# --- Scaling support (optional) ---
def scaling(close_price):
    if close_price.empty:
        return None, None
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(np.array(close_price).reshape(-1, 1))
    return scaled_data, scaler


def inverse_scaling(scaler, scaled_data):
    if scaler is None or scaled_data is None:
        return np.array([])
    return scaler.inverse_transform(np.array(scaled_data).reshape(-1, 1))
