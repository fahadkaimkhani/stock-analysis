import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

# --- Download stock data ---
def get_data(ticker):
    stock_data = yf.download(ticker, start='2024-01-01')
    return stock_data[['Close']]  # ✅ Correct case

# --- ADF Stationarity Test ---
def stationary_check(close_price):
    adf_test = adfuller(close_price)
    p_value = round(adf_test[1], 3)
    return p_value

# --- Apply 7-day rolling average ---
def get_rolling_mean(close_price):
    return close_price.rolling(window=7).mean().dropna()

# --- Determine differencing order ---
def get_differencing_order(close_price):
    p_value = stationary_check(close_price)
    d = 0
    while p_value > 0.05 and d < 3:  # ⛔ Add limit to prevent infinite loop
        d += 1
        close_price = close_price.diff().dropna()
        p_value = stationary_check(close_price)
    return d

# --- Fit ARIMA model and forecast ---
def fit_model(data, differencing_order):
    model = ARIMA(data, order=(5, differencing_order, 5))  # ✅ (5, d, 5) is fast & stable
    model_fit = model.fit()
    forecast = model_fit.get_forecast(steps=30)
    return forecast.predicted_mean

# --- Evaluate model performance ---
def evaluate_model(original_price, differencing_order):
    train_data = original_price[:-30]
    test_data = original_price[-30:]
    predictions = fit_model(train_data, differencing_order)
    predictions = predictions[:len(test_data)]  # Match lengths
    rmse = np.sqrt(mean_squared_error(test_data, predictions))
    return round(rmse, 2)

# --- Forecast the next 30 days ---
def get_forecast(original_price, differencing_order):
    predictions = fit_model(original_price, differencing_order)
    start_date = datetime.now().date()
    forecast_index = pd.date_range(start=start_date, periods=30, freq='D')
    forecast_df = pd.DataFrame(predictions.values, index=forecast_index, columns=['Close'])
    return forecast_df

# --- Scaling support (not currently used) ---
from sklearn.preprocessing import StandardScaler
def scaling(close_price):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(np.array(close_price).reshape(-1, 1))
    return scaled_data, scaler

def inverse_scaling(scaler, scaled_data):
    return scaler.inverse_transform(np.array(scaled_data).reshape(-1, 1))


