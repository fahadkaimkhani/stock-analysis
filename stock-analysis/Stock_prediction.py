import streamlit as st
import pandas as pd

from utils.model_train import (
    get_data,
    get_rolling_mean,
    get_differencing_order,
    scaling,
    evaluate_model,
    get_forecast,
    inverse_scaling
)

from utils.plotly_figure import (
    plotly_table,
    Moving_average  # ✅ Correct function name
)

st.set_page_config(
    page_title="Stock Prediction",
    page_icon="📉",
    layout="wide"
)

st.title("📈 Stock Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    ticker = st.text_input("Stock Ticker", "AAPL")

st.subheader(f"Predicting Next 30 Days Close Price for: {ticker}")

# Step 1: Load data and smooth it
close_price = get_data(ticker)
# ✅ Safety checks before continuing
if close_price is None or close_price.empty:
    st.error("⚠️ No stock data found for this ticker. Please enter a valid ticker symbol (e.g., AAPL, MSFT, TSLA).")
    st.stop()

if len(close_price) < 30:
    st.warning("⚠️ Not enough data available for analysis. Please try another stock or a wider date range.")
    st.stop()

rolling_price = get_rolling_mean(close_price)

# Step 2: Model prep
differencing_order = get_differencing_order(rolling_price)
rmse = evaluate_model(rolling_price, differencing_order)

# Step 3: Forecast future
forecast = get_forecast(rolling_price, differencing_order)

# Combine forecast with historical data
st.write("**Model RMSE Score:**", rmse)
st.write("##### Forecast Data (Next 30 Days)")

fig_table = plotly_table(forecast.sort_index(ascending=True).round(3))
fig_table.update_layout(height=220)
st.plotly_chart(fig_table, use_container_width=True)

# Combine forecast and historical data
forecast_full = pd.concat([rolling_price, forecast])

# Reset the index and fix column names
forecast_full = forecast_full.reset_index()

# Make sure the date column is named 'Date'
if 'ds' in forecast_full.columns:
    forecast_full.rename(columns={'ds': 'Date'}, inplace=True)
elif 'index' in forecast_full.columns:
    forecast_full.rename(columns={'index': 'Date'}, inplace=True)

# If 'Close' column doesn’t exist, use 'yhat' (forecasted price)
if 'Close' not in forecast_full.columns:
    if 'yhat' in forecast_full.columns:
        forecast_full['Close'] = forecast_full['yhat']
    else:
        forecast_full['Close'] = None

# Create fake Open, High, Low columns so chart doesn’t crash
for col in ['Open', 'High', 'Low']:
    if col not in forecast_full.columns:
        forecast_full[col] = forecast_full['Close']

# ✅ Now safely plot your chart
st.plotly_chart(Moving_average(forecast_full, num_period=150), use_container_width=True)

