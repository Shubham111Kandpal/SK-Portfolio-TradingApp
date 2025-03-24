# forecasting.py

import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from statsmodels.tsa.seasonal import STL
from scipy import stats

def run(company_name, ticker_symbol, start_date, end_date):
    st.title("📈 Forecasting")

    model_type = st.selectbox(
        "Select Forecasting Model",
        [
            "-- Select Model --",
            "Prophet",
            "ARMA",
            "ARIMA",
            "SARIMA",
            "Exponential Smoothing",
            "Holt-Winters",
            "Linear Regression",
            "LSTM"
        ]
    )

    if model_type == "-- Select Model --":
        st.info("Please select a forecasting model from the dropdown above.")
        return

    forecast_days = st.slider("Select number of days to forecast", min_value=7, max_value=365, value=100)

    # --- Preprocessing Options ---
    with st.expander("⚙️ Preprocessing Options Before Forecasting"):
        apply_log = st.checkbox("Apply Log Transformation")
        apply_diff = st.checkbox("Apply First Difference")
        apply_sqrt = st.checkbox("Apply Square Root Transformation")
        apply_boxcox = st.checkbox("Apply Box-Cox Transformation")
        apply_seasonal_adjustment = st.checkbox("Apply STL Seasonal Adjustment (Period=30)")

    today = datetime.today().date()
    ten_years_ago = today - timedelta(days=365 * 10)

    df = yf.download(ticker_symbol, start=ten_years_ago, end=today)

    if df.empty or 'Close' not in df.columns:
        st.warning("No data available for forecasting.")
        return

    # Preprocess Data
    df = df[['Close']]
    df.columns = [f'Close_{ticker_symbol}']
    df = df.reset_index().rename(columns={'Date': 'ds', f'Close_{ticker_symbol}': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df.dropna(inplace=True)

    ts = df['y'].copy()

    try:
        if apply_log:
            ts = np.log(ts[ts > 0])

        if apply_diff:
            ts = ts.diff()

        if apply_sqrt:
            ts = np.sqrt(ts)

        if apply_boxcox:
            ts_positive = ts[ts > 0].dropna()
            boxcox_values, _ = stats.boxcox(ts_positive.values.flatten())
            ts = pd.Series(boxcox_values, index=ts_positive.index)

        if apply_seasonal_adjustment:
            stl = STL(ts.dropna(), period=30, robust=True)
            res = stl.fit()
            ts = ts - res.seasonal

        df['y'] = ts
        df.dropna(inplace=True)
    except Exception as e:
        st.warning(f"Preprocessing failed: {e}")

    # Function to plot the forecasting
    def evaluate_and_plot(df, forecast_df, label):
        actual = df['y'][-forecast_days:]
        predicted = forecast_df['yhat'][:forecast_days]
        fig, ax = plt.subplots()
        ax.plot(df['ds'], df['y'], label="Actual")
        ax.plot(forecast_df['ds'], forecast_df['yhat'], label="Forecast", color='green')
        ax.set_title(label)
        ax.legend()
        st.pyplot(fig)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = mean_absolute_percentage_error(actual, predicted) * 100
        st.markdown(f"**RMSE:** {rmse:.2f}")
        st.markdown(f"**MAPE:** {mape:.2f}%")
        with st.expander("Forecast Data"):
            st.dataframe(forecast_df, use_container_width=True)

    # --- Forecasting Models ---
    if model_type == "Prophet":
        st.subheader("🔮 Prophet Forecast")
        with st.expander("ℹ️ About this model"):
            st.markdown(""" 
            **Prophet** is an additive time series forecasting model developed by Facebook. It is designed to handle:
            - **Trend**
            - **Seasonality (daily, weekly, yearly)**
            - **Holiday effects**

            It works well with missing data, large outliers, and non-linear trends with change points.

            **Best Use Cases:**
            - Business and product time series
            - Retail, marketing, or stocks with clear seasonality

            **Assumptions:**
            - Additive structure of trend + seasonality + holidays
            """)
        try:             
            with st.spinner("Training Prophet model..."):
                model = Prophet()
                model.fit(df)

                future = model.make_future_dataframe(periods=forecast_days)
                forecast = model.predict(future)

            st.success("✅ Forecast generated!")

            # Plot forecast
            fig1 = model.plot(forecast)
            st.pyplot(fig1)

            # Components
            st.subheader("📊 Forecast Components")
            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)

            # Evaluation (last 30 days vs prediction)
            actual = df.set_index('ds').y[-30:]
            predicted = forecast.set_index('ds').yhat.loc[actual.index]

            rmse = np.sqrt(mean_squared_error(actual, predicted))
            mape = mean_absolute_percentage_error(actual, predicted) * 100

            st.markdown(f"**RMSE:** {rmse:.2f}")
            st.markdown(f"**MAPE:** {mape:.2f}%")

            with st.expander("Forecast Data"):
                st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days), use_container_width=True)

        except Exception as e:
            st.error(f"Forecasting failed: {e}")

    elif model_type == "ARIMA":
        st.subheader("🔢 ARIMA Forecast")
        with st.expander("ℹ️ About this model"):
            st.markdown(""" 
            **ARIMA (AutoRegressive Integrated Moving Average)** adds the “Integration” (I) component to ARMA.

            - It handles **non-stationary** data by differencing the series.
            - Still uses AR and MA like ARMA.

            **Best Use Cases:**
            - Price series that show trend or drift
            - Forecasting a single asset with strong momentum

            **Assumptions:**
            - Data becomes stationary **after differencing**
            """)
        try:
            with st.spinner("Training ARIMA model (1,1,1)..."):
                model = ARIMA(df['y'], order=(1, 1, 1))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=forecast_days)
            st.success("✅ Forecast generated!")
            future_dates = pd.date_range(start=df['ds'].iloc[-1] + timedelta(days=1), periods=forecast_days)
            forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast})

            # Evaluation
            actual = df['y'][-forecast_days:]
            predicted = model_fit.predict(start=len(df) - forecast_days, end=len(df) - 1, typ='levels')

            # Plot
            fig, ax = plt.subplots()
            ax.plot(df['ds'], df['y'], label="Actual")
            ax.plot(forecast_df['ds'], forecast_df['yhat'], label="Forecast", color='green')
            ax.set_title("ARIMA Forecast")
            ax.legend()
            st.pyplot(fig)

            rmse = np.sqrt(mean_squared_error(actual, predicted))
            mape = mean_absolute_percentage_error(actual, predicted) * 100

            st.markdown(f"**RMSE:** {rmse:.2f}")
            st.markdown(f"**MAPE:** {mape:.2f}%")

            with st.expander("Forecast Data"):
                st.dataframe(forecast_df, use_container_width=True)

        except Exception as e:
            st.error(f"ARIMA forecasting failed: {e}")

    elif model_type == "ARMA":
        st.subheader("🪛 ARMA Forecast")
        with st.expander("ℹ️ About this model"):
            st.markdown(""" 
            **ARMA (AutoRegressive Moving Average)** is used for modeling stationary time series.

            - The **AR (p)** part models momentum or autocorrelation (past values).
            - The **MA (q)** part models noise or error from past residuals.

            **Best Use Cases:**
            - Stationary stock price returns
            - Short-term price dynamics

            **Assumptions:**
            - Data is **stationary**
            - No trend or strong seasonality
            """)
        try:
            with st.spinner("Training ARMA model (2,1)..."):
                model = ARIMA(df['y'], order=(2, 0, 1))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=forecast_days)
            st.success("✅ Forecast generated!")
            future_dates = pd.date_range(start=df['ds'].iloc[-1] + timedelta(days=1), periods=forecast_days)
            forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast})

            # Evaluation
            actual = df['y'][-forecast_days:]
            predicted = model_fit.predict(start=len(df) - forecast_days, end=len(df) - 1, typ='levels')

            # Plot
            fig, ax = plt.subplots()
            ax.plot(df['ds'], df['y'], label="Actual")
            ax.plot(forecast_df['ds'], forecast_df['yhat'], label="Forecast", color='green')
            ax.set_title("ARMA Forecast")
            ax.legend()
            st.pyplot(fig)

            rmse = np.sqrt(mean_squared_error(actual, predicted))
            mape = mean_absolute_percentage_error(actual, predicted) * 100

            st.markdown(f"**RMSE:** {rmse:.2f}")
            st.markdown(f"**MAPE:** {mape:.2f}%")

            with st.expander("Forecast Data"):
                st.dataframe(forecast_df, use_container_width=True)

        except Exception as e:
            st.error(f"ARMA forecasting failed: {e}")

    elif model_type == "SARIMA":
        st.subheader("📈 SARIMA Forecast")
        with st.expander("ℹ️ About this model"):
            st.markdown(""" 
            **SARIMA (Seasonal ARIMA)** extends ARIMA by modeling seasonality explicitly.

            - Adds seasonal AR, MA, and differencing terms.
            - Good for **monthly/weekly/daily cycles**.

            **Best Use Cases:**
            - Stocks or indices with repeating seasonal behavior
            - Year-end effects, quarterly earnings cycles

            **Assumptions:**
            - Seasonal structure is **constant and known**
            """)
        try:
            with st.spinner("Training SARIMA model (1,1,1)x(1,1,1,12)..."):
                model = SARIMAX(df['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                model_fit = model.fit(disp=False)
                forecast = model_fit.forecast(steps=forecast_days)
            st.success("✅ Forecast generated!")
            future_dates = pd.date_range(start=df['ds'].iloc[-1] + timedelta(days=1), periods=forecast_days)
            forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast})
            evaluate_and_plot(df, forecast_df, "SARIMA Forecast")
        except Exception as e:
            st.error(f"SARIMA forecasting failed: {e}")

    elif model_type == "Exponential Smoothing":
        st.subheader("🧮 Exponential Smoothing Forecast")
        with st.expander("ℹ️ About this model"):
            st.markdown(""" 
            **Exponential Smoothing (ETS)** gives higher weight to recent observations.

            - Works well for short-term forecasting
            - Can include **trend** and **level**, but no seasonality here

            **Best Use Cases:**
            - Short-term forecasts with no clear seasonal component
            - Prices that fluctuate around a steady mean

            **Assumptions:**
            - Recent history is more important than the distant past
            """)
        try:
            with st.spinner("Training Exponential Smoothing model..."):
                model = ExponentialSmoothing(df['y'], trend='add', seasonal=None)
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=forecast_days)
            st.success("✅ Forecast generated!")
            future_dates = pd.date_range(start=df['ds'].iloc[-1] + timedelta(days=1), periods=forecast_days)
            forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast})
            evaluate_and_plot(df, forecast_df, "Exponential Smoothing Forecast")
        except Exception as e:
            st.error(f"Exponential Smoothing failed: {e}")

    elif model_type == "Holt-Winters":
        st.subheader("🌤️ Holt-Winters Forecast")
        with st.expander("ℹ️ About this model"):
            st.markdown(""" 
            **Holt-Winters (Triple Exponential Smoothing)** adds **seasonality** to Exponential Smoothing.

            - Supports level, trend, and seasonal components
            - Can be **additive** or **multiplicative**

            **Best Use Cases:**
            - Cyclical stocks (e.g., consumer goods, travel)
            - Monthly or quarterly patterns

            **Assumptions:**
            - Seasonality is **stable and predictable**
            """)
        try:
            with st.spinner("Training Holt-Winters model..."):
                model = ExponentialSmoothing(df['y'], trend='add', seasonal='add', seasonal_periods=12)
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=forecast_days)
            st.success("✅ Forecast generated!")
            future_dates = pd.date_range(start=df['ds'].iloc[-1] + timedelta(days=1), periods=forecast_days)
            forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast})
            evaluate_and_plot(df, forecast_df, "Holt-Winters Forecast")
        except Exception as e:
            st.error(f"Holt-Winters failed: {e}")

    elif model_type == "Linear Regression":
        st.subheader("📐 Linear Regression Forecast")
        with st.expander("ℹ️ About this model"):
            st.markdown(""" 
            **Linear Regression** fits a straight line (or polynomial) to the time index.

            - Assumes linear or polynomial growth trend
            - Doesn’t model noise, autocorrelation, or seasonality natively

            **Best Use Cases:**
            - Long-term trend extrapolation
            - Early-stage proof-of-concept forecasting

            **Assumptions:**
            - Stock movement follows a **linear trend over time**
            - No autocorrelation in residuals
            """)
        try:
            with st.spinner("Training Linear Regression model..."):
                df['t'] = range(len(df))
                X = df[['t']]
                y = df['y']
                model = LinearRegression().fit(X, y)
                future_t = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
                forecast = model.predict(future_t)
            st.success("✅ Forecast generated!")
            future_dates = pd.date_range(start=df['ds'].iloc[-1] + timedelta(days=1), periods=forecast_days)
            forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast})
            evaluate_and_plot(df, forecast_df, "Linear Regression Forecast")
        except Exception as e:
            st.error(f"Linear Regression forecasting failed: {e}")

    elif model_type == "LSTM":
        st.subheader("🧠 LSTM Forecast")
        with st.expander("ℹ️ About this model"):
            st.markdown("""
            **LSTM (Long Short-Term Memory)** is a type of Recurrent Neural Network (RNN) designed to learn from sequential data. It is particularly effective at capturing long-term dependencies in time series like stock prices.

            ### 🔧 Architecture Used:
            - **Window size:** 30 past time steps (days) are used as input
            - **1 LSTM Layer:** 32 units
            - **1 Dense Layer:** Outputs a single forecasted value
            - **Forecasting Horizon:** Iteratively predicts `n` future days by rolling the last 30-day window
            - **Normalization:** Input series is scaled using MinMaxScaler for faster and stable training

            ### ✅ Strengths:
            - Captures temporal patterns and lagged effects
            - Can model non-linear dependencies
            - Works well even if the data has no obvious trend/seasonality

            ### ⚠️ Limitations:
            - Requires careful tuning to avoid overfitting
            - Training time increases with window size and sequence depth
            - No built-in support for seasonality (unlike SARIMA/Prophet)

            This version is optimized for speed and simplicity to work well on limited hardware.
            """)
        try:
            with st.spinner("Training simplified LSTM model..."):
                window_size = 30
                data = df[['y']].copy()
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(data)

                X, y_lstm = [], []
                for i in range(window_size, len(scaled_data)):
                    X.append(scaled_data[i - window_size:i])
                    y_lstm.append(scaled_data[i])

                X, y_lstm = np.array(X), np.array(y_lstm)
                X = X.reshape((X.shape[0], X.shape[1], 1))

                model = Sequential([
                    LSTM(32, input_shape=(X.shape[1], 1)),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                model.fit(X, y_lstm, epochs=5, batch_size=32, verbose=0)

                # Forecast next n days
                forecast = []
                input_seq = scaled_data[-window_size:].reshape(1, window_size, 1)
                for _ in range(forecast_days):
                    pred = model.predict(input_seq, verbose=0)[0][0]
                    forecast.append(pred)
                    input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

                forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
                future_dates = pd.date_range(start=df['ds'].iloc[-1] + timedelta(days=1), periods=forecast_days)
                forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast})

            st.success("✅ Forecast generated!")
            evaluate_and_plot(df, forecast_df, "LSTM Forecast")

        except Exception as e:
            st.error(f"LSTM forecasting failed: {e}")
