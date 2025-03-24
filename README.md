# 📊 SK Portfolio & Trading Tool

A comprehensive and interactive **Streamlit application** for stock market analysis, technical examination, and time series forecasting using statistical and deep learning models. Built and maintained by **[Shubham Kandpal](https://www.linkedin.com/in/shubham-kandpal-035711165)**.

---

## 🚀 Features

### 🏠 Home (Portfolio Page)
- Personal portfolio introduction
- Summary of experience, skills, education, and projects

### 📈 Stock Overview
- Real-time stock data fetched from **Yahoo Finance**
- Multiple chart types: Line, Area, Bar, Smoothed Line
- Interactive technical indicators:
  - SMA, EMA, Bollinger Bands
  - RSI, MACD, Volume bars
- Custom color themes and date filtering
- Annotation options for key events

### 🔍 Stock Examination
- Classical & STL decomposition
- Time series transformations (Log, Square Root, Differencing, Box-Cox)
- Stationarity testing:
  - Augmented Dickey-Fuller (ADF)
  - KPSS Test
- Distribution shift detection using Kolmogorov-Smirnov (KS) Test
- White noise/random walk visualization

### 📉 Stock Forecasting
- Forecasting models:
  - **Prophet**
  - **ARMA / ARIMA / SARIMA**
  - **Exponential Smoothing & Holt-Winters**
  - **Linear Regression**
  - **LSTM (deep learning)**
- Preprocessing options (log, differencing, Box-Cox, STL seasonal adjustment)
- RMSE and MAPE evaluation metrics
- Forecast visualization with component breakdown

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit, Altair
- **Backend**: Python
- **Forecasting Libraries**: Prophet, statsmodels, scikit-learn, TensorFlow (LSTM)
- **Data**: Yahoo Finance (via `yfinance`)
- **ML Tools**: NumPy, SciPy, pandas
- **Visualization**: matplotlib, Altair

---
