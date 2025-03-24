# examination.py

import streamlit as st
import yfinance as yf
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from datetime import datetime
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import ks_2samp
from scipy import stats
import matplotlib.pyplot as plt
import warnings

def run(company_name, ticker_symbol, start_date, end_date):
    st.title("🔍 Time Series Examination")

    df = yf.download(ticker_symbol, start=start_date, end=end_date)
    if df.empty:
        st.warning("No data available for the selected date range.")
        return

    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    ts = df['Close'].dropna()

    st.subheader("1️⃣ Time Series Overview")
    with st.expander("ℹ️ About this section"):
        st.markdown("""
        This section shows the **raw time series of closing stock prices** over the selected date range.

        - Helps you **visually inspect** the trend, volatility, and any visible seasonality.
        - Useful for getting a **baseline feel** for the structure of the data before applying any transformations or tests.
        """)
    st.line_chart(ts)

    st.subheader("2️⃣ Classical Decomposition")
    with st.expander("ℹ️ About this section"):
        st.markdown("""
        **Classical decomposition** splits the series into 3 components:
        - **Trend**: long-term movement
        - **Seasonal**: repeated cycles (e.g., weekly, monthly)
        - **Residual**: random noise

        **Model used** here is additive:  
        `Observed = Trend + Seasonal + Residual`

        - Useful to detect patterns and understand whether your series is **structured** or mostly **random**.
        """)
    freq = st.slider("Select frequency (days)", min_value=5, max_value=60, value=30)
    try:
        result = seasonal_decompose(ts, model='additive', period=freq)
        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        result.observed.plot(ax=axes[0], title='Observed')
        result.trend.plot(ax=axes[1], title='Trend')
        result.seasonal.plot(ax=axes[2], title='Seasonal')
        result.resid.plot(ax=axes[3], title='Residual')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Decomposition failed: {e}")

    st.subheader("3️⃣ STL Decomposition (LOESS)")
    with st.expander("ℹ️ About this section"):
        st.markdown("""
        **STL (Seasonal-Trend decomposition using LOESS)** is a more flexible and robust method than classical decomposition.

        It splits the series into:
        - **Trend**
        - **Seasonal**
        - **Residual**

        **Why it's better:**
        - Handles **nonlinear trends**
        - More robust to **outliers**
        - Allows seasonal components to **change over time**

        Use it to get a clearer view of underlying structure in real-world, noisy data.
        """)
    try:
        stl = STL(ts, period=freq, robust=True)
        res = stl.fit()
        fig = res.plot()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"STL Decomposition failed: {e}")

    st.subheader("4️⃣ Stationarity Transformations")
    with st.expander("ℹ️ About this section"):
        st.markdown("""
        Stationarity means the **statistical properties** (mean, variance) of the series do **not change over time**.

        These transformations help in:
        - **Removing trend or variance instability**
        - Making the series **stationary**, which is often a requirement for forecasting models

        **Common transformations:**
        - **Log / Square Root**: stabilizes variance
        - **Differencing**: removes trend
        - **Power / Box-Cox**: adjusts skewness and distribution

        Apply one or a combination to make your series more model-ready.
        """)
    transformations = st.multiselect(
        "Apply transformations", ["Log", "Square Root", "First Difference", "Power (2)", "Box-Cox"]
    )
    ts_transformed = ts.copy()

    for t in transformations:
        try:
            if t == "Log":
                ts_transformed = np.log(ts_transformed.replace(0, np.nan)).dropna()
            elif t == "Square Root":
                ts_transformed = np.sqrt(ts_transformed)
            elif t == "First Difference":
                ts_transformed = ts_transformed.diff().dropna()
            elif t == "Power (2)":
                ts_transformed = ts_transformed ** 2
            elif t == "Box-Cox":
                ts_positive = ts_transformed[ts_transformed > 0].dropna()

                try:
                    boxcox_values, _ = stats.boxcox(ts_positive.values.flatten())
                    ts_transformed = pd.Series(boxcox_values, index=ts_positive.index)
                except Exception as e:
                    st.warning(f"Box-Cox transformation failed: {e}")

        except Exception as e:
            st.warning(f"{t} transformation failed: {e}")

    st.line_chart(ts_transformed)

    st.subheader("5️⃣ ADF Test")
    with st.expander("ℹ️ About this section"):
        st.markdown("""
        The **Augmented Dickey-Fuller (ADF) Test** checks for **unit root**, which is a sign of **non-stationarity**.

        - **Null Hypothesis (H₀)**: Series has a unit root → non-stationary
        - **Alternative (H₁)**: Series is stationary

        **How to interpret:**
        - p-value < 0.05 → ✅ likely **stationary**
        - p-value ≥ 0.05 → ❌ likely **non-stationary**

        Use ADF to confirm if further transformation is needed.
        """)
    adf = adfuller(ts_transformed)
    st.write(f"ADF Statistic: {adf[0]:.4f}")
    st.write(f"p-value: {adf[1]:.4f}")
    for k, v in adf[4].items():
        st.write(f"Critical Value {k}: {v:.4f}")
    # Interpretation
    if adf[1] < 0.05:
        st.success("✅ ADF Test Conclusion: The series is likely **stationary** (p < 0.05).")
    else:
        st.warning("❌ ADF Test Conclusion: The series is likely **non-stationary** (p ≥ 0.05).")

    st.subheader("6️⃣ KPSS Test")
    with st.expander("ℹ️ About this section"):
        st.markdown("""
        The **KPSS Test** is the **mirror opposite** of the ADF test.

        - **Null Hypothesis (H₀)**: Series is stationary
        - **Alternative (H₁)**: Series is non-stationary

        **How to interpret:**
        - p-value < 0.05 → ❌ likely **non-stationary**
        - p-value ≥ 0.05 → ✅ likely **stationary**

        Use it alongside ADF for stronger conclusions.
        """)
    try:
        kpss_stat, p_value, _, crit = kpss(ts_transformed, regression='c', nlags="auto")
        st.write(f"KPSS Statistic: {kpss_stat:.4f}")
        st.write(f"p-value: {p_value:.4f}")
        for k, v in crit.items():
            st.write(f"Critical Value {k}: {v}")
    except Exception as e:
        st.error(f"KPSS failed: {e}")
     # Interpretation
    if p_value < 0.05:
        st.warning("❌ KPSS Test Conclusion: The series is likely **non-stationary** (p < 0.05).")
    else:
        st.success("✅ KPSS Test Conclusion: The series is likely **stationary** (p ≥ 0.05).")

    st.markdown("### 📌 Overall Stationarity Summary")
    if adf[1] < 0.05 and p_value >= 0.05:
        st.success("✅ Both ADF and KPSS agree: The series is **stationary**.")
    elif adf[1] >= 0.05 and p_value < 0.05:
        st.warning("⚠️ Both ADF and KPSS agree: The series is **non-stationary**.")
    else:
        st.info("ℹ️ ADF and KPSS disagree. Consider further analysis or transformation.")


    st.subheader("7️⃣ Kolmogorov-Smirnov Test")
    with st.expander("ℹ️ About this section"):
        st.markdown("""
        The **KS Test** compares two samples to check if they come from the **same distribution**.

        Here, we compare:
        - Values at time `t`
        - Values at time `t+1`

        **KS Statistic** shows:
        - **Low value (< 0.05)** → Distributions are similar → ✅ stable
        - **High value (> 0.15)** → Distributions differ → ❌ potential shift

        This helps detect **distributional drift** or **structural change** in the series.
        """)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, _ = ks_2samp(ts_transformed[:-1], ts_transformed[1:])

        st.write(f"KS Statistic: {float(stat):.4f}")

        # Interpretation
        if stat < 0.05:
            st.success("✅ The distributions are very similar. No significant shift detected.")
        elif 0.05 <= stat <= 0.15:
            st.info("⚠️ Slight distributional shift observed. Might be acceptable depending on use case.")
        else:
            st.warning("❌ Significant distributional difference detected between consecutive periods.")

    except Exception as e:
        st.error(f"KS Test failed: {e}")

    st.subheader("8️⃣ White Noise & Random Walk")
    with st.expander("ℹ️ About this section"):
        st.markdown("""
        This section checks whether the time series behaves like **white noise** (pure randomness).

        - **White noise** = no structure, no predictability
        - **Random walk** = previous value + noise

        The **first difference** of the series is plotted:
        - If it looks random → series might be a **random walk**
        - If it shows pattern → more structure exists, and it’s **modelable**

        Helps you judge whether the data is **predictable** or just noise.
        """)
    noise = ts_transformed.diff().dropna()
    plt.figure(figsize=(10, 3))
    plt.plot(noise)
    plt.title("White Noise Approximation (First Difference)")
    st.pyplot(plt)