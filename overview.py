# Overview.py

import streamlit as st
import yfinance as yf
import altair as alt
import pandas as pd

def run(company_name, ticker_symbol, start_date, end_date, chart_type, line_color):
    st.markdown("## 📊 Stock Viewer")
    st.subheader(f"{company_name} Stock - {chart_type} View")

    try:
        df = yf.download(ticker_symbol, start=start_date, end=end_date)
        if df.empty:
            st.warning("No data found for the selected range.")
            return

        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])

        overlays = []

        # --- Overlay Controls ---
        with st.expander("📐 Overlay Options"):
            show_sma = st.checkbox("Show SMA (Simple Moving Average)", value=False)
            sma_period = st.slider("SMA Period", 5, 100, 20)

            show_ema = st.checkbox("Show EMA (Exponential Moving Average)", value=False)
            ema_period = st.slider("EMA Period", 5, 100, 20)

            show_bollinger = st.checkbox("Show Bollinger Bands", value=False)
            boll_period = st.slider("Bollinger Period", 10, 50, 20)

            show_rsi = st.checkbox("Show RSI (Relative Strength Index)", value=False)
            rsi_period = st.slider("RSI Period", 5, 30, 14)

            show_macd = st.checkbox("Show MACD (Momentum)", value=False)

            show_volume = st.checkbox("Show Volume Bars", value=False)

        with st.expander("🗓️ Add Annotations"):
            annotate_dates = st.multiselect("Select Dates to Annotate", df['Date'].dt.strftime('%Y-%m-%d').tolist())

        # --- Compute Indicators ---
        if show_sma:
            df['SMA'] = df['Close'].rolling(window=sma_period).mean()
        if show_ema:
            df['EMA'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
        # --- Bollinger Bands Calculation ---
        if show_bollinger:
            df['BB_Middle'] = df['Close'].rolling(window=boll_period).mean()
            df['BB_Std'] = df['Close'].rolling(window=boll_period).std()
            df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
            df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
        # Computing RSI
        def compute_rsi(data, period):
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        if show_rsi:
            df['RSI'] = compute_rsi(df, rsi_period)
        # Computing MACD
        def compute_macd(data):
            ema12 = data['Close'].ewm(span=12, adjust=False).mean()
            ema26 = data['Close'].ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            return macd_line, signal_line
        if show_macd:
            df['MACD'], df['Signal'] = compute_macd(df)
        # --- Volume bar coloring (green = close > open) ---
        if show_volume:
            df['VolumeColor'] = df['Close'] > df['Open']
            df['VolumeColor'] = df['VolumeColor'].map({True: 'maroon', False: 'grey'})
        
        # --- Base Chart ---
        base = alt.Chart(df).encode(
            x=alt.X('Date:T', title='Date'),
            y=alt.Y('Close:Q', title='Close Price'),
            tooltip=['Date:T', 'Close:Q']
        ).add_params(
            alt.selection_interval(bind='scales', encodings=['x', 'y'], name="zoom")
        ).interactive()

        # Convert selected dates to datetime
        annotate_dt = pd.to_datetime(annotate_dates)
        annotations = alt.Chart(df[df['Date'].isin(annotate_dt)]).mark_rule(color='gold', strokeDash=[6, 3]).encode(
            x='Date:T',
            tooltip=[alt.Tooltip('Date:T', title='Annotation')]
        )

        # --- Main Chart Type ---
        if chart_type == "Line":
            chart = base.mark_line(color=line_color)
        elif chart_type == "Area":
            chart = base.mark_area(color=line_color, opacity=0.6)
        elif chart_type == "Bar":
            chart = base.mark_bar(color=line_color)
        elif chart_type == "Smoothed Line":
            chart = base.mark_line(color=line_color, interpolate='monotone')

        # --- Overlay Charts ---
        if show_sma:
            overlays.append(
                alt.Chart(df).mark_line(strokeDash=[4, 4], color='orange').encode(
                    x='Date:T', y='SMA:Q', tooltip=['Date:T', 'SMA:Q']
                )
            )
        if show_ema:
            overlays.append(
                alt.Chart(df).mark_line(strokeDash=[2, 2], color='purple').encode(
                    x='Date:T', y='EMA:Q', tooltip=['Date:T', 'EMA:Q']
                )
            )
        if show_bollinger:
            overlays.extend([
                alt.Chart(df).mark_line(color='green').encode(
                    x='Date:T', y='BB_Upper:Q', tooltip=['Date:T', 'BB_Upper:Q']
                ),
                alt.Chart(df).mark_line(color='green').encode(
                    x='Date:T', y='BB_Lower:Q', tooltip=['Date:T', 'BB_Lower:Q']
                )
            ])
        
        # --- Display Final Chart ---
        if overlays:
            combined_chart = chart
            for overlay in overlays:
                combined_chart += overlay

            if not annotations.data.empty:
                combined_chart += annotations

            st.altair_chart(combined_chart, use_container_width=True)
        else:
            st.altair_chart(chart, use_container_width=True)

        # Display RSI
        if show_rsi:
            st.markdown("### 📉 RSI Indicator")
            rsi_chart = alt.Chart(df).mark_line(color='crimson').encode(
                x=alt.X('Date:T', title='Date'),
                y=alt.Y('RSI:Q', title='RSI (0-100)', scale=alt.Scale(domain=[0, 100])),
                tooltip=['Date:T', 'RSI:Q']
            ).interactive()
            # Add horizontal lines at 70 and 30
            overbought = alt.Chart(pd.DataFrame({'y': [70]})).mark_rule(color='gray', strokeDash=[4,4]).encode(y='y')
            oversold = alt.Chart(pd.DataFrame({'y': [30]})).mark_rule(color='gray', strokeDash=[4,4]).encode(y='y')
            st.altair_chart(rsi_chart + overbought + oversold, use_container_width=True)

        # Display MACD
        if show_macd:
            st.markdown("### 📉 MACD Indicator")
            macd_chart = alt.Chart(df).mark_line(color='blue').encode(
                x=alt.X('Date:T', title='Date'),
                y=alt.Y('MACD:Q', title='MACD'),
                tooltip=['Date:T', 'MACD:Q']
            )
            signal_chart = alt.Chart(df).mark_line(color='orange', strokeDash=[3, 3]).encode(
                x='Date:T',
                y='Signal:Q',
                tooltip=['Date:T', 'Signal:Q']
            )
            st.altair_chart(macd_chart + signal_chart, use_container_width=True)

        # Display Volume chart
        if show_volume:
            st.markdown("### 📊 Volume Traded")
            volume_chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('Date:T', title='Date'),
                y=alt.Y('Volume:Q', title='Volume'),
                color=alt.Color('VolumeColor:N', scale=None),  # use raw color values
                tooltip=['Date:T', 'Volume:Q']
            ).interactive()

            st.altair_chart(volume_chart, use_container_width=True)

        st.download_button("📥 Download Data as CSV", data=df.to_csv(index=False), file_name=f"{ticker_symbol}_data.csv", mime='text/csv')

        # --- Raw Data ---
        with st.expander("📄 Raw Data"):
            st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"Error fetching data: {e}")
