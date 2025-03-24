import streamlit as st
from datetime import date
import overview
import examination
import forecasting
import yfinance as yf
import home
from pathlib import Path

# --- Page Setup ---
st.set_page_config(page_title="SK Portfolio & Trading Tool",page_icon="Images/Logo.png", layout="wide")

# --- CSS for styling ---
st.markdown("""
    <style>
    .main-title {
        font-size: 2em;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        color: #1f77b4;
        border-bottom: 2px solid #ccc;
    }
    .footer {
        text-align: center;
        padding-top: 2rem;
        color: #888;
        font-size: 0.9em;
        border-top: 1px solid #ddd;
        margin-top: 2rem;
    }
    .sidebar-title {
        font-size: 1.3em;
        margin-bottom: 10px;
    }
    .sidebar-title {
        font-size: 1.3em;
        margin-bottom: 10px;
        margin-top: 20px;
        font-weight: bold;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Custom Navigation ---
st.sidebar.image("Images/Logo.png", width=100)
st.sidebar.markdown('<div class="sidebar-title">🧭 Navigation</div>', unsafe_allow_html=True)
pages = ["Home", "Stock Overview", "Stock Examination", "Stock Forecasting"]

default_index = 0  # "Home" as default
page = st.sidebar.selectbox("Select Page", pages, index=default_index)

# --- Initialize session state for inputs ---
if 'company_name' not in st.session_state:
    st.session_state.company_name = "Apple"
if 'chart_type' not in st.session_state:
    st.session_state.chart_type = "Line"
if 'date_range' not in st.session_state:
    st.session_state.date_range = [date(2010, 1, 1), date.today()]
if 'line_color' not in st.session_state:
    st.session_state.line_color = "#1f77b4"

# --- Main Logic ---
st.markdown("---")

if page == "Home":
    home.run()
elif page == "Stock Overview":
    # --- Top Title ---
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        st.image("Images/Logo.png", width=60)
    with col_title:
        st.markdown('<div class="main-title">Trading Evaluation and Forecasting Tool</div>', unsafe_allow_html=True)

    # Editable top panel (only here)
    col1, col2, col3 = st.columns([1, 1, 2])
    company_options = {
        "Apple": "AAPL",
        "Amazon": "AMZN",
        "Google": "GOOGL",
        "Microsoft": "MSFT",
        "Meta": "META",
        "Netflix": "NFLX",
        "Tesla": "TSLA",
        "NVIDIA": "NVDA",
        "JPMorgan": "JPM",
        "Goldman Sachs": "GS",
        "Walmart": "WMT",
        "Coca-Cola": "KO",
        "Pfizer": "PFE",
        "ExxonMobil": "XOM",
        "Ford": "F",
        "Boeing": "BA",
        "PayPal": "PYPL",
        "Airbnb": "ABNB",
        "IBM": "IBM"
    }
    st.session_state.company_name = col1.selectbox("Select Company", list(company_options.keys()))
    ticker_symbol = company_options[st.session_state.company_name]
    st.session_state.chart_type = col2.selectbox("Select Chart Type", ["Line", "Area", "Bar", "Smoothed Line"])
    selected_range = col3.date_input("Select Date Range", st.session_state.date_range)
    if selected_range:
        st.session_state.date_range = selected_range
    st.session_state.line_color = st.color_picker("Pick a chart color", st.session_state.line_color)
    company_name = st.session_state.company_name
    chart_type = st.session_state.chart_type
    line_color = st.session_state.line_color
    start_date, end_date = st.session_state.date_range

    company_map = {
        "Apple": "AAPL",
        "Amazon": "AMZN",
        "Google": "GOOGL",
        "Microsoft": "MSFT",
        "Meta": "META",
        "Netflix": "NFLX",
        "Tesla": "TSLA",
        "NVIDIA": "NVDA",
        "JPMorgan": "JPM",
        "Goldman Sachs": "GS",
        "Walmart": "WMT",
        "Coca-Cola": "KO",
        "Pfizer": "PFE",
        "ExxonMobil": "XOM",
        "Ford": "F",
        "Boeing": "BA",
        "PayPal": "PYPL",
        "Airbnb": "ABNB",
        "IBM": "IBM"
    }
    ticker_symbol = company_map[company_name]

    overview.run(company_name, ticker_symbol, start_date, end_date, chart_type, line_color)

else:
    company_name = st.session_state.company_name
    chart_type = st.session_state.chart_type
    line_color = st.session_state.line_color
    start_date, end_date = st.session_state.date_range

    company_map = {
        "Apple": "AAPL",
        "Amazon": "AMZN",
        "Google": "GOOGL",
        "Microsoft": "MSFT",
        "Meta": "META",
        "Netflix": "NFLX",
        "Tesla": "TSLA",
        "NVIDIA": "NVDA",
        "JPMorgan": "JPM",
        "Goldman Sachs": "GS",
        "Walmart": "WMT",
        "Coca-Cola": "KO",
        "Pfizer": "PFE",
        "ExxonMobil": "XOM",
        "Ford": "F",
        "Boeing": "BA",
        "PayPal": "PYPL",
        "Airbnb": "ABNB",
        "IBM": "IBM"
    }
    ticker_symbol = company_map[company_name]

    st.markdown(f"### 📌 {company_name} — {start_date} to {end_date}")

    if page == "Stock Examination":
        # --- Top Title ---
        col_logo, col_title = st.columns([1, 5])
        with col_logo:
            st.image("Images/Logo.png", width=60)
        with col_title:
            st.markdown('<div class="main-title">Trading Evaluation and Forecasting Tool</div>', unsafe_allow_html=True)
        examination.run(company_name, ticker_symbol, start_date, end_date)
    elif page == "Stock Forecasting":
        # --- Top Title ---
        col_logo, col_title = st.columns([1, 5])
        with col_logo:
            st.image("Images/Logo.png", width=60)
        with col_title:
            st.markdown('<div class="main-title">Trading Evaluation and Forecasting Tool</div>', unsafe_allow_html=True)
        forecasting.run(company_name, ticker_symbol, start_date, end_date)

# --- Bottom Footer ---
st.markdown('<div class="footer">An application made by <strong>Shubham Kandpal</strong></div>', unsafe_allow_html=True)
