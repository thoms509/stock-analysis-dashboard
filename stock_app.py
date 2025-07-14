import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.title("Stock Technical Analysis Dashboard")

### Function to fetch S&P 500 tickers ###
@st.cache_data(show_spinner=False)
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    # Replace '.' with '-' for yfinance compatibility
    tickers = df["Symbol"].str.replace('.', '-', regex=False).tolist()
    return tickers

### Inputs ###
stock_list = get_sp500_tickers()
selected_stocks = st.multiselect("Select stocks", stock_list, default=["AAPL", "TSLA", "MSFT"])
start_date = st.date_input("Start Date", pd.to_datetime("2010-01-01"))
end_date = st.date_input("End Date", pd.Timestamp.today().date())

### Data Download & Processing ###
if selected_stocks:
    data = yf.download(selected_stocks, start=start_date, end=end_date, auto_adjust=True)
    
    # Handle multi-index for multiple tickers
    if len(selected_stocks) > 1:
        data = data["Close"]
    else:
        data = data["Close"].to_frame(name=selected_stocks[0])

    ### Normalized Price Chart ###
    norm_data = data.div(data.iloc[0]).mul(100)
    st.subheader("Normalized Price Chart")
    st.line_chart(norm_data)

    ### Calculate returns and risk stats ###
    returns = data.pct_change().dropna()
    sum_stats = returns.describe().T[["mean", "std"]].copy()
    sum_stats["mean_annual"] = sum_stats["mean"] * 252
    sum_stats["std_annual"] = sum_stats["std"] * np.sqrt(252)

    ### Recommendation Logic ###
    def recommend_stock(row):
        if row['mean_annual'] > 0.1 and row['std_annual'] < 0.2:
            return "Buy"
        elif row['mean_annual'] < 0.05 and row['std_annual'] > 0.3:
            return "Sell"
        else:
            return "Hold"

    sum_stats["Recommendation"] = sum_stats.apply(recommend_stock, axis=1)

    ### Display stats with recommendations ###
    st.subheader("Annualized Mean, Std Dev, and Recommendation")
    st.dataframe(sum_stats[["mean_annual", "std_annual", "Recommendation"]])

    ### Summary counts ###
    st.write("### Recommendation Summary")
    st.write(f"Buy: {(sum_stats['Recommendation'] == 'Buy').sum()}")
    st.write(f"Hold: {(sum_stats['Recommendation'] == 'Hold').sum()}")
    st.write(f"Sell: {(sum_stats['Recommendation'] == 'Sell').sum()}")

    ### Risk vs Return Scatter Plot ###
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(sum_stats["std_annual"], sum_stats["mean_annual"], s=50)

    for ticker in sum_stats.index:
        ax.annotate(ticker, xy=(sum_stats.loc[ticker, "std_annual"], sum_stats.loc[ticker, "mean_annual"] + 0.001))

    ax.set_xlabel("Annualized Risk (std)")
    ax.set_ylabel("Annualized Return (mean)")
    ax.set_title("Risk vs Return")
    st.pyplot(fig)
