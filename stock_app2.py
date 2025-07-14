import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.title("Stock Technical Analysis Dashboard")

### Fetch S&P 500 companies with tickers ###
@st.cache_data(show_spinner=False)
def get_sp500_info():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)  # yfinance compatibility
    # Create dict where keys = "Company Name (TICKER)", values = ticker
    display_name_to_ticker = {
        f"{row['Security']} ({row['Symbol']})": row["Symbol"] for _, row in df.iterrows()
    }
    return display_name_to_ticker

# Get dict for display
display_name_to_ticker = get_sp500_info()
display_names = list(display_name_to_ticker.keys())

# Dynamically pick some defaults (use ticker matching)
default_tickers = ["AAPL", "TSLA", "MSFT"]
default_display_names = [name for name, ticker in display_name_to_ticker.items() if ticker in default_tickers]

# Multi-select shows full company + ticker strings
selected_display_names = st.multiselect("Select Stocks", display_names, default=default_display_names)

# Map back to ticker symbols for processing
selected_tickers = [display_name_to_ticker[name] for name in selected_display_names]

# Date input
start_date = st.date_input("Start Date", pd.to_datetime("2010-01-01"))
end_date = st.date_input("End Date", pd.Timestamp.today().date())

if selected_tickers:
    data = yf.download(selected_tickers, start=start_date, end=end_date, auto_adjust=True)

    if data.empty:
        st.warning("No data found. Try a different selection or date range.")
        st.stop()

    if len(selected_tickers) > 1:
        data = data["Close"]
    else:
        data = data["Close"].to_frame(name=selected_tickers[0])

    # Normalize prices for chart
    norm_data = data.div(data.iloc[0]).mul(100)
    st.subheader("Normalized Price Chart")
    st.line_chart(norm_data)

    # Calculate returns and risk
    returns = data.pct_change().dropna()
    stats = returns.describe().T[["mean", "std"]]
    stats["mean_annual"] = stats["mean"] * 252
    stats["std_annual"] = stats["std"] * np.sqrt(252)

    # Recommendation logic
    def recommend(row):
        if row["mean_annual"] > 0.1 and row["std_annual"] < 0.2:
            return "Buy"
        elif row["mean_annual"] < 0.05 and row["std_annual"] > 0.3:
            return "Sell"
        else:
            return "Hold"

    stats["Recommendation"] = stats.apply(recommend, axis=1)

    st.subheader("Annualized Return, Risk, and Recommendation")
    st.dataframe(stats[["mean_annual", "std_annual", "Recommendation"]])

    st.write("### Recommendation Summary")
    st.write(f"Buy: {(stats['Recommendation'] == 'Buy').sum()}")
    st.write(f"Hold: {(stats['Recommendation'] == 'Hold').sum()}")
    st.write(f"Sell: {(stats['Recommendation'] == 'Sell').sum()}")

    # Scatter plot risk vs return
    color_map = {"Buy": "green", "Hold": "blue", "Sell": "red"}
    colors = stats["Recommendation"].map(color_map)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(stats["std_annual"], stats["mean_annual"], c=colors, s=50)

    for ticker in stats.index:
        # Get company + ticker display name for annotation
        display_name = next((name for name, t in display_name_to_ticker.items() if t == ticker), ticker)
        ax.annotate(display_name, xy=(stats.loc[ticker, "std_annual"], stats.loc[ticker, "mean_annual"] + 0.001))

    ax.set_xlabel("Annualized Risk (std)")
    ax.set_ylabel("Annualized Return (mean)")
    ax.set_title("Risk vs Return")
    ax.grid(True)
    st.pyplot(fig)
