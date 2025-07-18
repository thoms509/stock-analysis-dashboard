import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.title("Stock Technical Analysis Dashboard")

@st.cache_data(show_spinner=False)
def get_sp500_info():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)  # yfinance compatibility
    display_name_to_ticker = {
        f"{row['Security']} ({row['Symbol']})": row["Symbol"] for _, row in df.iterrows()
    }
    return display_name_to_ticker

display_name_to_ticker = get_sp500_info()
display_names = list(display_name_to_ticker.keys())

default_tickers = ["AAPL", "TSLA", "MSFT"]
default_display_names = [name for name, ticker in display_name_to_ticker.items() if ticker in default_tickers]

selected_display_names = st.multiselect("Select Stocks", display_names, default=default_display_names)
selected_tickers = [display_name_to_ticker[name] for name in selected_display_names]

# Date inputs with limits
start_date = st.date_input(
    "Start Date",
    value=pd.to_datetime("2010-01-01"),
    min_value=pd.to_datetime("1990-01-01"),
    max_value=pd.Timestamp.today().date()
)

end_date = st.date_input(
    "End Date",
    value=pd.Timestamp.today().date(),
    min_value=start_date,
    max_value=pd.Timestamp.today().date()
)

if selected_tickers:
    data = yf.download(selected_tickers, start=start_date, end=end_date, auto_adjust=True)

    if data.empty:
        st.warning("No data found. Try a different selection or date range.")
        st.stop()

    # Handle single vs multiple tickers properly
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    else:
        data = data[["Close"]].rename(columns={"Close": selected_tickers[0]})

    # Normalize each ticker by its own first valid price individually
    norm_data = data.copy()
    for col in norm_data.columns:
        first_valid_idx = norm_data[col].first_valid_index()
        if first_valid_idx is not None:
            first_price = norm_data.loc[first_valid_idx, col]
            norm_data[col] = norm_data[col] / first_price * 100
        else:
            norm_data[col] = np.nan  

    st.subheader("Normalized Price Chart")
    st.line_chart(norm_data)

    # Returns and Risk stats Calculations
    returns = data.pct_change().dropna()
    stats = returns.describe().T[["mean", "std"]]
    stats["mean_annual"] = stats["mean"] * 252
    stats["std_annual"] = stats["std"] * np.sqrt(252)

    # Buy/Hold/Sell recommendation
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

    # Risk vs Return Scatter Plot
    color_map = {"Buy": "green", "Hold": "blue", "Sell": "red"}
    colors = stats["Recommendation"].map(color_map)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(stats["std_annual"], stats["mean_annual"], c=colors, s=50)

    for ticker in stats.index:
        display_name = next((name for name, t in display_name_to_ticker.items() if t == ticker), ticker)
        ax.annotate(display_name, xy=(stats.loc[ticker, "std_annual"], stats.loc[ticker, "mean_annual"] + 0.001))

    ax.set_xlabel("Annualized Risk (std)")
    ax.set_ylabel("Annualized Return (mean)")
    ax.set_title("Risk vs Return")
    ax.grid(True)
    st.pyplot(fig)
