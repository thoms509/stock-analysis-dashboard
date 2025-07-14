### bash: "jupyter notebook" ###

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

### define stock tickers and download historical data ### 

stock = ["AAPL", "XPEV", "TSLA", "GOOGL", "INTC", "MSFT"]

stocks = yf.download(stock, start="2010-01-01", end="2020-12-08", auto_adjust=True)
data = stocks.loc[:, "Close"].copy()

### plot raw closing prices over time ###

data.plot(figsize = (17, 8), fontsize = 18)
plt.style.use("seaborn-v0_8-darkgrid")
plt.show()

### show first few rows of closing price data ###
data.head()

### calculate daily percentage change for each stock ###
data.pct_change()

### drop missing values and get summary stats of daily returns ###
com = data.pct_change().dropna()
com.describe()

### isolate mean and standard deviation from the summary stats ###
sum = com.describe().T.loc[:, ["mean", "std"]]
sum

### annualize the mean and standard deviation of returns ###
sum["mean"] = sum["mean"] * 252
sum["std`"] = sum["std"] * np.sqrt(252)
sum

### plot annualized risk (std) vs return (mean) for each stock ###
sum.plot.scatter(x = "std", y = "mean", figsize = (12,8), s = 50, fontsize = 15)
for i in sum.index:
    plt.annotate(i, xy=(sum.loc[i, "std"]+0.002, sum.loc[i, "mean"]+0.002), size = 15)

### view initial stock prices (used for normalization) ###
data.iloc[0]

### normalize stock prices to compare growth (base = 100) ###
data.div(data.iloc[0]).mul(100)

### plot normalized price data (rebased to 100) ###
normData = data.div(data.iloc[0]).mul(100)
normData.plot(figsize = (17, 8), fontsize = 18)
plt.style.use("seaborn-v0_8-darkgrid")
plt.show()

