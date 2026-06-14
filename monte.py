import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf

#getting data
def get_data(stocks, start, end):
    stock_data = yf.download(stocks, start=start, end=end, group_by='column')
    stock_data = stock_data['Close']
    returns = stock_data.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

stockList = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
stocks = [stock + '.NS' for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

meanReturns, covMatrix = get_data(stocks, startDate, endDate)
weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)

mc_sims = 100
T = 100

meanM = np.full(shape = (T, len(weights)), fill_value=meanReturns)
meanM = meanM.T

portfoliosim = np.full(shape=(T, mc_sims), fill_value=0.0)
initial = 10000
for m in range(0, mc_sims):
    Z = np.random.normal(size=(T, len(weights)))
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanM + np.inner(L, Z)
    portfoliosim[:,m] = np.cumprod(np.inner(weights, dailyReturns.T) + 1)*initial


plt.plot(portfoliosim)
plt.ylabel("Portfolio Value")
plt.xlabel("Days")
plt.title("Monte carlo simulations")
plt.show()