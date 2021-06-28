from pandas_datareader import data as web
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
assets = ["MSFT","AAPL","AMZN","GOOG","FB","TSLA","GOOGL","NVDA","CMCSA","ADBE","INTC","CSCO","NFLX","PEP","AVGO","TMUS","TXN","COST","AMGN","CHTR","SBUX","AMAT","INTU","ISRG","AMD","BKNG","MU","LRCX","MDLZ","ADP","GILD","FISV","CSX","ATVI","MELI","ADSK","ADI","ILMN","NXPI","BIDU"]
total_companies = len(assets)
weights = np.array([0.2]*total_companies)
stockStartDate = '2013-01-01'
today = datetime.today().strftime("%Y-%m-%d")
df = pd.DataFrame()
for stock in assets:
    df[stock] = web.DataReader(stock, data_source="yahoo",start=stockStartDate,end=today)["Adj Close"]
returns = df.pct_change()
cov_matrix_annual = returns.cov()*252
cov_matrix_annual.to_csv("data/covariance.csv")
means = returns.mean(axis=0)*252
means.to_csv("data/averages.csv")