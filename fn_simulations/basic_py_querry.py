import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
"""import warnings ; warnings.warn = lambda *args,**kwargs: None"""


"""This file uses YF to extract and save options data"""

stock_ID = 'GOOG'


def extract_maturity_date(contract_symbol, stock_ID = 'GOOG'):
    # goog is 4 letters
    N = len(stock_ID)
    date_str = contract_symbol[N:10]


    year = '20' + date_str[:2]
    month = date_str[2:4]
    day = date_str[4:6]


    maturity_date = f"{year}-{month}-{day}"

    return maturity_date

ticker = stock_ID
stock_data = yf.download(ticker, start='2023-01-01', end='2024-06-23')

plt.plot(stock_data['Adj Close'])
adj_closing = stock_data["Adj Close"]

adj_closing.to_csv("data/"+stock_ID+"_closing.csv")

goog = yf.Ticker(stock_ID)


exp_dates = goog.options



for i in range(len(exp_dates)):

    if i == 0:
        options_data = goog.option_chain(exp_dates[i])
        """Filter data down to where underlying currency is in USD"""

        calls = options_data.calls[options_data.calls['currency'] == 'USD']
        puts = options_data.puts[options_data.puts['currency'] == 'USD']

        puts_IDs = options_data.puts['contractSymbol']

        calls_IDs = options_data.calls['contractSymbol']

        puts_dates = [extract_maturity_date(u,stock_ID) for u in puts_IDs]
        calls_dates = [extract_maturity_date(u, stock_ID) for u in calls_IDs]

        calls['Maturity'] = calls_dates
        puts['Maturity'] = puts_dates





    else:

        local_options_data = goog.option_chain(exp_dates[i])
        local_calls = local_options_data.calls[local_options_data.calls['currency'] == 'USD']
        local_puts = local_options_data.puts[local_options_data.puts['currency'] == 'USD']

        puts_IDs = local_options_data.puts['contractSymbol']
        calls_IDs = local_options_data.calls['contractSymbol']

        puts_dates = [extract_maturity_date(u, stock_ID) for u in puts_IDs]
        calls_dates = [extract_maturity_date(u, stock_ID) for u in calls_IDs]

        local_calls['Maturity'] = calls_dates
        local_puts['Maturity'] = puts_dates

        calls = pd.concat([calls, local_calls], axis=0)
        puts = pd.concat([puts, local_puts], axis=0)











puts.to_csv('data/'+stock_ID+'_puts.csv')
calls.to_csv('data/'+stock_ID+'_calls.csv')
print(calls['Maturity'])
plt.show()